"""
02_build_panel.py
INPUT:
------
  data/interim/lar_cleaned_{year}.parquet  -- cleaned loan-level files (one per year)
  data/interim/lender_panel.parquet         -- LEI -> bank/nonbank crosswalk

OUTPUT:
-------
  data/final/panel_lei_year.parquet         -- main analysis panel
  data/final/panel_lei_year_balanced.parquet -- balanced subset (robustness)
  output/tables/panel_construction_log.csv  -- step-by-step row counts
  output/tables/panel_summary.csv           -- summary stats by year and type

WHY ANNUAL AND NOT QUARTERLY:
-----------------------------
The public HMDA Snapshot LAR records activity_year but not action_taken_date.
Quarterly granularity would require either (a) the HMDA Quarterly Filer
data, which only covers ~50 large institutions, or (b) institution-specific
MLAR files which retain dates. Both are described in the experimental
design as future robustness steps.

For the baseline panel we therefore aggregate annually. This means the
treatment indicator becomes "year >= 2022" rather than "quarter >= 2022Q2".
This is a real loss of precision around the treatment date, but it uses
fully public data and is defensible.

To switch to quarterly later: change TIME_UNIT below and add a quarter
column upstream in 01_clean_lar.py's add_quarter() function.

RUN:
----
  python code/02_build_panel.py
  python code/02_build_panel.py --min-loans 100  # stricter activity filter
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

YEARS = list(range(2018, 2025))

# Treatment date. Year 2022 is the first post-treatment year (Fed liftoff
# was March 2022). The treatment indicator is Post = 1 if year >= 2022.
# Sensitivity to alternative cutoffs (2021, 2023) is a documented
# robustness check.
TREATMENT_YEAR = 2022

# Lender activity filters. These are panel-level filters (i.e., they
# apply to the lender, not to individual loans).
DEFAULT_MIN_LOANS = 50          # per year, in years the lender is observed
DEFAULT_MIN_YEARS = 4           # at least 4 of 7 sample years observed

# Loan-purpose split for heterogeneity analysis (not the main outcome,
# but useful to carry through to the regression scripts).
PURCHASE_CODES = {1}
REFINANCE_CODES = {31, 32}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_panel")


@dataclass
class PanelLog:
    """Tracks the row counts at each panel-construction step."""
    steps: list[tuple[str, int]] = field(default_factory=list)

    def record(self, label: str, n: int) -> None:
        self.steps.append((label, n))
        log.info("  %-50s n=%s", label, f"{n:>10,d}")


# ---------------------------------------------------------------------------
# DTI parsing
# ---------------------------------------------------------------------------

def parse_dti(s: pd.Series) -> pd.Series:
    """Convert HMDA's debt_to_income_ratio strings to numeric.

    HMDA stores DTI as strings, with a mix of:
      - integer strings: "37", "42"
      - range bins (used for borrower privacy): "<20%", "20%-<30%",
        "30%-<36%", "36", "37", ..., "49", "50%-60%", ">60%", "Exempt", "NA"

    For analysis we collapse to numeric by taking the midpoint of bins
    and treating the open-ended bins as their boundary. "Exempt" and
    "NA" -> NaN.

    Why midpoints: most published HMDA papers (e.g., CFPB Annual Data
    Point analyses) use the same midpoint convention. It introduces
    measurement error but is unbiased on average within bins.
    """
    if s is None:
        return s
    s = s.astype("string").str.strip()

    bin_map = {
        "<20%": 15.0,
        "20%-<30%": 25.0,
        "30%-<36%": 33.0,
        "50%-60%": 55.0,
        ">60%": 65.0,
        "Exempt": np.nan,
        "NA": np.nan,
        "": np.nan,
    }
    out = s.map(bin_map)

    # Anything still unmapped should be a numeric string ("37", "42.5").
    # pd.to_numeric with errors='coerce' turns the rest into NaN.
    mask = out.isna() & s.notna() & ~s.isin(bin_map)
    out.loc[mask] = pd.to_numeric(s.loc[mask], errors="coerce")
    return out.astype(float)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_year(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one year of loan-level data to the LEI level.

    Each row of the output corresponds to one (lei, activity_year) cell.
    We compute:
      - origination counts (overall and by loan purpose)
      - origination dollar volume
      - average loan size, interest rate, DTI, income
      - share of FHA/VA loans (proxy for product mix)
      - geographic concentration (number of states served)

    Why these aggregations: the regression outcomes are ln(loan_count)
    and ln(loan_volume); the rest are time-varying lender controls or
    inputs to heterogeneity analysis.
    """
    df = df.copy()
    df["dti_numeric"] = parse_dti(df["debt_to_income_ratio"])
    # interest_rate and income can be "Exempt" or other strings in HMDA — coerce to float
    df["interest_rate"] = pd.to_numeric(df["interest_rate"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
    df["is_purchase"] = df["loan_purpose"].isin(PURCHASE_CODES).astype(int)
    df["is_refi"] = df["loan_purpose"].isin(REFINANCE_CODES).astype(int)
    # FHA = loan_type 2; VA = loan_type 3; FSA/RHS = loan_type 4
    df["is_govt_loan"] = df["loan_type"].isin([2, 3, 4]).astype(int)

    grouped = df.groupby(["lei", "activity_year"], observed=True)

    out = grouped.agg(
        loan_count=("loan_amount", "size"),
        loan_volume=("loan_amount", "sum"),
        avg_loan_size=("loan_amount", "mean"),
        median_loan_size=("loan_amount", "median"),
        avg_interest_rate=("interest_rate", "mean"),
        avg_dti=("dti_numeric", "mean"),
        avg_income=("income", "mean"),
        purchase_count=("is_purchase", "sum"),
        refi_count=("is_refi", "sum"),
        govt_loan_count=("is_govt_loan", "sum"),
        n_states=("state_code", "nunique"),
        n_counties=("county_code", "nunique"),
    ).reset_index()

    # Composition shares (more useful than counts for cross-lender comparison).
    out["purchase_share"] = out["purchase_count"] / out["loan_count"]
    out["refi_share"] = out["refi_count"] / out["loan_count"]
    out["govt_loan_share"] = out["govt_loan_count"] / out["loan_count"]

    return out


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_panel(min_loans: int, min_years: int,
                plog: PanelLog) -> pd.DataFrame:
    """Stack all years, merge lender classification, apply panel filters.

    Returns the unbalanced panel. The balanced version is built separately
    in build_balanced_panel().
    """
    # --- Load and stack each year's aggregated file ---
    yearly = []
    for year in YEARS:
        path = INTERIM_DIR / f"lar_cleaned_{year}.parquet"
        if not path.exists():
            log.warning("Missing %s, skipping", path.name)
            continue
        df = pd.read_parquet(path)
        agg = aggregate_year(df)
        yearly.append(agg)

    panel = pd.concat(yearly, ignore_index=True)
    plog.record("00. raw stacked LEI-year rows", len(panel))

    # --- Merge lender classification ---
    # Inner-merging the lender_panel ensures every panel row has a
    # bank/nonbank label. Rows whose LEI is not in the crosswalk are dropped.
    classification = pd.read_parquet(INTERIM_DIR / "lender_panel.parquet")
    panel = panel.merge(classification, on="lei", how="inner")
    plog.record("01. after merge with lender_panel.parquet", len(panel))

    # --- Drop excluded lender types (credit unions, mixed CFPB-supervised) ---
    # Why: the design narrows comparison to depository banks vs IMCs.
    # Credit unions and CFPB-only filers are documented exclusions.
    panel = panel[panel["lender_type"].isin(["bank", "nonbank"])]
    plog.record("02. drop excluded lender types", len(panel))

    # --- Build treatment indicators ---
    panel["nonbank"] = (panel["lender_type"] == "nonbank").astype(int)
    panel["post"] = (panel["activity_year"] >= TREATMENT_YEAR).astype(int)
    panel["nonbank_post"] = panel["nonbank"] * panel["post"]

    # --- Lender activity filter ---
    # Compute per-LEI: total years observed and minimum yearly loan count.
    # Drop lenders that don't meet both thresholds.
    activity = panel.groupby("lei").agg(
        years_observed=("activity_year", "nunique"),
        min_loans=("loan_count", "min"),
        total_loans=("loan_count", "sum"),
    ).reset_index()

    keep_leis = activity[
        (activity["years_observed"] >= min_years) &
        (activity["min_loans"] >= min_loans)
    ]["lei"]

    panel = panel[panel["lei"].isin(keep_leis)]
    plog.record(
        f"03. lenders w/ >={min_years} yrs and >={min_loans} loans/yr",
        len(panel)
    )

    # --- Final ordering and types ---
    panel = panel.sort_values(["lei", "activity_year"]).reset_index(drop=True)
    panel["activity_year"] = panel["activity_year"].astype(int)

    return panel


def build_balanced_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Restrict to LEIs observed in EVERY sample year.

    A balanced panel is more conservative (no compositional changes from
    entry/exit) but biases toward survivors. The experimental design uses
    the unbalanced panel as the main spec and the balanced panel as a
    robustness check.

    We do NOT use this as the main analysis panel. It is exported so the
    regression scripts can compare results.
    """
    n_years = panel["activity_year"].nunique()
    counts = panel.groupby("lei")["activity_year"].nunique()
    full_leis = counts[counts == n_years].index
    balanced = panel[panel["lei"].isin(full_leis)].copy()
    return balanced


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def make_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Year x lender-type summary: lender count, loan count, volume.

    This is the table to plot/print before any regression. If banks and
    nonbanks have wildly different baseline trajectories, the parallel
    trends assumption is suspect and the DiD is in trouble. Looking at
    this table is the first sanity check.
    """
    summary = (
        panel.groupby(["activity_year", "lender_type"], observed=True)
        .agg(
            n_lenders=("lei", "nunique"),
            total_loans=("loan_count", "sum"),
            total_volume_billions=("loan_volume", lambda x: x.sum() / 1e9),
            avg_loans_per_lender=("loan_count", "mean"),
        )
        .reset_index()
    )

    # Add nonbank share of originations within each year.
    pivot = summary.pivot(index="activity_year", columns="lender_type",
                          values="total_loans").fillna(0)
    pivot["nonbank_share"] = pivot["nonbank"] / (pivot["bank"] + pivot["nonbank"])

    summary = summary.merge(
        pivot["nonbank_share"].reset_index(),
        on="activity_year", how="left"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-loans", type=int, default=DEFAULT_MIN_LOANS,
                        help=f"Min loans/year per lender (default: {DEFAULT_MIN_LOANS})")
    parser.add_argument("--min-years", type=int, default=DEFAULT_MIN_YEARS,
                        help=f"Min years observed per lender (default: {DEFAULT_MIN_YEARS})")
    args = parser.parse_args(argv)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Building LEI-year panel")
    log.info("  min loans/year:    %d", args.min_loans)
    log.info("  min years observed: %d", args.min_years)
    log.info("  treatment year:    %d (post = activity_year >= %d)",
             TREATMENT_YEAR, TREATMENT_YEAR)
    log.info("=" * 60)

    plog = PanelLog()
    panel = build_panel(args.min_loans, args.min_years, plog)

    # --- Diagnostics on the final panel ---
    n_leis = panel["lei"].nunique()
    n_years = panel["activity_year"].nunique()
    log.info("Panel dimensions: %d unique LEIs x %d years = %d possible cells; %d filled (%.1f%%)",
             n_leis, n_years, n_leis * n_years, len(panel),
             100 * len(panel) / max(1, n_leis * n_years))

    type_counts = panel.drop_duplicates("lei")["lender_type"].value_counts()
    log.info("Lender type counts in final panel: %s",
             dict(type_counts))

    # --- Write outputs ---
    main_path = FINAL_DIR / "panel_lei_year.parquet"
    panel.to_parquet(main_path, index=False)
    log.info("Wrote main panel: %s (%s rows)", main_path.name,
             f"{len(panel):,d}")

    balanced = build_balanced_panel(panel)
    bal_path = FINAL_DIR / "panel_lei_year_balanced.parquet"
    balanced.to_parquet(bal_path, index=False)
    log.info("Wrote balanced panel: %s (%s rows, %d LEIs in all %d years)",
             bal_path.name, f"{len(balanced):,d}",
             balanced["lei"].nunique(), n_years)

    # --- Logs and summary ---
    log_df = pd.DataFrame(plog.steps, columns=["step", "n_rows"])
    log_df.to_csv(TABLES_DIR / "panel_construction_log.csv", index=False)

    summary = make_summary(panel)
    summary.to_csv(TABLES_DIR / "panel_summary.csv", index=False)
    log.info("Wrote panel construction log and year x type summary table.")

    # Print the summary table to stdout - the analyst should look at this
    # immediately to gut-check the data before running any regression.
    log.info("=" * 60)
    log.info("Panel summary (loans by year and lender type):")
    log.info("=" * 60)
    print(summary.to_string(index=False))

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
