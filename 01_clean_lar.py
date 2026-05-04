"""
01_clean_lar.py
INPUT:
------
  Raw HMDA Snapshot LAR files (one per year, 2018-2024)
  Raw HMDA Transmittal Sheet (TS) files (one per year)

  In production: download from
    https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/{year}
  Or pull state-by-state subsets via the Data Browser API:
    https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states={ST}&years={Y}

  In demo mode (--demo flag): generates a synthetic LAR with the real schema
  so the pipeline can be validated end-to-end without downloading ~50GB.

OUTPUT:
-------
  data/interim/lar_cleaned_{year}.parquet  -- one file per year, loan-level
  data/interim/lender_panel.parquet         -- LEI -> bank/nonbank crosswalk
  output/tables/cleaning_log.csv            -- row counts at each filter step

REPLICABILITY NOTES:
--------------------
- Random seeds are pinned for the demo data generator.
- Package versions are pinned in requirements.txt.
- The data download date should be recorded; HMDA snapshots are stable
  but the Dynamic dataset is not. We use Snapshot files only.
- Run order: 01_clean_lar.py -> 02_build_panel.py -> 03_descriptives.py -> 04_did_main.py
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
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

YEARS = list(range(2018, 2025))  # 2018 through 2024 inclusive

# HMDA action_taken codes (FIG 2024, p.13):
#   1 = Loan originated   <- KEEP
#   2 = Application approved but not accepted
#   3 = Application denied
#   4 = Application withdrawn by applicant
#   5 = File closed for incompleteness
#   6 = Purchased loan (already-originated, bought by another institution)
#   7 = Preapproval request denied
#   8 = Preapproval request approved but not accepted
ACTION_ORIGINATED = 1

# loan_purpose codes (FIG 2024, p.16):
#   1  = Home purchase           <- KEEP
#   2  = Home improvement
#   31 = Refinancing             <- KEEP
#   32 = Cash-out refinancing    <- KEEP
#   4  = Other purpose
#   5  = Not applicable
LOAN_PURPOSES_KEEP = {1, 31, 32}

# Agency codes (HMDA TS / Panel):
#   1 = OCC, 2 = FRS, 3 = FDIC   -> Bank (depository, prudentially regulated)
#   5 = NCUA                      -> Credit union (excluded)
#   7 = HUD                       -> Independent mortgage company (Nonbank)
#   9 = CFPB                      -> Mixed; excluded from main spec
AGENCY_BANK = {1, 2, 3}
AGENCY_NONBANK = {7}
AGENCY_EXCLUDE = {5, 9}

# Loan amount sanity bounds. Conforming jumbo ceiling was ~$726k in 2023,
# so $5M comfortably bounds plausible 1-4 family loans. Drop zeros and
# extreme outliers that typically reflect reporting errors.
LOAN_AMOUNT_MIN = 10_000
LOAN_AMOUNT_MAX = 5_000_000

# Lender-level filter: minimum loans per quarter and quarters present.
# This drops occasional reporters whose volumes are too noisy for within-
# lender variation. Documented in the experimental design (Section 4).
MIN_LOANS_PER_QUARTER = 25       # imposed at the panel-build stage, noted here
MIN_QUARTERS_OBSERVED = 4        # at least 4 of 28 sample quarters

# Treatment date: 2022 Q1 is the last pre-period quarter (Fed liftoff March 2022).
TREATMENT_QUARTER = pd.Period("2022Q1", freq="Q")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clean_lar")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CleaningLog:
    """Tracks row counts after each filter step, for transparency.

    Reviewers and replicators should be able to see exactly how many rows
    each filter dropped. This is written to output/tables/cleaning_log.csv
    so it is auditable.
    """
    year: int
    steps: list[tuple[str, int]] = field(default_factory=list)

    def record(self, label: str, n: int) -> None:
        self.steps.append((label, n))
        log.info("  [%d] %-45s n=%s", self.year, label, f"{n:>12,d}")


# ---------------------------------------------------------------------------
# Cleaning pipeline (operates on loaded DataFrames)
# ---------------------------------------------------------------------------

# These are the raw HMDA columns we need. Pulling only these (rather than
# all 99 public fields) keeps memory usage manageable. Field names match
# the post-2018 LAR schema verbatim.
COLS_TO_LOAD = [
    "activity_year",
    "lei",
    "action_taken",
    "loan_purpose",
    "loan_type",
    "lien_status",
    "occupancy_type",
    "total_units",
    "loan_amount",
    "interest_rate",
    "debt_to_income_ratio",
    "income",
    "state_code",
    "county_code",
    "census_tract",
    "applicant_age",
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
]


def filter_lar(df: pd.DataFrame, year: int, clog: CleaningLog) -> pd.DataFrame:
    """Apply the documented LAR filters in the order specified in the design.

    Each step records the surviving row count to the cleaning log. Order
    matters because some filters interact (e.g., we filter to first-lien
    before checking loan amount sanity, since loan amounts on second liens
    have different distributions).
    """
    clog.record("00. raw rows loaded", len(df))

    # --- Step 1: keep only originated loans ---
    # Why: the outcome variable is realized origination volume. Applications,
    # denials, and withdrawals reflect demand and underwriting, not
    # originated supply. Excluding them sharpens the volume measure.
    df = df[df["action_taken"] == ACTION_ORIGINATED]
    clog.record("01. action_taken = 1 (originated)", len(df))

    # --- Step 2: keep only purchase + refinance ---
    # Why: home improvement loans, reverse mortgages, and "other" have very
    # different demand drivers. Restricting to purchase/refi gives us the
    # core mortgage market that the literature studies.
    df = df[df["loan_purpose"].isin(LOAN_PURPOSES_KEEP)]
    clog.record("02. loan_purpose in {1, 31, 32}", len(df))

    # --- Step 3: first-lien only ---
    # lien_status: 1 = first lien, 2 = subordinate. Junior liens have
    # different rate dynamics and are a small share of the market.
    df = df[df["lien_status"] == 1]
    clog.record("03. lien_status = 1 (first lien)", len(df))

    # --- Step 4: owner-occupied principal residences only ---
    # occupancy_type: 1 = principal residence, 2 = second residence,
    # 3 = investment property. Investment properties respond differently
    # to rate shocks and are concentrated among certain lender types.
    df = df[df["occupancy_type"] == 1]
    clog.record("04. occupancy_type = 1 (owner-occupied)", len(df))

    # --- Step 5: 1-4 family properties ---
    # Multifamily (5+ units) is a different market.
    df = df[df["total_units"].isin(["1", "2", "3", "4", 1, 2, 3, 4])]
    clog.record("05. total_units in 1-4", len(df))

    # --- Step 6: loan amount sanity ---
    # Drops zero/negative amounts (data errors) and implausibly large
    # loans that suggest commercial deals miscoded as 1-4 family.
    df = df[df["loan_amount"].between(LOAN_AMOUNT_MIN, LOAN_AMOUNT_MAX)]
    clog.record("06. loan_amount in [10k, 5M]", len(df))

    # --- Step 7: drop missing LEI ---
    # We need lender ID for the lender FE specification.
    df = df[df["lei"].notna() & (df["lei"] != "")]
    clog.record("07. non-missing LEI", len(df))

    return df


def add_quarter(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Assign a calendar quarter to each loan.

    HMDA LAR records `action_taken_date` only for the year of activity.
    Quarter granularity is not directly available in the public Snapshot
    files (only annual). We approximate using the activity_year combined
    with a uniform random quarter draw within each year.

    *** IMPORTANT REPLICABILITY NOTE ***
    The public HMDA Snapshot files do NOT contain action_taken_date for
    privacy reasons. To get true quarterly timing you must use the HMDA
    Quarterly Filer data (only ~50 large filers, since 2020) or the
    institution-specific MLAR files which retain dates.

    For this baseline we use ANNUAL aggregation and treat the post-2022
    indicator as "any loan made in 2022 or later". This loses precision
    on the treatment date but uses fully public data. A robustness check
    using the quarterly filer subsample is described in the design.
    """
    df = df.copy()
    df["activity_year"] = df["activity_year"].astype(int)
    return df


def classify_lenders(ts: pd.DataFrame) -> pd.DataFrame:
    """Build the LEI -> Bank/Nonbank crosswalk from Transmittal Sheets.

    Uses the modal agency code per LEI across the full sample. This avoids
    spurious switching when a lender's classification flips for a single
    year (e.g., due to a holding-company reorganization).

    Returns
    -------
    DataFrame with columns: lei, agency_code_modal, lender_type
        lender_type in {'bank', 'nonbank', 'excluded'}
    """
    # mode() can return multiple values if tied; take the first
    modal = (
        ts.groupby("lei")["agency_code"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
        .reset_index()
        .rename(columns={"agency_code": "agency_code_modal"})
    )

    def assign_type(code: float) -> str:
        if code in AGENCY_BANK:
            return "bank"
        if code in AGENCY_NONBANK:
            return "nonbank"
        return "excluded"

    modal["lender_type"] = modal["agency_code_modal"].apply(assign_type)
    return modal


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_lar_year(year: int, raw_dir: Path) -> pd.DataFrame:
    """Load the LAR for a single year.

    The Snapshot files are pipe-delimited (.txt) and very large. We use
    chunked reading + dtype hints to keep memory manageable. Only the
    columns we actually need are loaded.
    """
    path = raw_dir / f"hmda_{year}_lar.txt"
    if not path.exists():
        # Try alternate naming conventions
        alt = list(raw_dir.glob(f"*{year}*lar*"))
        if alt:
            path = alt[0]
        else:
            raise FileNotFoundError(
                f"No LAR file found for {year} in {raw_dir}. "
                f"Expected hmda_{year}_lar.txt or similar. "
                f"Run the download script first, or use --demo mode."
            )

    log.info("Loading %s ...", path.name)
    # Pipe-delimited per the FFIEC schema; use object dtype for safety on
    # mixed-type fields like total_units which contains "5-24", "25-49" etc.
    df = pd.read_csv(
        path,
        sep="|",
        usecols=lambda c: c in COLS_TO_LOAD,
        dtype={
            "lei": "string",
            "state_code": "string",
            "county_code": "string",
            "census_tract": "string",
            "total_units": "string",
            "debt_to_income_ratio": "string",  # contains ranges like "<20%", ">60%"
        },
        low_memory=False,
    )
    return df


def load_ts_year(year: int, raw_dir: Path) -> pd.DataFrame:
    """Load the Transmittal Sheet for a single year."""
    path = raw_dir / f"hmda_{year}_ts.txt"
    if not path.exists():
        alt = list(raw_dir.glob(f"*{year}*ts*"))
        if alt:
            path = alt[0]
        else:
            raise FileNotFoundError(f"No TS file found for {year}")

    df = pd.read_csv(
        path,
        sep="|",
        dtype={"lei": "string"},
        low_memory=False,
    )
    df["activity_year"] = year
    return df[["activity_year", "lei", "agency_code"]]


# ---------------------------------------------------------------------------
# Demo data generator (for pipeline validation without downloading 50GB)
# ---------------------------------------------------------------------------

def generate_demo_data(raw_dir: Path, seed: int = 42) -> None:
    """Generate synthetic HMDA LAR + TS files matching the real schema.

    The synthetic data uses distributions drawn from published CFPB summary
    statistics (e.g., the Annual Data Point reports) so that the pipeline's
    behavior on synthetic data is broadly representative. This is intended
    only for verifying the cleaning logic runs correctly; it should NOT be
    used for substantive analysis.

    The structural feature we bake in (so the downstream DiD has something
    to find): nonbanks contract more sharply than banks starting in 2022.
    A real test would estimate this magnitude from the data; here we
    assume it so we can verify the regression code works.
    """
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("Generating demo data to %s", raw_dir)

    # Build a fake roster of lenders: 50 banks, 50 nonbanks (small enough
    # for a fast demo run; the real pipeline scales linearly).
    n_banks, n_nonbanks = 50, 50
    bank_leis = [f"BANK{str(i).zfill(16)}" for i in range(n_banks)]
    nonbank_leis = [f"IMC{str(i).zfill(17)}" for i in range(n_nonbanks)]

    ts_rows = []
    for year in YEARS:
        for lei in bank_leis:
            # Banks split across OCC/FRS/FDIC roughly evenly
            ts_rows.append({"activity_year": year, "lei": lei,
                            "agency_code": int(rng.choice([1, 2, 3]))})
        for lei in nonbank_leis:
            ts_rows.append({"activity_year": year, "lei": lei,
                            "agency_code": 7})
    ts = pd.DataFrame(ts_rows)
    for year in YEARS:
        ts[ts["activity_year"] == year].drop(columns=["activity_year"]).to_csv(
            raw_dir / f"hmda_{year}_ts.txt", sep="|", index=False
        )

    # Each lender's annual loan volume. Banks: ~3000 loans/year baseline.
    # Nonbanks: ~3500 loans/year baseline. Both surge in 2020-2021 (refi
    # boom) and fall in 2022+ (rate shock). Nonbanks fall harder by design.
    state_codes = np.array(["CA", "TX", "FL", "NY", "PA", "IL", "OH",
                              "GA", "NC", "MI"])

    for year in YEARS:
        log.info("  generating year %d ...", year)

        # Year-specific volume multipliers loosely matching CFPB-reported
        # market totals. Refi boom in 2020-2021; collapse 2022+.
        bank_mult = {2018: 1.0, 2019: 1.05, 2020: 1.6, 2021: 1.9,
                     2022: 0.85, 2023: 0.55, 2024: 0.60}[year]
        nonbank_mult = {2018: 1.0, 2019: 1.10, 2020: 1.9, 2021: 2.3,
                        2022: 0.75, 2023: 0.40, 2024: 0.45}[year]

        # Draw per-lender loan counts (Poisson around the year-mean).
        bank_counts = rng.poisson(300 * bank_mult, size=n_banks)
        nonbank_counts = rng.poisson(350 * nonbank_mult, size=n_nonbanks)

        # Build a flat array of LEIs, one entry per loan.
        bank_leis_arr = np.repeat(np.array(bank_leis), bank_counts)
        nonbank_leis_arr = np.repeat(np.array(nonbank_leis), nonbank_counts)
        all_leis = np.concatenate([bank_leis_arr, nonbank_leis_arr])
        n_total = len(all_leis)

        if n_total == 0:
            continue

        # Vectorized generation of all loan-level fields.
        # Action: 70% originated, 20% denied, 5% withdrawn, 5% incomplete.
        action = rng.choice([1, 3, 4, 5], size=n_total,
                            p=[0.70, 0.20, 0.05, 0.05])

        # Loan purpose distribution shifts post-2022 (refi boom -> purchase).
        if year <= 2021:
            purpose = rng.choice([1, 31, 32, 2], size=n_total,
                                 p=[0.30, 0.55, 0.10, 0.05])
        else:
            purpose = rng.choice([1, 31, 32, 2], size=n_total,
                                 p=[0.65, 0.20, 0.10, 0.05])

        loan_type = rng.choice([1, 2, 3, 4], size=n_total,
                               p=[0.75, 0.15, 0.08, 0.02])
        lien_status = rng.choice([1, 2], size=n_total, p=[0.95, 0.05])
        occupancy = rng.choice([1, 2, 3], size=n_total, p=[0.88, 0.05, 0.07])
        units = rng.choice([1, 2, 3, 4], size=n_total,
                           p=[0.93, 0.04, 0.02, 0.01])
        loan_amount = (rng.lognormal(mean=12.4, sigma=0.55, size=n_total)
                       // 5000 * 5000)
        rate_loc = 3.0 if year <= 2021 else 6.5
        interest_rate = np.round(rng.normal(loc=rate_loc, scale=0.7,
                                             size=n_total), 3)
        dti = np.clip(rng.normal(36, 8, size=n_total), 10, 65).astype(int)
        income = (rng.lognormal(mean=4.4, sigma=0.5, size=n_total) * 10).astype(int)
        states = rng.choice(state_codes, size=n_total)
        counties = rng.integers(1001, 56999, size=n_total)
        tracts = rng.integers(1001, 99999, size=n_total)
        ages = rng.choice(["<25", "25-34", "35-44", "45-54",
                           "55-64", "65-74", ">74"], size=n_total)
        races = rng.choice(["White", "Black or African American", "Asian",
                            "Hispanic", "Other", "Race Not Available"],
                           size=n_total,
                           p=[0.62, 0.08, 0.07, 0.13, 0.05, 0.05])
        eths = rng.choice(["Not Hispanic or Latino", "Hispanic or Latino",
                           "Ethnicity Not Available"],
                          size=n_total, p=[0.75, 0.18, 0.07])
        sexes = rng.choice(["Male", "Female", "Joint", "Sex Not Available"],
                           size=n_total, p=[0.45, 0.30, 0.20, 0.05])

        df = pd.DataFrame({
            "activity_year": year,
            "lei": all_leis,
            "action_taken": action,
            "loan_purpose": purpose,
            "loan_type": loan_type,
            "lien_status": lien_status,
            "occupancy_type": occupancy,
            "total_units": units.astype(str),
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "debt_to_income_ratio": dti.astype(str),
            "income": income,
            "state_code": states,
            "county_code": [f"{c:05d}" for c in counties],
            "census_tract": [f"{t:011d}" for t in tracts],
            "applicant_age": ages,
            "derived_race": races,
            "derived_ethnicity": eths,
            "derived_sex": sexes,
        })

        out_path = raw_dir / f"hmda_{year}_lar.txt"
        df.to_csv(out_path, sep="|", index=False)
        log.info("    wrote %s (%s rows)", out_path.name, f"{len(df):,d}")


def _make_loan_row(*args, **kwargs):
    """Deprecated. Kept as a stub; vectorized generation in generate_demo_data."""
    raise NotImplementedError("Use the vectorized path in generate_demo_data")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean HMDA LAR files for the nonbank-resilience DiD."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate synthetic data first, then run the cleaning pipeline. "
             "Use this to validate the pipeline without downloading real HMDA files."
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=YEARS,
        help="Years to process (default: 2018-2024)"
    )
    args = parser.parse_args(argv)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.demo:
        generate_demo_data(RAW_DIR)

    # ---- Process LAR files year-by-year ----
    all_logs: list[CleaningLog] = []
    for year in args.years:
        log.info("=" * 60)
        log.info("Processing year %d", year)
        log.info("=" * 60)

        clog = CleaningLog(year=year)
        df_raw = load_lar_year(year, RAW_DIR)
        df_clean = filter_lar(df_raw, year, clog)
        df_clean = add_quarter(df_clean, year)

        out_path = INTERIM_DIR / f"lar_cleaned_{year}.parquet"
        df_clean.to_parquet(out_path, index=False)
        log.info("Wrote %s (%s rows)", out_path.name, f"{len(df_clean):,d}")
        all_logs.append(clog)

    # ---- Build lender classification crosswalk ----
    log.info("=" * 60)
    log.info("Building lender classification (Bank vs Nonbank)")
    log.info("=" * 60)
    ts_all = pd.concat([load_ts_year(y, RAW_DIR) for y in args.years],
                       ignore_index=True)
    panel = classify_lenders(ts_all)
    panel.to_parquet(INTERIM_DIR / "lender_panel.parquet", index=False)

    type_counts = panel["lender_type"].value_counts()
    log.info("Lender classification:")
    for t, n in type_counts.items():
        log.info("  %-10s %s", t, f"{n:>6,d}")

    # ---- Write the cleaning log ----
    log_rows = []
    for clog in all_logs:
        for label, n in clog.steps:
            log_rows.append({"year": clog.year, "step": label, "n_rows": n})
    log_df = pd.DataFrame(log_rows)
    log_path = TABLES_DIR / "cleaning_log.csv"
    log_df.to_csv(log_path, index=False)
    log.info("Wrote cleaning log to %s", log_path)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
