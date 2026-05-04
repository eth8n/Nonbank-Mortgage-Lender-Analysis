"""
04_did_main.py

INPUT:
------
  data/final/panel_lei_year.parquet

OUTPUTS:
--------
  output/tables/table04_did_baseline.csv    -- main regression table
  output/tables/table05_did_heterogeneity.csv -- purchase vs refi split
  output/tables/table06_did_robustness.csv  -- alt treatment dates, balanced panel

RUN:
----
  python 04_did_main.py
  python 04_did_main.py --alt-cutoff 2023   # robustness: later treatment date
  python 04_did_main.py --balanced          # robustness: balanced panel only
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_DIR    = PROJECT_ROOT / "data" / "final"
TABLES_DIR   = PROJECT_ROOT / "output" / "tables"

TREATMENT_YEAR = 2022          # baseline: first post year
MIN_SUBGROUP_LOANS = 10        # minimum loans to include in purchase/refi split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("did_main")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_panel(panel: pd.DataFrame, treatment_year: int) -> pd.DataFrame:
    """Add log outcomes and re-code treatment indicators for a given cutoff.

    We take logs of the count and volume outcomes because:
      (a) Both are right-skewed — the log transformation improves linearity
          and makes the FE residuals closer to normal.
      (b) The log difference has an elasticity interpretation, which is
          more natural for cross-sectional comparisons of lenders at
          very different scales.
      (c) Standard in the HMDA/mortgage-origination literature
          (Buchak et al. 2018, Fuster et al. 2019).

    Controls are standardized (zero mean, unit SD within the panel) so that
    coefficient magnitudes are comparable across variables with different
    units. The treatment interaction is NOT standardized — β retains its
    log-point interpretation.
    """
    df = panel.copy()

    # Log outcomes (guarded against zeros, which shouldn't exist post-filter)
    df["ln_loan_count"]  = np.log(df["loan_count"].clip(lower=1))
    df["ln_loan_volume"] = np.log(df["loan_volume"].clip(lower=1))

    # Recode post/treatment indicators for the given cutoff year
    df["post"]        = (df["activity_year"] >= treatment_year).astype(int)
    df["nonbank_post"] = df["nonbank"] * df["post"]

    # Time-varying controls: standardize to zero mean / unit SD
    # (computed over the full panel, not separately by group)
    for col in ["avg_dti", "avg_income", "govt_loan_share"]:
        mu  = df[col].mean()
        sig = df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sig if sig > 0 else 0.0

    # Purchase / refi log counts for heterogeneity specs
    # Use clip(1) to handle lenders with zero loans in a subtype in a given year
    df["ln_purchase_count"] = np.log(df["purchase_count"].clip(lower=1))
    df["ln_refi_count"]     = np.log(df["refi_count"].clip(lower=1))

    # String IDs for pyfixest factor handling
    df["lei_fe"]  = df["lei"].astype(str)
    df["year_fe"] = df["activity_year"].astype(str)

    return df


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

CONTROLS_Z = "avg_dti_z + avg_income_z + govt_loan_share_z"

def run_twfe(df: pd.DataFrame,
             outcome: str,
             controls: bool = True,
             label: str = "") -> pf.Feols:
    """Fit one TWFE DiD regression using pyfixest.

    Formula: outcome ~ nonbank_post [+ controls] | lei_fe + year_fe,
             vcov = {'CRV1': 'lei_fe'}   (cluster by lender)

    The lender FE absorbs the Nonbank_i main effect; the year FE absorbs
    the Post_t main effect. The coefficient on nonbank_post is β.

    We suppress pyfixest's informational prints for clean logging.
    """
    rhs = "nonbank_post"
    if controls:
        rhs += f" + {CONTROLS_Z}"

    formula = f"{outcome} ~ {rhs} | lei_fe + year_fe"
    log.info("  Fitting: %s  [%s]", formula, label or outcome)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pf.feols(formula, data=df, vcov={"CRV1": "lei_fe"})

    return fit


def extract_did_row(fit: pf.Feols, label: str, n_lenders: int) -> dict:
    """Pull the key statistics from a fitted Feols object into a dict row.

    We extract:
      - beta:      point estimate on nonbank_post
      - se:        clustered SE
      - t_stat:    t-statistic
      - p_value:   p-value
      - ci_lo/hi:  95% confidence interval
      - r2_within: within-R² (variation explained after removing FEs)
      - n_obs:     observations used in estimation
      - n_lenders: unique lenders (passed in from calling scope)
    """
    # pyfixest 0.50: tidy() uses coefficient name as index ("Coefficient")
    tidy  = fit.tidy()
    row   = tidy.loc["nonbank_post"]

    beta  = float(fit.coef()["nonbank_post"])
    se    = float(fit.se()["nonbank_post"])
    tstat = float(fit.tstat()["nonbank_post"])
    pval  = float(fit.pvalue()["nonbank_post"])
    ci    = fit.confint()
    ci_lo = float(ci.loc["nonbank_post", "2.5%"])
    ci_hi = float(ci.loc["nonbank_post", "97.5%"])

    # pyfixest 0.50: get_performance() returns None but the values are stored
    # as private attributes on the Feols object.
    try:
        resids    = fit.resid()
        n_obs     = len(resids)
        r2_within = float(getattr(fit, "_r2_within", np.nan))
        r2        = float(getattr(fit, "_r2",        np.nan))
    except Exception:
        n_obs     = np.nan
        r2_within = np.nan
        r2        = np.nan

    def _fmt(v):
        return round(v, 4) if not np.isnan(v) else np.nan

    return {
        "spec":       label,
        "beta":       round(beta,  4),
        "se":         round(se,    4),
        "t_stat":     round(tstat, 3),
        "p_value":    round(pval,  4),
        "ci_95_lo":   round(ci_lo, 4),
        "ci_95_hi":   round(ci_hi, 4),
        "r2":         _fmt(r2),
        "r2_within":  _fmt(r2_within),
        "n_obs":      n_obs,
        "n_lenders":  n_lenders,
    }


def significance_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ---------------------------------------------------------------------------
# Main regression table (Table 4)
# ---------------------------------------------------------------------------

def run_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Four outcome columns, with and without controls.

    Columns of Table 4 (following standard DiD paper formatting):
      (1) ln(loan_count),  no controls
      (2) ln(loan_count),  with controls     <- main spec
      (3) ln(loan_volume), with controls
      (4) avg_loan_size,   with controls     <- composition check
      (5) avg_interest_rate, with controls   <- mechanism check

    All specifications: lender FE + year FE, SE clustered by lender.
    """
    log.info("=" * 60)
    log.info("Table 4: Baseline TWFE DiD")
    log.info("=" * 60)

    specs = [
        ("ln_loan_count",      False, "(1) ln(loan_count) — no controls"),
        ("ln_loan_count",      True,  "(2) ln(loan_count) — with controls"),
        ("ln_loan_volume",     True,  "(3) ln(loan_volume)"),
        ("avg_loan_size",      True,  "(4) avg_loan_size"),
        ("avg_interest_rate",  True,  "(5) avg_interest_rate"),
    ]

    rows = []
    for outcome, ctrl, label in specs:
        fit = run_twfe(df, outcome, controls=ctrl, label=label)
        nl  = df["lei"].nunique()
        row = extract_did_row(fit, label, nl)
        rows.append(row)
        stars = significance_stars(row["p_value"])
        log.info("    β = %+.4f%s  (SE=%.4f, p=%.3f)  [%s]",
                 row["beta"], stars, row["se"], row["p_value"], label)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Heterogeneity table (Table 5): purchase vs refi
# ---------------------------------------------------------------------------

def run_heterogeneity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase vs refinance split.

    Motivation (from experimental design, Section 5):
      Refinances are highly rate-sensitive; purchases are less so.
      If β is more negative for refis, it supports the funding-model
      channel: higher rates crater refi demand, and nonbanks — who
      were more dependent on refi volume — shed capacity faster.
      If β is similar for purchases and refis, the contraction is
      more structural/supply-side.

    We restrict each sub-sample to lender-years with at least
    MIN_SUBGROUP_LOANS originations of that type to avoid noisy
    observations from lenders that barely participated in a segment.
    """
    log.info("=" * 60)
    log.info("Table 5: Heterogeneity by loan purpose")
    log.info("=" * 60)

    rows = []
    for outcome, min_col, label in [
        ("ln_purchase_count", "purchase_count", "Purchase loans"),
        ("ln_refi_count",     "refi_count",     "Refi loans"),
    ]:
        sub = df[df[min_col] >= MIN_SUBGROUP_LOANS].copy()
        fit = run_twfe(sub, outcome, controls=True, label=label)
        nl  = sub["lei"].nunique()
        row = extract_did_row(fit, label, nl)
        rows.append(row)
        stars = significance_stars(row["p_value"])
        log.info("    β = %+.4f%s  (SE=%.4f, p=%.3f)  [%s]",
                 row["beta"], stars, row["se"], row["p_value"], label)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustness table (Table 6)
# ---------------------------------------------------------------------------

def run_robustness_table(panel_raw: pd.DataFrame,
                         treatment_year: int) -> pd.DataFrame:
    """Alternative samples and treatment date.

    Checks:
      R1: Baseline (repeated for reference)
      R2: Balanced panel only (no entry/exit)
      R3: Alt treatment cutoff = 2023
      R4: Strict activity filter — min 100 loans/yr (vs baseline 50)
    """
    log.info("=" * 60)
    log.info("Table 6: Robustness checks")
    log.info("=" * 60)

    rows = []

    # R1: Baseline
    df = prepare_panel(panel_raw, treatment_year)
    fit = run_twfe(df, "ln_loan_count", controls=True, label="R1 Baseline")
    rows.append(extract_did_row(fit, "R1: Baseline", df["lei"].nunique()))

    # R2: Balanced panel
    balanced = pd.read_parquet(FINAL_DIR / "panel_lei_year_balanced.parquet")
    df_bal   = prepare_panel(balanced, treatment_year)
    fit = run_twfe(df_bal, "ln_loan_count", controls=True,
                   label="R2 Balanced panel")
    rows.append(extract_did_row(fit, "R2: Balanced panel", df_bal["lei"].nunique()))

    # R3: Alt treatment cutoff = 2023
    df_alt = prepare_panel(panel_raw, 2023)
    fit = run_twfe(df_alt, "ln_loan_count", controls=True,
                   label="R3 Post=2023+")
    rows.append(extract_did_row(fit, "R3: Post = year≥2023", df_alt["lei"].nunique()))

    # R4: Stricter lender activity filter (≥100 loans/yr minimum)
    # Recompute from the panel: keep only LEIs where every observed year
    # had at least 100 loans.
    activity = panel_raw.groupby("lei")["loan_count"].min()
    keep_leis = activity[activity >= 100].index
    panel_strict = panel_raw[panel_raw["lei"].isin(keep_leis)]
    df_strict = prepare_panel(panel_strict, treatment_year)
    fit = run_twfe(df_strict, "ln_loan_count", controls=True,
                   label="R4 Min 100 loans/yr")
    rows.append(extract_did_row(fit, "R4: Min 100 loans/yr", df_strict["lei"].nunique()))

    for r in rows:
        stars = significance_stars(r["p_value"])
        log.info("    β = %+.4f%s  (SE=%.4f, p=%.3f)  [%s]",
                 r["beta"], stars, r["se"], r["p_value"], r["spec"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame, title: str) -> None:
    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)
    display = df.copy()
    display["stars"] = display["p_value"].apply(significance_stars)
    display["beta_str"] = (display["beta"].apply(lambda x: f"{x:+.4f}")
                           + display["stars"])
    display["ci_str"] = (display["ci_95_lo"].apply(lambda x: f"[{x:.4f},")
                         + display["ci_95_hi"].apply(lambda x: f" {x:.4f}]"))
    cols = ["spec", "beta_str", "se", "ci_str", "p_value",
            "r2", "r2_within", "n_obs", "n_lenders"]
    print(display[cols].to_string(index=False))
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Baseline TWFE DiD regressions for nonbank-resilience study."
    )
    parser.add_argument(
        "--alt-cutoff", type=int, default=TREATMENT_YEAR,
        help=f"Treatment year cutoff for baseline (default: {TREATMENT_YEAR})"
    )
    parser.add_argument(
        "--balanced", action="store_true",
        help="Use the balanced panel (lenders observed in all 7 years)"
    )
    args = parser.parse_args(argv)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load panel ----
    panel_path = FINAL_DIR / "panel_lei_year.parquet"
    if not panel_path.exists():
        log.error("Panel not found. Run 02_build_panel.py first.")
        return 1

    panel_raw = pd.read_parquet(panel_path)
    if args.balanced:
        bal_path = FINAL_DIR / "panel_lei_year_balanced.parquet"
        panel_raw = pd.read_parquet(bal_path)
        log.info("Using balanced panel (%d rows, %d LEIs)",
                 len(panel_raw), panel_raw["lei"].nunique())
    else:
        log.info("Using unbalanced panel (%d rows, %d LEIs)",
                 len(panel_raw), panel_raw["lei"].nunique())

    treatment_year = args.alt_cutoff
    log.info("Treatment year: %d  (Post = 1 if activity_year >= %d)",
             treatment_year, treatment_year)

    df = prepare_panel(panel_raw, treatment_year)

    log.info("Panel composition after preparation:")
    log.info("  Banks:    %d LEIs, %d lender-years",
             df[df["nonbank"]==0]["lei"].nunique(),
             (df["nonbank"]==0).sum())
    log.info("  Nonbanks: %d LEIs, %d lender-years",
             df[df["nonbank"]==1]["lei"].nunique(),
             (df["nonbank"]==1).sum())
    log.info("  Pre-period rows:  %d  (year < %d)",
             (df["post"]==0).sum(), treatment_year)
    log.info("  Post-period rows: %d  (year >= %d)",
             (df["post"]==1).sum(), treatment_year)

    # ---- Run tables ----
    t4 = run_baseline_table(df)
    t5 = run_heterogeneity_table(df)
    t6 = run_robustness_table(panel_raw, treatment_year)

    # ---- Save ----
    t4.to_csv(TABLES_DIR / "table04_did_baseline.csv",     index=False)
    t5.to_csv(TABLES_DIR / "table05_did_heterogeneity.csv", index=False)
    t6.to_csv(TABLES_DIR / "table06_did_robustness.csv",   index=False)
    log.info("Saved table04/05/06 to %s", TABLES_DIR)

    # ---- Print ----
    print_table(t4, "Table 4: Baseline TWFE DiD")
    print_table(t5, "Table 5: Heterogeneity — Purchase vs Refi")
    print_table(t6, "Table 6: Robustness checks")

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
