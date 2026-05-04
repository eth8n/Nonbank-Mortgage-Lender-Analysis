"""
03_descriptives.py
INPUT:
------
  data/final/panel_lei_year.parquet   -- main analysis panel from 02_build_panel.py

OUTPUTS:
--------
  output/figures/
    fig01_loan_volume_by_type.png      -- log loan volume over time by lender type
    fig02_nonbank_market_share.png     -- nonbank share of originations over time
    fig03_loan_composition.png         -- purchase vs refi mix by lender type
    fig04_avg_interest_rate.png        -- mean interest rate trajectories
    fig05_lender_counts.png            -- active lenders per year by type

  output/tables/
    table01_summary_stats.csv          -- bank vs nonbank means, pre vs post
    table02_year_by_type_summary.csv   -- year x lender-type cell counts
    table03_lender_dynamics.csv        -- entry, exit, persistence by type

RUN:
----
  python code/03_descriptives.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required in headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_DIR = PROJECT_ROOT / "data" / "final"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

TREATMENT_YEAR = 2022

# Plot styling. We use a deliberately neutral, journal-style aesthetic:
# black/grey palette, no gridlines, small sans-serif. The point of these
# figures is clarity, not decoration.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Two-color palette: banks = navy, nonbanks = orange. Colorblind-safe,
# print-legible, and consistent across all figures.
COLOR_BANK = "#1f3b6c"
COLOR_NONBANK = "#d35400"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("descriptives")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_treatment_line(ax, year: int = TREATMENT_YEAR,
                       label: str = "Fed liftoff (2022)") -> None:
    """Add a vertical dashed line at the treatment year.

    Used on every time-series figure so the eye anchors on the treatment
    date. We place it at year - 0.5 to sit between the last pre-period
    year and the first post-period year, which is the conventional
    placement for annual-frequency event boundaries.
    """
    ax.axvline(year - 0.5, color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(year - 0.4, ymax - 0.05 * (ymax - ymin),
            label, fontsize=8, va="top", color="black", alpha=0.7)


def aggregate_by_year_type(panel: pd.DataFrame) -> pd.DataFrame:
    """Year x lender-type aggregates used by several plots."""
    return (
        panel.groupby(["activity_year", "lender_type"], observed=True)
        .agg(
            n_lenders=("lei", "nunique"),
            total_loans=("loan_count", "sum"),
            total_volume=("loan_volume", "sum"),
            mean_loan_size=("avg_loan_size", "mean"),
            mean_interest_rate=("avg_interest_rate", "mean"),
            mean_purchase_share=("purchase_share", "mean"),
            mean_refi_share=("refi_share", "mean"),
            mean_govt_share=("govt_loan_share", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig01_loan_volume_by_type(panel: pd.DataFrame, agg: pd.DataFrame) -> None:
    """Figure 1: log loan volume by lender type over time.

    THIS IS THE PRE-TRENDS PLOT. Look for parallel movement before 2022
    and divergence after. If banks and nonbanks were drifting apart
    pre-2022, the parallel-trends assumption is suspect.

    We plot both total volume (top) and average per-lender volume (bottom)
    because total volume confounds intensive (per-lender) and extensive
    (number of lenders) margins. Per-lender volume is the cleaner pre-
    trends test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel (a): total log volume
    for lt, color in [("bank", COLOR_BANK), ("nonbank", COLOR_NONBANK)]:
        s = agg[agg["lender_type"] == lt]
        axes[0].plot(s["activity_year"], np.log(s["total_volume"]),
                     marker="o", color=color, linewidth=1.8,
                     markersize=5, label=lt.capitalize())
    axes[0].set_ylabel("log(total origination volume, USD)")
    axes[0].set_xlabel("Year")
    axes[0].set_title("(a) Total origination volume")
    axes[0].legend(loc="lower left")
    add_treatment_line(axes[0])

    # Panel (b): mean log volume per lender (intensive margin)
    per_lender = panel.copy()
    per_lender["log_volume"] = np.log(per_lender["loan_volume"])
    pl_agg = (per_lender.groupby(["activity_year", "lender_type"],
                                  observed=True)["log_volume"]
              .mean().reset_index())
    for lt, color in [("bank", COLOR_BANK), ("nonbank", COLOR_NONBANK)]:
        s = pl_agg[pl_agg["lender_type"] == lt]
        axes[1].plot(s["activity_year"], s["log_volume"],
                     marker="o", color=color, linewidth=1.8,
                     markersize=5, label=lt.capitalize())
    axes[1].set_ylabel("Mean log(volume) per lender")
    axes[1].set_xlabel("Year")
    axes[1].set_title("(b) Per-lender origination volume")
    axes[1].legend(loc="lower left")
    add_treatment_line(axes[1])

    fig.suptitle("Origination volume by lender type, 2018-2024",
                 y=1.02, fontsize=12)
    out = FIGURES_DIR / "fig01_loan_volume_by_type.png"
    fig.savefig(out)
    plt.close(fig)
    log.info("  wrote %s", out.name)


def fig02_nonbank_market_share(agg: pd.DataFrame) -> None:
    """Figure 2: nonbank share of originations over time.

    The descriptive headline. Y-axis 0-1 (or 0-100%). Visually shows
    whether the post-2022 period interrupts a continuing rise in nonbank
    dominance.
    """
    pivot = agg.pivot(index="activity_year", columns="lender_type",
                      values="total_loans").fillna(0)
    pivot["nonbank_share"] = pivot["nonbank"] / (pivot["bank"] + pivot["nonbank"])
    pivot["nonbank_volume_share"] = (
        agg.pivot(index="activity_year", columns="lender_type",
                  values="total_volume").fillna(0)
        .pipe(lambda d: d["nonbank"] / (d["bank"] + d["nonbank"]))
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(pivot.index, pivot["nonbank_share"], marker="o",
            color=COLOR_NONBANK, linewidth=2,
            label="Share of loan count")
    ax.plot(pivot.index, pivot["nonbank_volume_share"], marker="s",
            color=COLOR_NONBANK, linewidth=2, alpha=0.5,
            linestyle="--", label="Share of dollar volume")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Nonbank share of originations")
    ax.set_xlabel("Year")
    ax.set_title("Nonbank market share, 2018-2024")
    ax.legend(loc="lower left")
    add_treatment_line(ax)

    out = FIGURES_DIR / "fig02_nonbank_market_share.png"
    fig.savefig(out)
    plt.close(fig)
    log.info("  wrote %s", out.name)


def fig03_loan_composition(agg: pd.DataFrame) -> None:
    """Figure 3: purchase vs refi share by lender type.

    Mechanism diagnostic. If nonbanks were significantly more refi-heavy
    pre-2022, then the post-2022 collapse in refi demand mechanically
    hits nonbanks harder regardless of any funding-model channel. This
    figure tells the reader whether that's a real concern.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, var, title in [
        (axes[0], "mean_purchase_share", "(a) Purchase share"),
        (axes[1], "mean_refi_share",     "(b) Refinance share"),
    ]:
        for lt, color in [("bank", COLOR_BANK), ("nonbank", COLOR_NONBANK)]:
            s = agg[agg["lender_type"] == lt]
            ax.plot(s["activity_year"], s[var], marker="o", color=color,
                    linewidth=1.8, markersize=5, label=lt.capitalize())
        ax.set_xlabel("Year")
        ax.set_ylabel("Share of originations")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title)
        ax.legend(loc="best")
        add_treatment_line(ax)

    fig.suptitle("Loan-purpose composition by lender type",
                 y=1.02, fontsize=12)
    out = FIGURES_DIR / "fig03_loan_composition.png"
    fig.savefig(out)
    plt.close(fig)
    log.info("  wrote %s", out.name)


def fig04_avg_interest_rate(agg: pd.DataFrame) -> None:
    """Figure 4: average interest rate trajectories.

    Sanity check that the rate shock actually shows up in the data, and
    that banks and nonbanks faced similar rates (so the channel of
    interest is funding-side, not demand-side rate differentiation).
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lt, color in [("bank", COLOR_BANK), ("nonbank", COLOR_NONBANK)]:
        s = agg[agg["lender_type"] == lt]
        ax.plot(s["activity_year"], s["mean_interest_rate"],
                marker="o", color=color, linewidth=1.8, markersize=5,
                label=lt.capitalize())
    ax.set_ylabel("Mean origination interest rate (%)")
    ax.set_xlabel("Year")
    ax.set_title("Average mortgage interest rate by lender type")
    ax.legend(loc="best")
    add_treatment_line(ax)

    out = FIGURES_DIR / "fig04_avg_interest_rate.png"
    fig.savefig(out)
    plt.close(fig)
    log.info("  wrote %s", out.name)


def fig05_lender_counts(agg: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Figure 5: number of active lenders by year and type.

    Documents the extensive margin. If many nonbanks exited post-2022
    while banks held steady, the per-lender estimates understate
    aggregate contraction. This figure motivates why both the panel
    estimate and the aggregate share matter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel (a): active lender counts
    for lt, color in [("bank", COLOR_BANK), ("nonbank", COLOR_NONBANK)]:
        s = agg[agg["lender_type"] == lt]
        axes[0].plot(s["activity_year"], s["n_lenders"], marker="o",
                     color=color, linewidth=1.8, markersize=5,
                     label=lt.capitalize())
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Active lenders")
    axes[0].set_title("(a) Active lenders per year")
    axes[0].legend(loc="best")
    add_treatment_line(axes[0])

    # Panel (b): cumulative entry/exit. We define entry as the first year
    # a LEI appears, exit as the year after its last appearance (if before
    # the sample end).
    first_year = panel.groupby("lei")["activity_year"].min().rename("first_year")
    last_year = panel.groupby("lei")["activity_year"].max().rename("last_year")
    types = panel.drop_duplicates("lei").set_index("lei")["lender_type"]
    timing = pd.concat([first_year, last_year, types], axis=1)
    sample_end = panel["activity_year"].max()

    entry_counts = (timing.reset_index()
                    .groupby(["first_year", "lender_type"]).size()
                    .rename("n_entries").reset_index())
    exit_counts = (timing[timing["last_year"] < sample_end]
                   .reset_index().assign(exit_year=lambda d: d["last_year"] + 1)
                   .groupby(["exit_year", "lender_type"]).size()
                   .rename("n_exits").reset_index())

    width = 0.35
    years = sorted(panel["activity_year"].unique())
    x = np.arange(len(years))

    bank_entries = [int(entry_counts.query(
        f"first_year == {y} and lender_type == 'bank'")["n_entries"].sum())
        for y in years]
    nonbank_entries = [int(entry_counts.query(
        f"first_year == {y} and lender_type == 'nonbank'")["n_entries"].sum())
        for y in years]
    bank_exits = [-int(exit_counts.query(
        f"exit_year == {y} and lender_type == 'bank'")["n_exits"].sum())
        for y in years]
    nonbank_exits = [-int(exit_counts.query(
        f"exit_year == {y} and lender_type == 'nonbank'")["n_exits"].sum())
        for y in years]

    axes[1].bar(x - width/2, bank_entries, width, color=COLOR_BANK,
                label="Bank entry")
    axes[1].bar(x + width/2, nonbank_entries, width, color=COLOR_NONBANK,
                label="Nonbank entry")
    axes[1].bar(x - width/2, bank_exits, width, color=COLOR_BANK, alpha=0.4)
    axes[1].bar(x + width/2, nonbank_exits, width, color=COLOR_NONBANK, alpha=0.4)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(years)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Lenders entering (+) / exiting (-)")
    axes[1].set_title("(b) Entry and exit by year")
    axes[1].legend(loc="best", fontsize=8)

    fig.suptitle("Lender extensive-margin dynamics",
                 y=1.02, fontsize=12)
    out = FIGURES_DIR / "fig05_lender_counts.png"
    fig.savefig(out)
    plt.close(fig)
    log.info("  wrote %s", out.name)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table01_summary_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Table 1: pre/post means by lender type, with simple difference.

    The standard "Table 1" of any DiD paper. We compute the four cell
    means: bank-pre, bank-post, nonbank-pre, nonbank-post. The DiD
    estimator is the difference of differences, but reporting the means
    themselves is more informative for the reader.

    NOTE: this is a *descriptive* difference, not the regression estimate.
    The regression in 04_did_main.py controls for lender FE and time FE.
    The simple means here ignore those.
    """
    panel = panel.copy()
    panel["period"] = np.where(panel["activity_year"] >= TREATMENT_YEAR,
                                "Post (>=2022)", "Pre (<2022)")

    # Variables to summarize. Mix of outcomes and observable controls.
    vars_to_summarize = [
        "loan_count", "loan_volume", "avg_loan_size",
        "avg_interest_rate", "avg_dti", "avg_income",
        "purchase_share", "refi_share", "govt_loan_share",
        "n_states",
    ]

    rows = []
    for v in vars_to_summarize:
        cell_means = (panel.groupby(["lender_type", "period"], observed=True)[v]
                      .mean().unstack())
        # Compute the four cell means and the simple DiD: change in nonbank
        # minus change in bank (post-pre).
        delta_bank = cell_means.loc["bank", "Post (>=2022)"] - \
                     cell_means.loc["bank", "Pre (<2022)"]
        delta_nonbank = cell_means.loc["nonbank", "Post (>=2022)"] - \
                        cell_means.loc["nonbank", "Pre (<2022)"]
        did = delta_nonbank - delta_bank
        rows.append({
            "variable": v,
            "bank_pre":     cell_means.loc["bank",    "Pre (<2022)"],
            "bank_post":    cell_means.loc["bank",    "Post (>=2022)"],
            "nonbank_pre":  cell_means.loc["nonbank", "Pre (<2022)"],
            "nonbank_post": cell_means.loc["nonbank", "Post (>=2022)"],
            "delta_bank":   delta_bank,
            "delta_nonbank": delta_nonbank,
            "raw_did":      did,
        })

    return pd.DataFrame(rows).round(4)


def table02_year_by_type_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Table 2: cell counts to catch any year-type cells with too few obs.

    If a particular (year, lender_type) has very few lenders or loans,
    estimates for that cell will be noisy and event-study coefficients
    will be unstable. This table is a check on that.
    """
    out = (
        panel.groupby(["activity_year", "lender_type"], observed=True)
        .agg(
            n_lenders=("lei", "nunique"),
            total_loans=("loan_count", "sum"),
            total_volume_billions=("loan_volume", lambda x: x.sum() / 1e9),
            median_lender_loans=("loan_count", "median"),
        )
        .reset_index()
        .round(2)
    )
    return out


def table03_lender_dynamics(panel: pd.DataFrame) -> pd.DataFrame:
    """Table 3: persistence, entry, and exit rates by lender type.

    Reports, for each lender type:
      - n_lenders_total: distinct LEIs ever observed
      - mean_years_observed: average years the lender appears
      - share_balanced: fraction of lenders observed in all sample years
      - share_exited_pre_2022: fraction whose last year is < 2022
      - share_exited_post_2022: fraction whose last year is in [2022, 2023]
        (we cannot count 2024-final-year exits as exits since the sample ends)

    The exit-rate split is the single most important diagnostic for
    survivorship bias. If nonbanks exited at higher rates post-2022,
    the within-lender DiD estimate will UNDERSTATE the true contraction,
    because the worst-hit nonbanks drop out of the sample entirely.
    """
    last_year = panel.groupby("lei")["activity_year"].max()
    n_years_obs = panel.groupby("lei")["activity_year"].nunique()
    types = panel.drop_duplicates("lei").set_index("lei")["lender_type"]
    sample_end = panel["activity_year"].max()
    n_sample_years = panel["activity_year"].nunique()

    by_lei = pd.DataFrame({
        "lender_type": types,
        "last_year": last_year,
        "n_years_observed": n_years_obs,
    })
    by_lei["balanced"] = by_lei["n_years_observed"] == n_sample_years
    by_lei["exited_pre_2022"] = by_lei["last_year"] < TREATMENT_YEAR
    by_lei["exited_post_2022"] = (
        (by_lei["last_year"] >= TREATMENT_YEAR) &
        (by_lei["last_year"] < sample_end)
    )

    rows = []
    for lt in ["bank", "nonbank"]:
        sub = by_lei[by_lei["lender_type"] == lt]
        rows.append({
            "lender_type": lt,
            "n_lenders_total": len(sub),
            "mean_years_observed": sub["n_years_observed"].mean(),
            "share_balanced": sub["balanced"].mean(),
            "share_exited_pre_2022": sub["exited_pre_2022"].mean(),
            "share_exited_post_2022": sub["exited_post_2022"].mean(),
        })
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Building descriptive tables and figures")
    log.info("=" * 60)

    panel_path = FINAL_DIR / "panel_lei_year.parquet"
    if not panel_path.exists():
        log.error("Panel not found at %s. Run 02_build_panel.py first.",
                  panel_path)
        return 1

    panel = pd.read_parquet(panel_path)
    log.info("Loaded panel: %d rows, %d unique LEIs, years %d-%d",
             len(panel), panel["lei"].nunique(),
             panel["activity_year"].min(), panel["activity_year"].max())

    agg = aggregate_by_year_type(panel)

    # --- Figures ---
    log.info("Generating figures:")
    fig01_loan_volume_by_type(panel, agg)
    fig02_nonbank_market_share(agg)
    fig03_loan_composition(agg)
    fig04_avg_interest_rate(agg)
    fig05_lender_counts(agg, panel)

    # --- Tables ---
    log.info("Generating tables:")
    t1 = table01_summary_stats(panel)
    t1.to_csv(TABLES_DIR / "table01_summary_stats.csv", index=False)
    log.info("  wrote table01_summary_stats.csv")

    t2 = table02_year_by_type_summary(panel)
    t2.to_csv(TABLES_DIR / "table02_year_by_type_summary.csv", index=False)
    log.info("  wrote table02_year_by_type_summary.csv")

    t3 = table03_lender_dynamics(panel)
    t3.to_csv(TABLES_DIR / "table03_lender_dynamics.csv", index=False)
    log.info("  wrote table03_lender_dynamics.csv")

    # --- Print Table 1 to stdout for immediate inspection ---
    log.info("=" * 60)
    log.info("Table 1: pre/post means by lender type")
    log.info("(simple DiD = delta_nonbank - delta_bank; not the regression estimate)")
    log.info("=" * 60)
    print(t1.to_string(index=False))

    log.info("=" * 60)
    log.info("Table 3: lender dynamics (survivorship diagnostic)")
    log.info("=" * 60)
    print(t3.to_string(index=False))

    log.info("Done. Inspect output/figures/fig01_loan_volume_by_type.png FIRST.")
    log.info("If pre-trends are not visibly parallel, the DiD is in trouble.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
