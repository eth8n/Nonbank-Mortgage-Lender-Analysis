# Nonbank Resilience to the 2022 Monetary Tightening

Replication package for Financial Crises Project.

---

## Requirements

- Python 3.10 or later
- All package dependencies are listed in `requirements.txt`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

All data are publicly available at no cost from the CFPB.


| File                                       | Source                                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| HMDA LAR (loan-level, 2018–2024)           | [CFPB Data Browser](https://ffiec.cfpb.gov/data-browser/)                                              |
| HMDA Transmittal Sheet / Panel (2018–2024) | [CFPB Data Publication](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/) |


> **Note on Transmittal Sheet files:** The TS/Panel files must be downloaded manually from the CFPB publication page for each year and placed in `data/raw/` before running `01_clean_lar.py`. The LAR files are downloaded automatically by `00_download.py`.

---

## Pipeline

Run the five scripts in order. Each script reads from the outputs of the previous one.

### Step 0 — Download HMDA LAR data

Downloads state-level LAR CSV files from the CFPB Data Browser API for 2018–2024.

```bash
# California only (used in this study, ~1–2 GB total)
python 00_download.py --states CA
```

Output: `data/raw/hmda_CA_{year}.csv` (one file per year per state)

---

### Step 1 — Clean LAR and build lender crosswalk

```bash
python 01_clean_lar.py
```

Output:

- `data/interim/lar_cleaned_{year}.parquet` — filtered loan-level file per year
- `data/interim/lender_panel.parquet` — LEI → bank/nonbank crosswalk

---

### Step 2 — Build analysis panel

```bash
python 02_build_panel.py
```

Output:

- `data/final/panel_lei_year.parquet`
- `data/final/panel_lei_year_balanced.parquet`
- `output/tables/panel_construction_log.csv`
- `output/tables/panel_summary.csv`

---

### Step 3 — Descriptive tables and figures

```bash
python 03_descriptives.py
```

Output:

- `output/figures/fig01_loan_volume_by_type.png` — pre-trends plot (key parallel-trends diagnostic)
- `output/tables/table01_summary_stats.csv`
- `output/tables/table02_year_by_type_summary.csv`
- `output/tables/table03_lender_dynamics.csv`

---

### Step 4 — Baseline DiD regressions

Estimates the two-way fixed effects DiD model across five outcome variables, the purchase/refi heterogeneity split, and four robustness checks. Uses `pyfixest` with standard errors clustered at the lender (LEI) level.

```bash
python 04_did_main.py
```

Output:

- `output/tables/table04_did_baseline.csv` — baseline TWFE results 
- `output/tables/table06_did_robustness.csv` — Robustness checks

---

## Project Structure

```
Crises_project/
├── 00_download.py               # Step 0: download HMDA LAR
├── 01_clean_lar.py              # Step 1: clean and classify lenders
├── 02_build_panel.py            # Step 2: aggregate to lender-year panel
├── 03_descriptives.py           # Step 3: descriptive tables and figures
├── 04_did_main.py               # Step 4: DiD regressions
├── requirements.txt
├── data/
│   ├── raw/                     # Raw HMDA downloads (not tracked)
│   ├── interim/                 # Cleaned loan-level parquets
│   └── final/                   # Analysis panels
└── output/
    ├── figures/                 # PNG figures
    └── tables/                  # CSV result tables
```
