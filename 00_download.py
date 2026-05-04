"""
00_download.py
==============
Downloads HMDA Snapshot LAR + TS files for 2018-2024.

Two modes:
  --states ST [ST ...]  : pull state-by-state via the Data Browser API
                          (smaller, more manageable)
  --national            : download the full national Snapshot LAR for each
                          year. Be warned: these files are tens of GB each.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

YEARS = list(range(2018, 2025))

# State codes covering ~60% of US mortgage volume.
DEFAULT_STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
                  "NJ", "VA", "WA", "AZ", "MA"]

DATA_BROWSER_CSV = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
SNAPSHOT_LAR_TEMPLATE = (
    "https://s3.amazonaws.com/cfpb-hmda-public/prod/"
    "snapshot-data/{year}/{year}_public_lar_csv.zip"
)
SNAPSHOT_TS_TEMPLATE = (
    "https://s3.amazonaws.com/cfpb-hmda-public/prod/"
    "snapshot-data/{year}/{year}_public_ts_csv.zip"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")


def download_state_year(year: int, state: str, out_path: Path,
                        chunk_size: int = 1024 * 1024) -> None:
    """Pull one state-year via the Data Browser API and stream to disk.

    The /view/csv endpoint requires either a geographic filter or LEI.
    We pass `states={state}&years={year}` and stream the response.
    """
    params = {"states": state, "years": year}
    url = f"{DATA_BROWSER_CSV}?{urlencode(params)}"
    log.info("  GET %s -> %s", url, out_path.name)

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def download_states(states: list[str], years: list[int]) -> None:
    """Download state-by-state for the given years; concatenate per year."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        log.info("Year %d", year)
        # Download each state to a temporary file, then concatenate.
        partials = []
        for state in states:
            partial = RAW_DIR / f"hmda_{year}_{state}.csv"
            if partial.exists():
                log.info("  %s already exists, skipping", partial.name)
            else:
                try:
                    download_state_year(year, state, partial)
                    time.sleep(1)  # be polite to the API
                except requests.RequestException as e:
                    log.error("  %s/%d failed: %s", state, year, e)
                    continue
            partials.append(partial)

        # Concatenate to a single LAR file per year, matching the
        # naming convention expected by 01_clean_lar.py
        combined = RAW_DIR / f"hmda_{year}_lar.txt"
        log.info("  concatenating %d states -> %s", len(partials), combined.name)
        first = True
        with open(combined, "w") as out:
            for partial in partials:
                if not partial.exists():
                    continue
                with open(partial) as inp:
                    header = inp.readline()
                    if first:
                        out.write(header.replace(",", "|"))
                        first = False
                    for line in inp:
                        out.write(line.replace(",", "|"))


def download_national(years: list[int]) -> None:
    """Download the full national Snapshot LAR + TS files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        for kind, template in (("lar", SNAPSHOT_LAR_TEMPLATE),
                                ("ts", SNAPSHOT_TS_TEMPLATE)):
            url = template.format(year=year)
            out = RAW_DIR / f"hmda_{year}_{kind}.zip"
            if out.exists():
                log.info("%s already exists, skipping", out.name)
                continue
            log.info("Downloading %s ...", url)
            with requests.get(url, stream=True, timeout=1800) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            log.info("  saved %s (%.1f MB)", out.name,
                     out.stat().st_size / 1e6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="+", default=None,
                        help="State codes to download (default: large-states list)")
    parser.add_argument("--national", action="store_true",
                        help="Download full national Snapshot zip files")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    args = parser.parse_args(argv)

    if args.national:
        download_national(args.years)
    else:
        states = args.states or DEFAULT_STATES
        download_states(states, args.years)
    return 0


if __name__ == "__main__":
    sys.exit(main())
