"""
data.py: single source of truth for historical match results.

Downloads the martj42/international_results dataset on first run and caches it
locally. Call refresh() any time you want to pull the latest results (the repo
updates within a day of each match).

All other modules import load_results() from here; they never touch the CSV
path directly.
"""

import os
import csv
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "data_cache")
CSV_PATH = os.path.join(CACHE_DIR, "results.csv")
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

SINCE = "2006-01-01"


def refresh():
    """(Re)download the dataset from the martj42 repo."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Fetching {URL} ...")
    urllib.request.urlretrieve(URL, CSV_PATH)
    n = sum(1 for _ in open(CSV_PATH)) - 1
    print(f"  cached {n:,} matches -> {CSV_PATH}")


def load_results(since=SINCE):
    """
    Return a list of completed match dicts with keys:
      date, home_team, away_team, home_score, away_score, tournament, neutral
    """
    if not os.path.exists(CSV_PATH):
        print("results.csv not found — downloading now.")
        refresh()

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["date"]
            if d < since:
                continue
            hs, as_ = r["home_score"].strip(), r["away_score"].strip()
            if not hs or not as_ or hs in ("NA", "") or as_ in ("NA", ""):
                continue
            rows.append({
                "date":       d,
                "home_team":  r["home_team"].strip(),
                "away_team":  r["away_team"].strip(),
                "home_score": int(hs),
                "away_score": int(as_),
                "tournament": r["tournament"].strip(),
                "neutral":    r["neutral"].strip().upper() == "TRUE",
            })

    rows.sort(key=lambda r: r["date"])
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        refresh()
    else:
        rows = load_results()
        print(f"Loaded {len(rows):,} matches ({rows[0]['date']} .. {rows[-1]['date']})")
