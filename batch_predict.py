#!/usr/bin/env python3
"""
batch_predict.py: forecast every group-stage (and known knockout) match.

Usage
-----
  python batch_predict.py
  python batch_predict.py --csv out.csv
  python batch_predict.py --refresh
"""

import argparse
import os

import pandas as pd

from forecaster.data      import load_results, refresh
from forecaster.features  import build_dataset
from forecaster.ratings   import fit as fit_ratings, lambdas as rating_lambdas
from forecaster.scoreline import fit_lams_from_supremacy, symmetric_result_probs, DC_RHO
from forecaster.fixtures  import load_fixtures
from forecaster.names     import normalize
from predict              import _get_model, _symmetric_calibrated_probs, _tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="predictions/all_predictions.csv")
    ap.add_argument("--no-csv",   action="store_true")
    ap.add_argument("--refresh",  action="store_true")
    args = ap.parse_args()

    if args.refresh:
        refresh()

    print("\nLoading data + building features ...")
    results        = load_results()
    dataset, final_elo = build_dataset(results)
    valid_teams    = {normalize(r["home_team"]) for r in results} | \
                     {normalize(r["away_team"]) for r in results}

    ratings_model = fit_ratings(results, ref_date="2026-08-01")

    fixtures = [
        fx for fx in load_fixtures()
        if fx["home"] in valid_teams and fx["away"] in valid_teams
    ]
    fixtures.sort(key=lambda m: (str(m["date"]), str(m["match_number"])))

    if not fixtures:
        print("No predictable fixtures found.")
        return

    print(f"Predicting {len(fixtures)} matches ...\n")

    rows = []
    for fx in fixtures:
        home, away     = fx["home"], fx["away"]
        match_date     = fx["date"]
        neutral        = fx["neutral"]

        model, calibrators = _get_model(dataset, match_date)
        p_h, p_d, p_a = _symmetric_calibrated_probs(
            model, calibrators, results, final_elo,
            home, away, match_date, neutral, 4)

        lh_base, la_base = rating_lambdas(ratings_model, home, away, neutral=neutral)
        lam_home, lam_away = fit_lams_from_supremacy(
            p_h, p_a, lh_base + la_base, rho=DC_RHO)

        p_home, p_draw, p_away = symmetric_result_probs(lam_home, lam_away)

        outcomes = [(fx["home_disp"], p_home),
                    ("Draw", p_draw),
                    (fx["away_disp"], p_away)]
        pick, conf = max(outcomes, key=lambda x: x[1])
        he = final_elo.get(home, 1500.0)
        ae = final_elo.get(away, 1500.0)
        tag = _tag(conf, p_home, p_away, he, ae)

        rows.append({
            "match":      fx["match_number"],
            "date":       match_date,
            "group":      fx["group"],
            "home":       fx["home_disp"],
            "away":       fx["away_disp"],
            "neutral":    fx["neutral"],
            "lam_home":   round(lam_home, 3),
            "lam_away":   round(lam_away, 3),
            "p_home":     round(p_home, 4),
            "p_draw":     round(p_draw, 4),
            "p_away":     round(p_away, 4),
            "pick":       pick,
            "confidence": round(conf, 4),
            "tag":        tag,
        })

    W = 100
    print("=" * W)
    print(f"  {'#':<5}{'Date':<12}{'Grp':<7}{'Match':<34}"
          f"{'Home':>7}{'Draw':>7}{'Away':>7}  Pick")
    print("=" * W)
    for r in rows:
        mstr = f"{r['home']} v {r['away']}"
        num  = str(r["match"]).replace("Match ", "")
        print(f"  {num:<5}{r['date']:<12}{str(r['group']):<7}{mstr[:33]:<34}"
              f"{r['p_home']*100:>6.1f}%{r['p_draw']*100:>6.1f}%{r['p_away']*100:>6.1f}%"
              f"  {r['pick']} ({r['confidence']*100:.0f}% {r['tag']})")
    print("=" * W)
    print(f"  {len(rows)} matches predicted.\n")

    if not args.no_csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"  CSV saved -> {args.csv}\n")


if __name__ == "__main__":
    main()
