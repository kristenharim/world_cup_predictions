#!/usr/bin/env python3
"""
batch_predict.py — predict every group-stage match in one run.

Loops over the fixtures, skips knockout rows whose teams are still
placeholders (e.g. "Winner match 91"), and predicts each match where both
real teams are known. Prints a table and writes a CSV.

Fidelity: this mirrors predict_today.py exactly — the model is retrained
with a cutoff at each match's date so it never peeks at the future. Models
are cached per date, so we train once per match-day, not once per match.

    python batch_predict.py                 # table + CSV (default)
    python batch_predict.py --csv out.csv   # custom CSV path
    python batch_predict.py --no-csv        # table only
"""
import argparse
import os
import sys

import pandas as pd

import predict_today as pt


def trained_model_for(dataset, cutoff, _cache={}):
    """Train (or reuse) a model with validation data cut off at `cutoff`."""
    if cutoff not in _cache:
        train, val = pt.split_by_date(dataset, pt.TRAIN_START, pt.VAL_START, cutoff)
        model, _, _ = pt.train_model(train, val)
        _cache[cutoff] = model
    return _cache[cutoff]


def group_stage_fixtures(valid_teams):
    """Yield fixture dicts for matches where both teams are real (not placeholders)."""
    fx = pd.read_csv(pt.FIXTURES_PATH)
    for _, row in fx.iterrows():
        teams = str(row["teams"])
        if " v " not in teams:
            continue
        left, right = [p.strip() for p in teams.split(" v ")]
        home, away = pt.map_fixture_name(left), pt.map_fixture_name(right)
        if home not in valid_teams or away not in valid_teams:
            continue  # knockout placeholder — can't predict yet
        yield {
            "match": row.get("match_number", ""),
            "group": row.get("group", ""),
            "stadium": row.get("stadium", ""),
            "date": row.get("date_dt", ""),
            "home_disp": left, "away_disp": right,
            "home": home, "away": away,
        }


def main():
    ap = argparse.ArgumentParser(description="Predict every group-stage match at once.")
    ap.add_argument("--csv", default="predictions/all_group_predictions.csv",
                    help="where to write the CSV (default: predictions/all_group_predictions.csv)")
    ap.add_argument("--no-csv", action="store_true", help="print the table but don't write a CSV")
    args = ap.parse_args()

    print("\nLoading data + building features ...")
    results = pt.load_results()
    dataset, final_elo = pt.build_dataset(results)
    valid_teams = set(results["home_team"]) | set(results["away_team"])
    long = pt.per_team_long(results)

    fixtures = sorted(group_stage_fixtures(valid_teams), key=lambda m: (str(m["date"]), str(m["match"])))
    if not fixtures:
        print("  No predictable group-stage fixtures found.")
        return

    print(f"Predicting {len(fixtures)} matches "
          f"(training one model per match-day) ...\n")

    rows = []
    for m in fixtures:
        model = trained_model_for(dataset, m["date"])
        p_home, p_draw, p_away = pt.predict_symmetric(
            model, long, final_elo, m["home"], m["away"], m["date"],
            pt.MATCH_NEUTRAL, pt.MATCH_WEIGHT)
        outcomes = [(m["home_disp"], p_home), ("Draw", p_draw), (m["away_disp"], p_away)]
        pick, conf = max(outcomes, key=lambda x: x[1])
        he, ae = final_elo.get(m["home"], pt.ELO_BASE), final_elo.get(m["away"], pt.ELO_BASE)
        tag = pt.tag_match(conf, p_home, p_away, he, ae)
        rows.append({
            "match": m["match"], "date": m["date"], "group": m["group"],
            "home": m["home_disp"], "away": m["away_disp"],
            "p_home": round(p_home, 4), "p_draw": round(p_draw, 4), "p_away": round(p_away, 4),
            "pick": pick, "confidence": round(conf, 4), "tag": tag,
        })

    # ── table ──────────────────────────────────────────────────────────────
    print("=" * 96)
    print(f"  {'#':<4}{'Date':<12}{'Group':<9}{'Match':<34}"
          f"{'Home':>6}{'Draw':>7}{'Away':>7}  Pick")
    print("=" * 96)
    for r in rows:
        match_str = f"{r['home']} v {r['away']}"
        num = str(r["match"]).replace("Match ", "")
        print(f"  {num:<4}{r['date']:<12}{str(r['group']):<9}{match_str[:33]:<34}"
              f"{r['p_home']*100:>5.1f}%{r['p_draw']*100:>6.1f}%{r['p_away']*100:>6.1f}%"
              f"  {r['pick']} ({r['confidence']*100:.0f}% {r['tag']})")
    print("=" * 96)
    print(f"  {len(rows)} matches predicted.")

    # ── CSV ────────────────────────────────────────────────────────────────
    if not args.no_csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"  CSV saved -> {args.csv}\n")


if __name__ == "__main__":
    main()
