#!/usr/bin/env python3
"""
predict.py: 2026 FIFA World Cup match forecaster
==================================================

Produces W/D/L probabilities and a full Jump Cup prop slate from one coherent
scoreline distribution (Option 3: calibrated XGBoost tilt + ratings total anchor).

Usage
-----
  python predict.py "Switzerland" "Canada"
  python predict.py Spain "Cabo Verde" --questions examples/match.json
  python predict.py --refresh Switzerland Canada
"""

import os
import sys
import json
import argparse

import pandas as pd

from forecaster.data      import load_results, refresh
from forecaster.features  import build_dataset, form_as_of, h2h_as_of
from forecaster.wdl_model import (FEATURES, TRAIN_START, VAL_START,
                                   split_by_date, train_model, predict_proba,
                                   evaluate, _fill_na)
from forecaster.ratings   import fit as fit_ratings, lambdas as rating_lambdas
from forecaster.scoreline import (fit_lams_from_supremacy, symmetric_result_probs,
                                   compute_props, result_probs, DC_RHO)
from forecaster.fixtures  import find_fixture, list_team_names
from forecaster.chart     import make_chart
from forecaster.names     import normalize

_MODEL_CACHE: dict = {}


def _get_model(dataset, cutoff_date: str):
    if cutoff_date not in _MODEL_CACHE:
        train, val = split_by_date(dataset, TRAIN_START, VAL_START, cutoff_date)
        if len(train) < 100 or len(val) < 20:
            raise RuntimeError(
                f"Insufficient training data for cutoff {cutoff_date}: "
                f"{len(train)} train / {len(val)} val rows.")
        _MODEL_CACHE[cutoff_date] = train_model(train, val)
    return _MODEL_CACHE[cutoff_date]


def _match_row(results, final_elo, home, away, match_date, neutral, weight):
    hf = form_as_of(results, home, match_date)
    af = form_as_of(results, away, match_date)
    he = final_elo.get(home, 1500.0)
    ae = final_elo.get(away, 1500.0)
    n, wr, gd = h2h_as_of(results, home, away, match_date)
    row = {
        "neutral":           int(neutral),
        "tournament_weight": weight,
        "home_elo":          he,
        "away_elo":          ae,
        "elo_diff":          he - ae,
        "home_win5":         hf["win5"],  "away_win5":  af["win5"],
        "home_gd5":          hf["gd5"],   "away_gd5":   af["gd5"],
        "home_win10":        hf["win10"], "away_win10": af["win10"],
        "home_rest_days":    hf["rest_days"],
        "away_rest_days":    af["rest_days"],
        "h2h_n":             n,
        "h2h_home_winrate":  wr if not pd.isna(wr) else 0.5,
        "h2h_home_gd":       gd if not pd.isna(gd) else 0.0,
    }
    return _fill_na(pd.DataFrame([row]))[FEATURES].astype(float)


def _symmetric_calibrated_probs(model, calibrators, results, final_elo,
                                  home, away, match_date, neutral, weight):
    X_ab = _match_row(results, final_elo, home, away, match_date, neutral, weight)
    X_ba = _match_row(results, final_elo, away, home, match_date, neutral, weight)
    ph_ab, pd_ab, pa_ab = predict_proba(model, calibrators, X_ab)
    ph_ba, pd_ba, pa_ba = predict_proba(model, calibrators, X_ba)
    p_h  = (ph_ab + pa_ba) / 2
    p_d  = (pd_ab + pd_ba) / 2
    p_a  = (pa_ab + ph_ba) / 2
    tot  = p_h + p_d + p_a
    return p_h / tot, p_d / tot, p_a / tot


def _tag(conf, p_home, p_away, he, ae):
    fav_is_home   = p_home >= p_away
    elo_fav_home  = he >= ae
    upset = fav_is_home != elo_fav_home
    strength = "LOCK" if conf >= 0.60 else ("LEAN" if conf >= 0.45 else "TOSS-UP")
    return strength + ("  ⚠️  UPSET PICK" if upset else "")


def _default_questions(home_name: str, away_name: str) -> list:
    return [
        {"id": "q1",  "type": "result",           "params": {"side": "home"},
         "text": f"Will {home_name} win the match?"},
        {"id": "q2",  "type": "result",           "params": {"side": "draw"},
         "text": "Will the match end in a draw?"},
        {"id": "q3",  "type": "result",           "params": {"side": "away"},
         "text": f"Will {away_name} win the match?"},
        {"id": "q4",  "type": "match_total_over", "params": {"line": 2, "scope": "match"},
         "text": "Will there be 3 or more total goals?"},
        {"id": "q5",  "type": "match_total_under","params": {"line": 2, "scope": "match"},
         "text": "Will the match have 2 or fewer total goals?"},
        {"id": "q6",  "type": "btts",             "params": {},
         "text": "Will both teams score?"},
        {"id": "q7",  "type": "ht_result",        "params": {"outcome": "tie"},
         "text": "At halftime, will the match be tied?"},
        {"id": "q8",  "type": "team_scores",      "params": {"side": "home", "scope": "match"},
         "text": f"Will {home_name} score at least 1 goal?"},
        {"id": "q9",  "type": "team_scores",      "params": {"side": "away", "scope": "match"},
         "text": f"Will {away_name} score at least 1 goal?"},
        {"id": "q10", "type": "pen_or_red",       "params": {},
         "text": "Will a penalty be awarded OR a red card be shown?"},
    ]


def forecast(team_a: str, team_b: str,
             questions: list | None = None,
             verbose: bool = True,
             save_chart: bool = True) -> dict:
    if verbose:
        print("\nLoading data + building features ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)

    fx = find_fixture(team_a, team_b)
    if fx is None:
        raise ValueError(
            f"No World Cup fixture found for '{team_a}' vs '{team_b}'.\n"
            f"Teams in the schedule: {', '.join(list_team_names())}")

    home = fx["home"]
    away = fx["away"]
    match_date = fx["date"]
    neutral    = fx["neutral"]
    weight     = 4

    valid_teams = {normalize(r["home_team"]) for r in results} | \
                  {normalize(r["away_team"]) for r in results}
    if home not in valid_teams or away not in valid_teams:
        raise ValueError(
            f"One or both teams ({home}, {away}) have no historical results.")

    if verbose:
        print(f"Training XGBoost (cutoff {match_date}) ...")
    model, calibrators = _get_model(dataset, match_date)
    p_home_xgb, p_draw_xgb, p_away_xgb = _symmetric_calibrated_probs(
        model, calibrators, results, final_elo,
        home, away, match_date, neutral, weight)

    ratings_model = fit_ratings(results, ref_date=match_date)
    lam_h_base, lam_a_base = rating_lambdas(
        ratings_model, home, away, neutral=neutral)
    total_anchor = lam_h_base + lam_a_base

    lam_home, lam_away = fit_lams_from_supremacy(
        p_home_xgb, p_away_xgb, total_anchor, rho=DC_RHO)

    p_home, p_draw, p_away = symmetric_result_probs(lam_home, lam_away)

    if questions is None:
        questions = _default_questions(fx["home_disp"], fx["away_disp"])

    match_ctx = {
        "home":      fx["home_disp"],
        "away":      fx["away_disp"],
        "lam_home":  lam_home,
        "lam_away":  lam_away,
        "questions": questions,
    }
    compute_props(match_ctx)

    outcomes = [(fx["home_disp"], p_home), ("Draw", p_draw), (fx["away_disp"], p_away)]
    pick, conf = max(outcomes, key=lambda x: x[1])
    he = final_elo.get(home, 1500.0)
    ae = final_elo.get(away, 1500.0)
    tag = _tag(conf, p_home, p_away, he, ae)

    chart_path = None
    if save_chart:
        out_dir = os.path.join("predictions", str(match_date))
        chart_path = make_chart(fx, p_home, p_draw, p_away,
                                lam_home, lam_away, out_dir)

    if verbose:
        _print_result(fx, p_home, p_draw, p_away, pick, conf, tag,
                      lam_home, lam_away, match_ctx["questions"], chart_path)

    return {
        "home": fx["home_disp"], "away": fx["away_disp"],
        "date": match_date, "group": fx["group"], "stadium": fx["stadium"],
        "neutral": neutral,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "pick": pick, "confidence": conf, "tag": tag,
        "lam_home": lam_home, "lam_away": lam_away,
        "lam_home_ratings": lam_h_base, "lam_away_ratings": lam_a_base,
        "p_home_xgb": p_home_xgb, "p_draw_xgb": p_draw_xgb, "p_away_xgb": p_away_xgb,
        "elo_home": he, "elo_away": ae,
        "questions": match_ctx["questions"],
        "chart_path": chart_path,
    }


def _print_result(fx, p_home, p_draw, p_away, pick, conf, tag,
                  lam_home, lam_away, questions, chart_path):
    W = 64
    print("\n" + "=" * W)
    print(f"  {fx['home_disp']} vs {fx['away_disp']}")
    print(f"  {fx['date']}  ·  {fx['group']}  ·  {fx['stadium']}")
    if not fx["neutral"]:
        print(f"  (host-nation home venue — not neutral)")
    print("=" * W)
    print(f"  {'xG (home)':20} {lam_home:.2f}")
    print(f"  {'xG (away)':20} {lam_away:.2f}")
    print("-" * W)
    print(f"  {fx['home_disp']:<26} win   {p_home*100:>5.1f}%")
    print(f"  {'Draw':<26}       {p_draw*100:>5.1f}%")
    print(f"  {fx['away_disp']:<26} win   {p_away*100:>5.1f}%")
    print("-" * W)
    print(f"  PICK: {pick}  ({conf*100:.1f}%)   [{tag}]")
    print("=" * W)

    if questions:
        print("\n  PROP SLATE")
        print("  " + "-" * 62)
        print(f"  {'Question':<48} {'Model':>6}")
        print("  " + "-" * 62)
        for q in questions:
            p = q.get("model_prob", float("nan"))
            label = q.get("text", q["type"])
            print(f"  {label[:47]:<48} {p*100:>5.1f}%")
        print("  " + "-" * 62)
        print("  Note: fouls/cards/corners/SoT use base-rate defaults unless")
        print("  overridden in --questions JSON params.\n")

    if chart_path:
        print(f"  Chart saved -> {chart_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="2026 World Cup match forecaster (W/D/L + prop slate)")
    parser.add_argument("teams", nargs="*",
                        help="Two team names, e.g. Switzerland Canada")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download results.csv before forecasting")
    parser.add_argument("--questions", metavar="FILE",
                        help="JSON file with questions (cup-data format)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip saving the PNG chart")
    parser.add_argument("--eval", action="store_true",
                        help="Print validation metrics after training")
    args = parser.parse_args()

    if args.refresh:
        refresh()

    if len(args.teams) >= 2:
        team_a, team_b = args.teams[0], args.teams[1]
    else:
        print("Enter the two teams to predict.")
        team_a = input("  Team 1: ").strip()
        team_b = input("  Team 2: ").strip()

    questions = None
    if args.questions:
        with open(args.questions) as f:
            data = json.load(f)
        questions = data if isinstance(data, list) else data.get("questions")

    try:
        result = forecast(team_a, team_b,
                          questions=questions,
                          verbose=True,
                          save_chart=not args.no_chart)
        if args.eval:
            results  = load_results()
            dataset, _ = build_dataset(results)
            train, val = split_by_date(dataset, cutoff=result["date"])
            m, cal = train_model(train, val)
            print("\n  Validation metrics:")
            evaluate(m, cal, val)

    except ValueError as e:
        print(f"\n  Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
