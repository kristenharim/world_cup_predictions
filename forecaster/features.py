"""
features.py — Elo ratings, recent form, rest days, head-to-head.

All features are computed strictly from data BEFORE the match being predicted
(no look-ahead). The main entry point is build_dataset(), which returns a
(dataset, final_elo) pair. final_elo is the ratings state at the end of the
training window, used at inference time.

Design notes
------------
- Elo uses pre-match ratings (computed before each game is applied).
- Rolling form uses shift(1) so the game itself is never included.
- H2H uses shift(1).expanding() for the same reason.
- All of this mirrors the original predict_today.py hygiene.
"""

import math
import numpy as np
import pandas as pd
from .names import normalize

# ── Elo constants ────────────────────────────────────────────────────────────
ELO_BASE     = 1500.0
ELO_K        = 32
ELO_HOME_ADV = 60       # points; 0 for neutral venues

# ── Tournament importance weight ─────────────────────────────────────────────
def tournament_weight(name: str) -> int:
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 4
    if "qualif" in t:
        return 3
    big = ["uefa nations", "copa america", "afc asian cup", "africa cup",
           "concacaf", "uefa euro", "confederations"]
    if any(tok in t for tok in big):
        return 3
    if "friendly" in t:
        return 1
    return 2


# ── Core Elo computation ──────────────────────────────────────────────────────
def compute_elo(df: pd.DataFrame):
    """
    Attach home_elo / away_elo / elo_diff to every row.
    Returns (df_with_elo, final_ratings_dict).
    """
    df = df.sort_values("date").reset_index(drop=True)
    rating = {}
    home_pre = np.zeros(len(df))
    away_pre = np.zeros(len(df))

    for i, row in df.iterrows():
        rh = rating.get(row.home_team, ELO_BASE)
        ra = rating.get(row.away_team, ELO_BASE)
        home_pre[i] = rh
        away_pre[i] = ra

        bonus = 0 if row.neutral else ELO_HOME_ADV
        exp_home = 1 / (1 + 10 ** (-((rh + bonus) - ra) / 400))
        score_home = 1.0 if row.label == 0 else (0.5 if row.label == 1 else 0.0)

        margin = abs(int(row.home_score) - int(row.away_score))
        # Margin multiplier: bigger wins move the needle more, but tempered by
        # the Elo gap so that expected blowouts shift less than upsets.
        mult = math.log(max(margin, 1) + 1) * (2.2 / (abs(rh - ra) * 0.001 + 2.2))

        rating[row.home_team] = rh + ELO_K * mult * (score_home - exp_home)
        rating[row.away_team] = ra + ELO_K * mult * ((1 - score_home) - (1 - exp_home))

    df["home_elo"]  = home_pre
    df["away_elo"]  = away_pre
    df["elo_diff"]  = home_pre - away_pre
    return df, rating


# ── Per-team long format helper ───────────────────────────────────────────────
def _per_team_long(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "date": df["date"].values, "team": df["home_team"].values,
        "opp":  df["away_team"].values,
        "gf":   df["home_score"].values, "ga": df["away_score"].values,
    })
    away = pd.DataFrame({
        "date": df["date"].values, "team": df["away_team"].values,
        "opp":  df["home_team"].values,
        "gf":   df["away_score"].values, "ga": df["home_score"].values,
    })
    long = pd.concat([home, away], ignore_index=True)
    long["result"] = np.where(long["gf"] > long["ga"], 1.0,
                              np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]
    return long


def _add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    long = _per_team_long(df).sort_values(["team", "date"]).reset_index(drop=True)
    long["prev_date"]   = long.groupby("team")["date"].shift(1)
    long["result_lag"]  = long.groupby("team")["result"].shift(1)
    long["gd_lag"]      = long.groupby("team")["gd"].shift(1)
    long["win5"]  = long.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    long["gd5"]   = long.groupby("team")["gd_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    long["win10"] = long.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(10, min_periods=1).mean())
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days
    form = long[["date", "team", "win5", "gd5", "win10", "rest_days"]].drop_duplicates(["date", "team"])

    df = df.merge(
        form.rename(columns={"team": "home_team", "win5": "home_win5",
                             "gd5": "home_gd5", "win10": "home_win10",
                             "rest_days": "home_rest_days"}),
        on=["date", "home_team"], how="left")
    df = df.merge(
        form.rename(columns={"team": "away_team", "win5": "away_win5",
                             "gd5": "away_gd5", "win10": "away_win10",
                             "rest_days": "away_rest_days"}),
        on=["date", "away_team"], how="left")
    return df


def _add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    long = _per_team_long(df).sort_values(["team", "opp", "date"]).reset_index(drop=True)
    g = long.groupby(["team", "opp"])
    long["h2h_n"]       = g.cumcount()
    long["h2h_winrate"] = g["result"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    long["h2h_gd"]      = g["gd"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    h2h = long[["date", "team", "opp", "h2h_n", "h2h_winrate", "h2h_gd"]]\
              .drop_duplicates(["date", "team", "opp"])
    df = df.merge(
        h2h.rename(columns={"team": "home_team", "opp": "away_team",
                             "h2h_winrate": "h2h_home_winrate",
                             "h2h_gd": "h2h_home_gd"}),
        on=["date", "home_team", "away_team"], how="left")
    return df


def build_dataset(results: list) -> tuple:
    """
    Build a fully-featured DataFrame from the raw results list.

    Returns
    -------
    (dataset: pd.DataFrame, final_elo: dict)
      dataset   — one row per historical match, with all features + label
      final_elo — {team: rating} at end of training window (for inference)
    """
    df = pd.DataFrame(results)
    df["date"]        = pd.to_datetime(df["date"])
    df["home_team"]   = df["home_team"].map(normalize)
    df["away_team"]   = df["away_team"].map(normalize)
    df["neutral"]     = df["neutral"].astype(int)
    df["label"]       = np.where(df["home_score"] > df["away_score"], 0,
                                 np.where(df["home_score"] == df["away_score"], 1, 2))
    df["tournament_weight"] = df["tournament"].map(tournament_weight)

    df, final_elo = compute_elo(df)
    df = _add_form_features(df)
    df = _add_h2h_features(df)
    return df, final_elo


# ── Inference-time form / h2h lookups ────────────────────────────────────────
def form_as_of(results: list, team: str, asof: str) -> dict:
    """Recent-form stats for `team` using only results before `asof`."""
    long_rows = []
    for r in results:
        if r["date"] >= asof:
            continue
        t_norm = normalize(r["home_team"])
        o_norm = normalize(r["away_team"])
        if t_norm == team:
            gf, ga = r["home_score"], r["away_score"]
        elif o_norm == team:
            gf, ga = r["away_score"], r["home_score"]
        else:
            continue
        long_rows.append({"date": r["date"], "gf": gf, "ga": ga})

    if not long_rows:
        return {"win5": 0.5, "gd5": 0.0, "win10": 0.5, "rest_days": 30.0}

    long_rows.sort(key=lambda x: x["date"])
    l5  = long_rows[-5:]
    l10 = long_rows[-10:]

    def res(row):
        return 1.0 if row["gf"] > row["ga"] else (0.5 if row["gf"] == row["ga"] else 0.0)

    last_date = long_rows[-1]["date"]
    rest = (pd.Timestamp(asof) - pd.Timestamp(last_date)).days

    return {
        "win5":      float(sum(res(r) for r in l5) / len(l5)),
        "gd5":       float(sum(r["gf"] - r["ga"] for r in l5) / len(l5)),
        "win10":     float(sum(res(r) for r in l10) / len(l10)),
        "rest_days": float(rest),
    }


def h2h_as_of(results: list, home: str, away: str, asof: str) -> tuple:
    """Head-to-head stats (from home's perspective) before `asof`."""
    rows = []
    for r in results:
        if r["date"] >= asof:
            continue
        hn = normalize(r["home_team"])
        an = normalize(r["away_team"])
        if hn == home and an == away:
            rows.append({"result": 1.0 if r["home_score"] > r["away_score"]
                                    else (0.5 if r["home_score"] == r["away_score"] else 0.0),
                         "gd": r["home_score"] - r["away_score"]})
        elif hn == away and an == home:
            rows.append({"result": 1.0 if r["away_score"] > r["home_score"]
                                    else (0.5 if r["home_score"] == r["away_score"] else 0.0),
                         "gd": r["away_score"] - r["home_score"]})
    if not rows:
        return 0.0, float("nan"), float("nan")
    return (float(len(rows)),
            float(sum(r["result"] for r in rows) / len(rows)),
            float(sum(r["gd"] for r in rows) / len(rows)))
