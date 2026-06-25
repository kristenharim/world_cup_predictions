"""
ratings.py — recency-weighted, opponent-adjusted Poisson attack/defense ratings.

Fits attack (att) and defense (dfn) ratings for every team by iteratively
solving for consistent goal expectations. Shrinks toward the league mean using
pseudo-match regularization so teams with thin data don't dominate.

The output (lam_home, lam_away) for any fixture is the primary input to the
Poisson scoreline engine (scoreline.py) and to the lambda-inversion step in
the hybrid model (model.py).

Ported from sports-betting-fifa/engine/scripts/ratings.py; no stdlib-only
requirement here since we already depend on pandas/numpy for features.py.
"""

import math
from .names import normalize

# ── Hyperparameters ──────────────────────────────────────────────────────────
HALFLIFE_YEARS  = 2.5      # exponential recency decay
SINCE           = "2016-01-01"
FRIENDLY_WEIGHT = 0.5      # friendlies carry less signal
SHRINK_PSEUDO   = 6.0      # pseudo-matches pulling att/dfn toward league mean
ITERS           = 25

# Non-neutral venue factors: home side scores ~25% more than league average,
# away side ~25% less. These are applied during fitting and inference.
GAMMA_HOME, GAMMA_AWAY = 1.25, 0.75


def _years_between(d1: str, d2: str) -> float:
    (y1, m1, da1) = map(int, d1.split("-"))
    (y2, m2, da2) = map(int, d2.split("-"))
    return ((y2 - y1) * 365 + (m2 - m1) * 30 + (da2 - da1)) / 365.0


def fit(results: list, ref_date: str) -> dict:
    """
    Fit attack / defense ratings from `results` (already filtered to >= SINCE).

    Parameters
    ----------
    results   : list of dicts from data.load_results()
    ref_date  : ISO date string used as "today" for recency weighting

    Returns
    -------
    {"att": {team: float}, "dfn": {team: float}, "mu": float}
    """
    # Filter to SINCE and completed matches only
    matches = [
        (r["date"], normalize(r["home_team"]), normalize(r["away_team"]),
         r["home_score"], r["away_score"], r["tournament"], r["neutral"])
        for r in results
        if r["date"] >= SINCE
    ]

    teams = set(m[1] for m in matches) | set(m[2] for m in matches)
    att = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}

    # Per-match weights + league mean goals/side
    W = []
    tot_g = tot_w = 0.0
    for (d, h, a, hs, as_, trn, neu) in matches:
        w = 0.5 ** (_years_between(d, ref_date) / HALFLIFE_YEARS)
        if "friendly" in trn.lower():
            w *= FRIENDLY_WEIGHT
        W.append(w)
        tot_g += w * (hs + as_)
        tot_w += w
    mu = tot_g / (2 * tot_w)

    for _ in range(ITERS):
        a_num = {t: SHRINK_PSEUDO * mu for t in teams}
        a_den = {t: SHRINK_PSEUDO * mu for t in teams}
        d_num = {t: SHRINK_PSEUDO * mu for t in teams}
        d_den = {t: SHRINK_PSEUDO * mu for t in teams}

        for (d, h, a, hs, as_, trn, neu), w in zip(matches, W):
            gh, ga = (GAMMA_HOME, GAMMA_AWAY) if not neu else (1.0, 1.0)
            a_num[h] += w * hs;    a_den[h] += w * mu * dfn[a] * gh
            a_num[a] += w * as_;   a_den[a] += w * mu * dfn[h] * ga
            d_num[a] += w * hs;    d_den[a] += w * mu * att[h] * gh
            d_num[h] += w * as_;   d_den[h] += w * mu * att[a] * ga

        att = {t: a_num[t] / a_den[t] for t in teams}
        dfn = {t: d_num[t] / d_den[t] for t in teams}

        # Normalize: attack mean = 1
        m_att = sum(att.values()) / len(att)
        att = {t: v / m_att for t, v in att.items()}
        dfn = {t: v * m_att for t, v in dfn.items()}

    return {"att": att, "dfn": dfn, "mu": mu}


def lambdas(model: dict, home: str, away: str, neutral: bool = True) -> tuple:
    """
    Expected goals for each side from the fitted rating model.

    neutral=True  → no venue adjustment (correct for 2026 WC for non-host nations)
    neutral=False → apply GAMMA_HOME / GAMMA_AWAY
    """
    att, dfn, mu = model["att"], model["dfn"], model["mu"]
    h = normalize(home)
    a = normalize(away)
    ah = att.get(h, 1.0)
    dh = dfn.get(h, 1.0)
    aa = att.get(a, 1.0)
    da = dfn.get(a, 1.0)
    gh, ga = (GAMMA_HOME, GAMMA_AWAY) if not neutral else (1.0, 1.0)
    lh = max(0.20, mu * ah * da * gh)
    la = max(0.20, mu * aa * dh * ga)
    return round(lh, 3), round(la, 3)
