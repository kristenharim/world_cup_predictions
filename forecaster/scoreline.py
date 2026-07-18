"""
scoreline.py: Poisson scoreline engine with Dixon-Coles correction.

This is the heart of the prop-pricing system. One (lam_home, lam_away) pair →
one scoreline distribution → every question type, mutually coherent.

Dixon-Coles correction
----------------------
Independent-Poisson systematically underestimates the frequency of 0-0, 1-0,
0-1, 1-1 scorelines and overestimates rare high-scoring draws. The DC τ
adjustment reweights these four cells. ρ (rho) controls the correction
strength; the value 0.13 was calibrated on the martj42 international dataset
(lower RPS on the 2025+ test set vs ρ=0). You can re-estimate it with the
backtest in backtest.py.

Lambda inversion (Option 3 core)
---------------------------------
fit_lams_from_supremacy() recovers (lam_home, lam_away) from:
  - A calibrated home-win probability tilt from the XGBoost model
  - A total-goals anchor from the ratings model (or a sane default)

This keeps the result distribution faithful to the learned features while
ensuring every prop derived from the grid is internally consistent with it.

Ported and extended from sports-betting-fifa/engine/scripts/model.py.
Submission clamp removed (it was only for the crowd-relative scoring game).
"""

import math

# ── Poisson primitives ────────────────────────────────────────────────────────

def _pois_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def pois_ge(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    return max(0.0, 1.0 - sum(_pois_pmf(i, lam) for i in range(k)))


def clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


# ── Dixon-Coles τ correction ──────────────────────────────────────────────────
# Calibrated on international results; set to 0 to disable.
DC_RHO = 0.13

def _dc_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """
    Dixon-Coles correction factor for low-scoring cells.
    τ(0,0) = 1 − ρ·lh·la
    τ(1,0) = 1 + ρ·la
    τ(0,1) = 1 + ρ·lh
    τ(1,1) = 1 − ρ
    All other cells: τ = 1
    """
    if   i == 0 and j == 0: return 1 - rho * lh * la
    elif i == 1 and j == 0: return 1 + rho * la
    elif i == 0 and j == 1: return 1 + rho * lh
    elif i == 1 and j == 1: return 1 - rho
    return 1.0


def _grid(lh: float, la: float, maxg: int = 12, rho: float = DC_RHO) -> dict:
    """
    Full scoreline probability grid {(i, j): prob} with Dixon-Coles correction.
    Grid is renormalized to sum to 1 after the τ correction.
    """
    raw = {}
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            p = _pois_pmf(i, lh) * _pois_pmf(j, la) * _dc_tau(i, j, lh, la, rho)
            raw[(i, j)] = max(p, 0.0)
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ── W/D/L from grid ───────────────────────────────────────────────────────────

def result_probs(lh: float, la: float, rho: float = DC_RHO) -> tuple:
    """(home_win, draw, away_win) from the Dixon-Coles grid."""
    g = _grid(lh, la, rho=rho)
    hw = sum(p for (i, j), p in g.items() if i > j)
    dr = sum(p for (i, j), p in g.items() if i == j)
    aw = sum(p for (i, j), p in g.items() if i < j)
    return hw, dr, aw


# ── Lambda inversion: the Option 3 core ──────────────────────────────────────

def fit_lams_from_supremacy(p_home: float, p_away: float,
                            total_anchor: float, rho: float = DC_RHO) -> tuple:
    """
    Recover (lam_home, lam_away) from a calibrated home/away win tilt + a
    total-goals anchor.

    Instead of fitting to all three W/D/L numbers (which makes total goals
    depend on noisy draw probability), we:
      1. Pin  t = lam_home + lam_away = total_anchor   (from ratings model)
      2. Binary-search for  s = lam_home − lam_away  such that the grid's
         (home_win − away_win) matches the calibrated XGBoost tilt.

    This keeps the result tilt faithful to the learned features while anchoring
    total goals to the ratings model. The draw probability emerges from the grid
    rather than being forced (which is better, since XGBoost under-calls draws).

    Parameters
    ----------
    p_home, p_away  : calibrated probabilities from wdl_model.predict_proba
    total_anchor    : lam_home + lam_away from ratings.lambdas (sum of the two)
    rho             : Dixon-Coles correction strength

    Returns
    -------
    (lam_home, lam_away)
    """
    mu = max(total_anchor, 0.4)
    target = p_home - p_away      # the tilt we want to reproduce

    lo, hi = -mu * 0.99, mu * 0.99
    for _ in range(60):
        s = (lo + hi) / 2
        lh = max((mu + s) / 2, 0.05)
        la = max((mu - s) / 2, 0.05)
        hw, _, aw = result_probs(lh, la, rho=rho)
        if (hw - aw) < target:
            lo = s
        else:
            hi = s

    s  = (lo + hi) / 2
    lh = max((mu + s) / 2, 0.10)
    la = max((mu - s) / 2, 0.10)
    return round(lh, 3), round(la, 3)


# ── Symmetric averaging ───────────────────────────────────────────────────────

def symmetric_result_probs(lh: float, la: float, rho: float = DC_RHO) -> tuple:
    """
    Average (A vs B) and (B vs A) grids to cancel the arbitrary team-order
    artifact on neutral venues. Always returns (p_A, p_draw, p_B).
    """
    hw1, dr1, aw1 = result_probs(lh, la, rho)
    hw2, dr2, aw2 = result_probs(la, lh, rho)
    p_a  = (hw1 + aw2) / 2
    p_d  = (dr1 + dr2) / 2
    p_b  = (aw1 + hw2) / 2
    tot  = p_a + p_d + p_b
    return p_a / tot, p_d / tot, p_b / tot


# ── Peripheral count-prop helpers ─────────────────────────────────────────────

def _p_more(lam_a: float, lam_b: float, maxc: int = 45) -> float:
    """P(A > B) for two independent Poissons (fouls, corners, cards, SoT)."""
    pa = [_pois_pmf(i, lam_a) for i in range(maxc + 1)]
    pb = [_pois_pmf(j, lam_b) for j in range(maxc + 1)]
    cb = [0.0] * (maxc + 2)
    run = 0.0
    for j in range(maxc + 1):
        run += pb[j]
        cb[j] = run
    return sum(pa[i] * (cb[i - 1] if i >= 1 else 0.0) for i in range(maxc + 1))


# ── Default base rates for peripheral (non-goals) markets ────────────────────
# Per-team-per-match unless noted. Override per-question in params.
DEFAULTS = {
    "share_2h":  0.55,   # second halves carry ~55% of goals
    "corners":   5.0,    # expected corners per team
    "sot":       4.5,    # expected shots on target per team
    "fouls":     12.0,   # expected fouls per team
    "cards":     1.9,    # expected cards (yellow+red) per team
    "offsides":  1.8,    # expected offsides per team
    "p_pen":     0.27,   # P(penalty awarded in match)
    "p_red":     0.09,   # P(red card shown in match)
}

# Damping factors for player props (see model.py comment; pure Poisson runs hot)
PLAYER_RATE_DAMP  = 0.85
TEAM_SCORE_ZINFL  = 0.06   # extra zero-inflation for team_scores


def _m(match: dict, key: str):
    return match.get(key, DEFAULTS[key])


def _scope_factor(scope, share_2h: float) -> float:
    if scope in (None, "match", "full"):
        return 1.0
    if scope in ("1h", "first", "fh"):
        return 1.0 - share_2h
    if scope in ("2h", "second", "sh"):
        return share_2h
    raise ValueError(f"unknown scope: {scope!r}")


# ── Main question dispatcher ──────────────────────────────────────────────────

def question_prob(q: dict, match: dict) -> float:
    """
    q      : {"id": ..., "type": ..., "params": {...}}
    match  : {"lam_home": float, "lam_away": float, ...peripheral keys...}

    Returns the model's fair probability that the question resolves YES.
    Unknown types raise ValueError.
    """
    t    = q["type"]
    p    = q.get("params", {})
    share = _m(match, "share_2h")
    lh, la = match["lam_home"], match["lam_away"]
    scope   = p.get("scope", "match")
    f       = _scope_factor(scope, share)
    slh, sla = lh * f, la * f

    if t == "result":
        hw, dr, aw = result_probs(slh, sla)
        return {"home": hw, "draw": dr, "away": aw}[p["side"]]

    if t == "match_total_over":
        return pois_ge(int(p["line"]), slh + sla)

    if t == "match_total_under":
        return 1.0 - pois_ge(int(p["line"]) + 1, slh + sla)

    if t == "team_total_over":
        lam = slh if p["side"] == "home" else sla
        return pois_ge(int(p["line"]), lam)

    if t == "team_scores":
        lam = slh if p["side"] == "home" else sla
        return (1.0 - TEAM_SCORE_ZINFL) * pois_ge(1, lam)

    if t == "btts":
        return (1 - math.exp(-lh)) * (1 - math.exp(-la))

    if t == "ht_result":
        f1 = _scope_factor("1h", share)
        hw, dr, aw = result_probs(lh * f1, la * f1)
        return {"tie": dr, "home_lead": hw, "away_lead": aw}[p.get("outcome", "tie")]

    if t == "player_scores":
        xg = p["player_xg"] * (f if "scope" in p else 1.0) * PLAYER_RATE_DAMP
        return 1 - math.exp(-xg)

    if t == "player_sot_over":
        xsot = p["player_xsot"] * (f if "scope" in p else 1.0) * PLAYER_RATE_DAMP
        return pois_ge(int(p.get("n", 1)), xsot)

    if t == "match_sot_over":
        lam = (p.get("sot_home", _m(match, "sot")) +
               p.get("sot_away", _m(match, "sot"))) * f
        return pois_ge(int(p["n"]), lam)

    if t == "team_sot_over":
        lam = p.get("sot", _m(match, "sot")) * f
        return pois_ge(int(p["n"]), lam)

    if t == "team_corners_over":
        lam = p.get("corners", _m(match, "corners"))
        return pois_ge(int(p["n"]), lam)

    if t == "more_fouls":
        fh = p.get("fouls_home", _m(match, "fouls"))
        fa = p.get("fouls_away", _m(match, "fouls"))
        return _p_more(fa, fh) if p["side"] == "away" else _p_more(fh, fa)

    if t == "pen_or_red":
        pp = p.get("p_pen", _m(match, "p_pen"))
        pr = p.get("p_red", _m(match, "p_red"))
        return 1 - (1 - pp) * (1 - pr)

    if t == "team_cards_over":
        lam = p.get("cards", _m(match, "cards")) * f
        return pois_ge(int(p.get("n", 1)), lam)

    if t == "match_cards_over":
        lam = (p.get("cards_home", _m(match, "cards")) +
               p.get("cards_away", _m(match, "cards"))) * f
        return pois_ge(int(p["n"]), lam)

    if t == "offside_over":
        lam = p.get("offsides", _m(match, "offsides"))
        return pois_ge(int(p["n"]), lam)

    if t == "both_teams_sot":
        n = int(p.get("n", 1))
        lh_sot = p.get("sot_home", _m(match, "sot")) * f
        la_sot = p.get("sot_away", _m(match, "sot")) * f
        return pois_ge(n, lh_sot) * pois_ge(n, la_sot)

    if t == "btts_and_total":
        line = int(p.get("line", 3))
        g = _grid(lh, la)
        return sum(pr for (i, j), pr in g.items() if i >= 1 and j >= 1 and i + j >= line)

    if t == "team_first_goal":
        tot = slh + sla
        if tot <= 0:
            return 0.0
        lam_side = slh if p["side"] == "home" else sla
        return (lam_side / tot) * (1 - math.exp(-tot))

    if t == "half_goals_more":
        lam_tot = lh + la
        lam_2h  = lam_tot * share
        lam_1h  = lam_tot * (1.0 - share)
        which   = p.get("half", "2h")
        return (_p_more(lam_2h, lam_1h) if which in ("2h", "second", "sh")
                else _p_more(lam_1h, lam_2h))

    if t == "first_goal_and_team_scores":
        tot = lh + la
        if tot <= 0:
            return 0.0
        lam_first = lh if p.get("first_side", "home") == "home" else la
        p_first   = (lam_first / tot) * (1 - math.exp(-tot))
        sc_f      = _scope_factor(p.get("score_scope", "2h"), share)
        lam_sc    = (lh if p.get("score_side", "away") == "home" else la) * sc_f
        return p_first * (1 - math.exp(-lam_sc))

    if t == "more_sot":
        sh = p.get("sot_home", _m(match, "sot")) * f
        sa = p.get("sot_away", _m(match, "sot")) * f
        return _p_more(sa, sh) if p["side"] == "away" else _p_more(sh, sa)

    if t == "more_cards":
        ch = p.get("cards_home", _m(match, "cards")) * f
        ca = p.get("cards_away", _m(match, "cards")) * f
        return _p_more(ch, ca) if p["side"] == "home" else _p_more(ca, ch)

    if t == "more_corners":
        ch = p.get("corners_home", _m(match, "corners")) * f
        ca = p.get("corners_away", _m(match, "corners")) * f
        return _p_more(ch, ca) if p["side"] == "home" else _p_more(ca, ch)

    raise ValueError(f"unknown question type: {t!r}")


def compute_props(match: dict) -> dict:
    """
    Attach model_prob to every question in match["questions"] in-place.
    match must carry lam_home, lam_away, and optionally peripheral overrides.
    Returns the match dict.
    """
    for q in match.get("questions", []):
        q["model_prob"] = clamp(question_prob(q, match))
    return match
