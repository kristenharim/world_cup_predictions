"""
fixtures.py — load and look up 2026 World Cup fixtures.

Fixes the host-nation neutral bug from world_cup_predictions:
Mexico plays in Mexico City, USA in the US, Canada in Canada — those are NOT
neutral venues for those teams and the +60 Elo advantage / GAMMA factors
should apply. The neutral flag is per-fixture, not a global constant.
"""

import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(HERE, "..", "data_cache", "fixtures.csv")

from .names import normalize_fixture_side

# Teams playing at home venues in the 2026 WC (receive venue advantage)
# USA, Canada, Mexico are hosts; all others are neutral.
HOST_HOME_FIXTURES = {
    # (home_canonical, away_canonical) where home has genuine home advantage
    # Identified by city: US cities for USA, Canadian cities for Canada,
    # Mexican cities for Mexico.
}

# Stadiums in each host country (for auto-detection)
_US_STADIUMS = {
    "Los Angeles Stadium", "New York Stadium", "Dallas Stadium",
    "San Francisco Stadium", "Seattle Stadium", "Houston Stadium",
    "Miami Stadium", "Kansas City Stadium", "Philadelphia Stadium",
    "Atlanta Stadium", "Boston Stadium",
}
_CA_STADIUMS = {
    "Toronto Stadium", "Vancouver Stadium",
}
_MX_STADIUMS = {
    "Mexico City Stadium", "Guadalajara Stadium", "Estadio Guadalajara",
    "Monterrey Stadium",
}

_HOST_TEAMS = {
    "United States": _US_STADIUMS,
    "Canada":        _CA_STADIUMS,
    "Mexico":        _MX_STADIUMS,
}


def _is_home_venue(team_canon: str, stadium: str) -> bool:
    """True if the team is playing at one of their host-country stadiums."""
    stadiums = _HOST_TEAMS.get(team_canon, set())
    # substring match to handle minor naming variations
    for s in stadiums:
        if s.lower() in stadium.lower() or stadium.lower() in s.lower():
            return True
    return False


def _neutral_flag(home_canon: str, away_canon: str, stadium: str) -> bool:
    """
    Return True (neutral venue) unless one side is a host nation playing
    at their home stadium.
    """
    if _is_home_venue(home_canon, stadium):
        return False   # home team has genuine home advantage
    if _is_home_venue(away_canon, stadium):
        return False   # away team is actually home (their stadium, reversed listing)
    return True        # genuinely neutral


def load_fixtures() -> list:
    """Return a list of fixture dicts from data_cache/fixtures.csv."""
    fixtures = []
    with open(FIXTURES_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teams = str(row.get("teams", ""))
            if " v " not in teams:
                continue
            left, right = [p.strip() for p in teams.split(" v ", 1)]
            home_raw, away_raw = left, right
            home = normalize_fixture_side(home_raw)
            away = normalize_fixture_side(away_raw)
            stadium = row.get("stadium", "").strip()
            neutral = _neutral_flag(home, away, stadium)
            fixtures.append({
                "match_number": row.get("match_number", "").strip(),
                "group":        row.get("group", "").strip(),
                "stadium":      stadium,
                "date":         row.get("date_dt", "").strip(),
                "home_disp":    home_raw,
                "away_disp":    away_raw,
                "home":         home,
                "away":         away,
                "neutral":      neutral,
            })
    return fixtures


def find_fixture(team_a: str, team_b: str) -> dict | None:
    """
    Find the fixture for the two named teams (order doesn't matter).
    Returns None if not found.
    """
    a = normalize_fixture_side(team_a)
    b = normalize_fixture_side(team_b)
    for fx in load_fixtures():
        if (fx["home"] == a and fx["away"] == b) or \
           (fx["home"] == b and fx["away"] == a):
            return fx
    # Fuzzy fallback: case-insensitive substring
    al, bl = a.lower(), b.lower()
    for fx in load_fixtures():
        hl, awl = fx["home"].lower(), fx["away"].lower()
        if (al in hl or hl in al) and (bl in awl or awl in bl):
            return fx
        if (bl in hl or hl in bl) and (al in awl or awl in al):
            return fx
    return None


def list_team_names() -> list:
    """Sorted list of all team names in the fixture list (display form)."""
    names = set()
    for fx in load_fixtures():
        # Skip knockout placeholder rows
        for side in (fx["home_disp"], fx["away_disp"]):
            if not any(w in side.lower() for w in ["winner", "runner", "third", "place", "group", "match"]):
                names.add(side)
    return sorted(names)
