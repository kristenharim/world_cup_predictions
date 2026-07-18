"""
names.py: canonical team name normalization.

The martj42 dataset, fixtures.csv, and user input all use slightly different
spellings. Everything funnels through normalize() before being stored or looked up.
"""

# martj42 dataset uses these spellings; map uncommon variants to them.
_CANON = {
    # user-friendly / FIFA / fixture variants -> martj42 canonical
    "usa":                    "United States",
    "united states":          "United States",
    "us":                     "United States",
    "iran":                   "IR Iran",
    "ir iran":                "IR Iran",
    "south korea":            "Korea Republic",
    "korea republic":         "Korea Republic",
    "republic of ireland":    "Ireland",
    "turkey":                 "Türkiye",
    "turkiye":                "Türkiye",
    "cape verde":             "Cabo Verde",
    "ivory coast":            "Côte d'Ivoire",
    "cote d'ivoire":          "Côte d'Ivoire",
    "côte d'ivoire":          "Côte d'Ivoire",
    "czech republic":         "Czechia",
    "czechia":                "Czechia",
    "curacao":                "Curaçao",
    "curaçao":                "Curaçao",
    "dr congo":               "DR Congo",
    "congo dr":               "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "bosnia":                 "Bosnia and Herzegovina",
    "bosnia-herzegovina":     "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
}

# fixtures.csv uses some additional spellings for the same teams
_FIXTURE_OVERRIDES = {
    "IR Iran":          "IR Iran",
    "Korea Republic":   "Korea Republic",
    "Türkiye":          "Türkiye",
    "Congo DR":         "DR Congo",
    "Côte d'Ivoire":    "Côte d'Ivoire",
    "Czechia":          "Czechia",
    "Curaçao":          "Curaçao",
    "USA":              "United States",
    "Cape Verde":       "Cabo Verde",
}


def normalize(name: str) -> str:
    """Return the canonical martj42 team name for any input spelling."""
    if not isinstance(name, str):
        return name
    stripped = name.strip()
    canon = _CANON.get(stripped.lower())
    if canon:
        return canon
    override = _FIXTURE_OVERRIDES.get(stripped)
    if override:
        return override
    return stripped


def normalize_fixture_side(raw: str) -> str:
    """Normalize a team name as it appears in fixtures.csv."""
    return normalize(raw.strip())
