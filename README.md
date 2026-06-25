# 2026 FIFA World Cup match forecaster

Predict **win / draw / loss** probabilities and a **full prop slate** (totals,
BTTS, halftime, fouls, cards, player shots, and more) for any 2026 World Cup
match — all derived from one internally consistent scoreline distribution.

Fork-friendly ML project: readable modules, no notebooks, no framework soup.

```bash
pip install -r requirements.txt
python predict.py "Switzerland" "Canada"
```

```
================================================================
  Switzerland vs Canada
  2026-06-24  ·  Group B  ·  BC Place Vancouver
================================================================
  xG (home)            1.44
  xG (away)            1.26
----------------------------------------------------------------
  Switzerland                win    42.7%
  Draw                              22.5%
  Canada                     win    34.8%
----------------------------------------------------------------
  PICK: Switzerland  (42.7%)   [TOSS-UP]
================================================================

  PROP SLATE
  --------------------------------------------------------------
  Will Switzerland win the match?                   42.7%
  Will there be 3 or more total goals?              75.2%
  Will both teams score?                            54.7%
  ...
```

## How it works

Two models, one coherent output (Option 3 hybrid):

```
martj42 results (data_cache/results.csv)
         │
         ├─→ Elo + form + rest + h2h  →  XGBoost  →  calibrated home/away tilt
         │
         └─→ opponent-adjusted Poisson ratings  →  total goal anchor
                              │
                fit_lams_from_supremacy()  →  (lam_home, lam_away)
                              │
                Dixon-Coles scoreline grid (ρ = 0.13)
                              │
                W/D/L + every prop question type
```

**Why not just XGBoost W/D/L?** A classifier can't price goal props. **Why not
fit both λs to all three W/D/L numbers?** Draw probability is noisy — that
corrupts over/under and BTTS. So XGBoost sets the *tilt*, ratings set *total
goals*, and the grid derives everything else consistently.

Features (all strictly pre-match, no look-ahead):

- **Elo** from every international since 2006 (+60 home bonus where applicable)
- **Recent form** — win rate and goal diff over last 5 / 10 games
- **Rest days** and **head-to-head** history
- **Host-nation fix** — USA, Mexico, and Canada get real home advantage in their
  host-country stadiums (not treated as neutral)

XGBoost outputs are **isotonic-calibrated** on a held-out validation slice so
the probabilities are honest, not overconfident softmax scores.

## Quick start

```bash
pip install -r requirements.txt

python predict.py "Spain" "Cabo Verde"
python predict.py USA Paraguay
python predict.py Switzerland Canada --questions examples/match.json
python batch_predict.py
```

Team order doesn't matter; common spellings work (`Iran` → `IR Iran`, `USA` →
`United States`). Charts save to `predictions/<date>/`.

## Refresh before a match

The [martj42/international_results](https://github.com/martj42/international_results)
dataset updates within a day of every real match. Form, rest days, and h2h are
computed from that file — no other data sources needed.

**Run with `--refresh` within 24 hours of kickoff** so the model sees the latest
results:

```bash
python predict.py --refresh "Switzerland" "Canada"
# or
python -c "from forecaster.data import refresh; refresh()"
```

## Custom question slates

Pass a JSON file with a `questions` list (see `references/question-types.md`):

```json
{
  "home": "Switzerland", "away": "Canada",
  "questions": [
    {"id": "q1", "type": "match_total_over",
     "params": {"line": 3, "scope": "match"},
     "text": "Will there be 4 or more total goals?"},
    {"id": "q2", "type": "more_fouls",
     "params": {"side": "away", "fouls_home": 11.0, "fouls_away": 13.0},
     "text": "Will Canada commit more fouls than Switzerland?"}
  ]
}
```

Count props (fouls, cards, corners, SoT) use base-rate defaults unless you
override params per question. See `references/base-rates.md`.

## File layout

```
forecaster/
  data.py          fetch / load martj42 results
  features.py      Elo + form + rest + h2h
  wdl_model.py     XGBoost + isotonic calibration
  ratings.py       opponent-adjusted Poisson → total goal anchor
  scoreline.py     Dixon-Coles grid + lambda inversion + all prop types
  fixtures.py      2026 schedule + host-nation neutral flag
  chart.py         branded PNG chart
data_cache/
  fixtures.csv     2026 WC schedule (static)
  results.csv      auto-downloaded, gitignored
references/
  question-types.md
  base-rates.md
predict.py         single-match CLI
batch_predict.py   all-fixtures batch run
```

## What it doesn't do (yet)

- No injuries, lineups, or player-level xG beyond what you pass in question params
- Player-to-score props need `player_xg` inputs — treat as low-confidence without them
- Knockout placeholders in `fixtures.csv` are skipped until real teams are known

## Data

- **Results:** [martj42/international_results](https://github.com/martj42/international_results)
- **Fixtures:** official 2026 schedule in `data_cache/fixtures.csv`

## License

MIT — do whatever you want with it.
