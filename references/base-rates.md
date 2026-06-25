# Base rates — priors for peripheral markets

When you don't have a specific number, these are the defaults baked into
`model.DEFAULTS`. They're per-team-per-match unless noted. Override per question
in `params` whenever you have a sharper read (style, lineup, ref tendencies).

| market | default | source / reasoning |
|---|---|---|
| 2nd-half share of goals (`share_2h`) | 0.55 | second halves average more goals (fatigue, chasing) |
| corners per team | 5.0 | top-flight / international average band 4.5–5.5 |
| shots on target per team | 4.5 | average band 4–5 |
| fouls committed per team | 12.0 | average band 10–14; underdogs/low-blocks foul more |
| P(penalty awarded in match) | 0.27 | ~0.25–0.30 of matches see a penalty |
| P(red card in match) | 0.09 | ~8–10% of matches see a red |
| P(penalty OR red) | ≈0.34 | 1−(1−0.27)(1−0.09) |
| HT level (tie) | ~0.40–0.45 | depends on supremacy; the model computes it from xG |

## Adjustments worth making by hand
- **Low-block underdog** (deep defensive line): more fouls and corners conceded,
  fewer offsides drawn by the favorite, lower total goals. Bump the underdog's
  `fouls`, trim total xG.
- **Derby / high-stakes / strict referee**: bump `p_red` and `p_pen`.
- **Star striker** vs a weak back line: raise his `player_xg` (share of a higher
  team xG) and `player_xsot`.
- **Game state**: a favorite expected to lead will see the underdog chase in the
  2nd half — nudges 2nd-half goals and shots up for the trailing side.

These priors are deliberately conservative. As the ledger grows, revisit them
against realized outcomes (the calibration table in `cup.py report` will show
systematic misses — e.g. if your 2nd-half-goal questions are always under, raise
`share_2h`).
