# Question types: the model's vocabulary

Every question on a slate maps to one `type` understood by `model.py`. Each
takes `params`. Goal-based types derive from the match's `lam_home` / `lam_away`
(scaled by `scope`); peripheral types fall back to base-rate defaults in
`model.DEFAULTS` (override per question in `params`).

`scope` is one of `match` (default), `1h`, `2h`. The split is `share_2h` (default
0.55, second halves carry more goals).

| type | params | resolves YES when | notes |
|---|---|---|---|
| `result` | `side`: home/away/draw | that side wins (in scope) | from scoreline grid |
| `match_total_over` | `line` (int), `scope` | total goals ≥ line in scope | "2+ in 2nd half" → line 2, scope 2h |
| `match_total_under` | `line` (int), `scope` | total goals ≤ line in scope | "2 or fewer total goals" → line 2 |
| `team_total_over` | `side`, `line`, `scope` | that side scores ≥ line | |
| `team_scores` | `side`, `scope` | that side scores ≥1 | |
| `btts` | — | both teams score (match) | |
| `ht_result` | `outcome`: tie/home_lead/away_lead | that halftime state | forced to 1h |
| `player_scores` | `player_xg`, opt `scope` | player scores ≥1 | xg ≈ his share of team xG |
| `player_sot_over` | `n`, `player_xsot`, opt `scope` | player ≥ n shots on target | striker xsot ≈ 1.0 |
| `match_sot_over` | `n`, `scope`, opt `sot_home`/`sot_away` | total SoT ≥ n in scope | defaults = DEFAULTS["sot"] each |
| `team_sot_over` | `side`, `n`, `scope`, opt `sot` | side SoT ≥ n | |
| `team_corners_over` | `side`, `n`, opt `corners` | side corners ≥ n (match) | |
| `more_corners` | `side`, `scope`, opt `corners_home`/`corners_away` | that side has more corners | P(A>B) in scope |
| `more_fouls` | `side`, opt `fouls_home`/`fouls_away` | that side commits more fouls | Skellam P(A>B) |
| `more_sot` | `side`, `scope`, opt `sot_home`/`sot_away` | that side has more SoT | P(A>B) in scope |
| `more_cards` | `side`, `scope`, opt `cards_home`/`cards_away` | that side gets more cards | P(A>B) |
| `team_first_goal` | `side`, `scope` | that side scores the first goal in scope | rate-ratio × P(≥1 goal) |
| `half_goals_more` | opt `half` (2h/1h, dflt 2h) | that half has strictly more total goals than the other | P(A>B) on the share_2h goal split; ties drag it below 50% |
| `first_goal_and_team_scores` | `first_side`, `score_side`, opt `score_scope` (dflt 2h) | `first_side` scores the match's first goal AND `score_side` scores in scope | product of the two legs (indep-Poisson approx) |
| `pen_or_red` | opt `p_pen`, `p_red` | a penalty OR a red card | mostly base-rate |
| `team_cards_over` | `n`, `scope`, opt `cards` | that side ≥ n cards in scope | defaults = DEFAULTS["cards"] |
| `match_cards_over` | `n`, `scope`, opt `cards_home`/`cards_away` | total cards (both teams) ≥ n in scope | "4+ total cards" → n 4 |
| `offside_over` | `n`, opt `offsides` | a side caught offside ≥ n times (match) | default offsides 1.8/team |
| `both_teams_sot` | `n`, `scope`, opt `sot_home`/`sot_away` | both teams ≥ n SoT in scope | product of two Poisson tails |
| `btts_and_total` | opt `line` (default 3) | both teams score AND total ≥ line (match) | from the scoreline grid |

## Contrarian tags (drive the `contrarian` pod only)
Add to a question's `tags` list:
- `favorite`: crowd overrates this favorite; pod regresses the crowd toward 50%.
- `star_player`: name-recognition prop; pod fades the crowd down.
- `base_rate:<p>`: pod regresses the crowd toward `<p>` (e.g. `base_rate:0.36`
  for penalty-or-red). Use for questions with a strong historical anchor.

## Worked mapping
The 10-question Canada vs Bosnia slate is fully encoded in
`match-template.json`. Copy it as your starting point and swap in the new
match's numbers.
