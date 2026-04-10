# Mean Edge Strategy Lab

Shared workspace for parallel strategy exploration and benchmarking.

It contains:
- standalone strategy candidates (by round),
- benchmark artifacts (`*.json`) for reproducibility,
- and submission-ready path guidance for fast handoff.

## Current Submission Candidates

Use these in priority order:

1. `team_round15/strategy_team_q2_p1_holdout_guard.py` (current fast-round winner)
2. `team_round14/strategy_team_p1_o4_toxtrend_guard.py` (conservative holdout fallback)
3. `team_round13/strategy_team_o4_m2_trend_depth_asym.py` (stable baseline fallback)

Recent fast mixed checks (`seed_start=0,300,900`, 100 sims each):
- `q2`: robust mean `36.586164`
- `p1`: robust mean `36.482608`
- `o4`: robust mean `36.479582`

## Directory Layout

- `submission/`
  - Stable handoff files and default benchmark candidate lists.
- `team_round5/` ... `team_round17/`
  - Parallel round outputs (strategies + ranking artifacts).
- `team_round3/`, `team_round4/`
  - Earlier team rounds.
- `grind/`
  - Search/tuning scripts and historical artifacts.
- `directions/`, `team_directions/`
  - Earlier architecture exploration phases.

## Evaluation Protocol

Quick screen:
- `uv run python strategies/mean_edge_lab/compare_strategies.py <paths...> --simulations 100 --workers 8 --seed-starts 0,300`

Fast holdout check:
- `uv run python strategies/mean_edge_lab/compare_strategies.py <paths...> --simulations 120 --workers 8 --seed-starts 900,1200`

Mixed anti-overfit tie-break:
- `uv run python strategies/mean_edge_lab/compare_strategies.py <paths...> --simulations 100 --workers 8 --seed-starts 0,300,900`

Sandbox smoke:
- `uv run orderbook-pm run <strategy_path> --sandbox --simulations 10 --json`

## Rules for New Strategies

- Keep files standalone for sandbox compatibility:
  - allowed imports: `orderbook_pm_challenge.strategy`, `orderbook_pm_challenge.types`, stdlib.
- Do not overwrite other agent outputs.
- Store ranking JSONs with the corresponding round folder.
