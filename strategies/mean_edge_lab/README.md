# Mean Edge Strategy Lab

This directory is a shared workspace for parallel strategy exploration.
It keeps all standalone strategy variants, grinder scripts, team rounds,
and promoted submission candidates in one place.

## Current Best Strategy

- Primary submit file: `submission/strategy_round4_best_standalone.py`
- Latest verified metrics (200 sims, 8 workers):
  - `seed_start=0`: `Mean Edge 10.728356`
  - `seed_start=300`: `Mean Edge 17.437226`
  - `seed_start=600`: `Mean Edge 14.107487`
- Sandbox-safe (no local project imports beyond `orderbook_pm_challenge.strategy`
  and `orderbook_pm_challenge.types`).

## Directory Layout

- `submission/`
  - Stable standalone strategies ready for handoff/submission.
- `team_round3/`, `team_round4/`
  - Parallel agent outputs and ranking artifacts per round.
- `team_directions/`
  - Earlier team architecture explorations.
- `grind/`
  - Search/tuning scripts plus exported candidate files and result artifacts.
- `directions/`
  - Architecture-level direction prototypes.
- `optimize.py`, `explore_directions.py`
  - Generic exploration drivers from early lab phases.

## Parallel Agent Workflow

1. Pick a unique output path (example: `team_round5/strategy_team_g2_<theme>.py`).
2. Keep strategy standalone:
   - Allowed imports should stay within `orderbook_pm_challenge.strategy`,
     `orderbook_pm_challenge.types`, and stdlib.
3. Run baseline eval:
   - `uv run orderbook-pm run <strategy_path> --simulations 200 --workers 8 --seed-start 0`
4. Run robustness eval:
   - repeat for `--seed-start 300` and `--seed-start 600`
5. Run sandbox compatibility check:
   - `uv run orderbook-pm run <strategy_path> --simulations 200 --workers 8 --seed-start 0 --sandbox`
6. Record results in the corresponding team round ranking JSON.
7. Promote winners into `submission/` as standalone files.

## Notes

- Strategy files that import local helper modules can fail sandbox mode.
- Grinder and ranking JSON artifacts are intentionally kept for reproducibility.
- If multiple agents are active, never write to another agent's output file.
