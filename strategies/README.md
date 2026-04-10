# Prediction Market Strategy Lab

## Quick Start for New Teams
1. Read techniques/README.md to learn what's been discovered
2. Read best/ to understand current top strategies
3. Create your team dir: strategies/teams/<team-name>/
4. Build on proven techniques, test with 200+ sims, verify on 500

## Global Leaderboard (verified 200+ sims)
| Strategy | Mean Edge | Retail | Arb | Team |
|----------|-----------|--------|-----|------|
| teams/strat2/strategy_arb_hunter.py | +17.10 | +36.29 | -19.19 | strat2 |
| teams/strat2/strategy_multilevel.py | +14.60 | +29.83 | -15.23 | strat2 |
| teams/strat2/strategy_combined.py | +14.16 | +38.18 | -24.01 | strat2 |
| best/strategy_current_best.py (round4) | +13.21 | +24.37 | -11.17 | lab |

## Run a strategy
```
uv run orderbook-pm run strategies/best/strategy_current_best.py --simulations 200 --workers 4
```

## Verification protocol
- Quick test: 20 sims (iteration only)
- Report: 200 sims minimum
- Verify: 500 sims before claiming as "best"
- Cross-validate: run twice at different sim counts
