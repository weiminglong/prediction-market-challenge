# Current Best Strategies

Verified submission candidates. Only update when a strategy beats current best on 500+ sims.

## Current Best
- strategy_current_best.py — Round4 convex inventory + trend + shock (Mean Edge +13.21, 500 sims)
  - Source: teams/lab/submission/strategy_round4_best_standalone.py

## Verification
```
uv run orderbook-pm run strategies/best/strategy_current_best.py --simulations 500 --workers 4
```
