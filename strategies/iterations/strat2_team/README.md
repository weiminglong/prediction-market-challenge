# Strat2 Team Strategies

Strategies developed by the strat2 agent team (5 parallel agents) exploring new directions for mean edge > 10.

## Leaderboard

| Strategy | Mean Edge | Retail | Arb | Sims | Builder |
|----------|-----------|--------|-----|------|---------|
| strategy_arb_hunter.py | +17.10 (pending 500-sim verify) | +36.29 | -19.19 | 200 | arb-hunter |
| strategy_multilevel.py | +14.60 / +20.82 (500s) | +29.83 / +42.90 | -15.23 / -22.08 | 200/500 | multilevel-dev |
| strategy_combined.py | +14.16 | +38.18 | -24.01 | 200 | retail-dev |
| strategy_model_based.py | +10.60 | +21.17 | -10.57 | 100 | model-dev |
| strategy_retail_capture.py | +1.39 | +5.35 | -3.96 | 200 | retail-dev |
| strategy_bayesian.py | WIP | -- | -- | -- | bayesian-dev |

## Key Techniques Discovered

1. **Multi-level quoting** (multilevel-dev): L1 + L2 (+3 ticks, 40%) + L3 (+6 ticks, 20%) + L4 + inside-spread sentinels. Each level captures retail that doesn't reach inner levels.
2. **Trend-based asymmetric L1 sizing** (arb-hunter): When trend is strong, reduce size on the side the arb will sweep. Saved +3.3 arb edge while keeping retail flat.
3. **Model-vol size boost** (retail-dev): Boost base_size up to 2x when prob near extremes (low per-step vol = safe to size up). Drives highest retail capture (+38.18).
4. **Vol-adaptive spread + sizing** (retail-dev): Compute per-step probability volatility via phi(probit(p)); quote tight with big size at extremes, wide with small size near 0.5.
5. **Shock detection** (from round4): Detect large mid moves, pause losing side for N steps, shrink size to 15%.
6. **Fill bias tracking** (from v15): Infer fair value direction from fill patterns (buy fills = prob dropped, sell fills = prob rose).
7. **Convex inventory management** (from round4): Hermite spline + cubic tail for smooth inventory skew.

## Architecture

All strategies build on the round4 core (fill bias + vol EMA + trend + shock detection + convex inventory). The key differentiators are:
- **arb-hunter**: round4 + multilevel (L1-L4) + trend-based asymmetric L1 sizing for arb defense
- **multilevel-dev**: round4 + 3-level ladder (L1/L2/L3) + inside-spread sentinels
- **retail-dev (combined)**: round4 + model-based vol boost for aggressive sizing at prob extremes
- **model-dev**: inside-competitor-only quoting + fill-aware shock + probability vol filtering

## Reference Strategies (from main)

- `strategies/mean_edge_lab/submission/strategy_round4_best_standalone.py` -- +13.21 (500 sims, very stable)
- `strategies/iterations/hunt_arch_best_v15.py` -- +6.17 (50 sims)
