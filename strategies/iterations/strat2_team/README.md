# Strat2 Team Strategies

Strategies developed by the strat2 agent team exploring new directions for mean edge > 10.

## Leaderboard (as of session)

| Strategy | Mean Edge | Retail | Arb | Sims | Builder |
|----------|-----------|--------|-----|------|---------|
| strategy_multilevel.py | +14.60 (200s), +20.82 (500s unverified) | +29.83 / +42.90 | -15.23 / -22.08 | 200/500 | multilevel-dev |
| strategy_combined.py | +14.16 | +38.18 | -24.01 | 200 | retail-dev |
| strategy_arb_hunter.py | +12.68 | +24.58 | -11.90 | 200 | arb-hunter |
| strategy_model_based.py | +10.60 | +21.17 | -10.57 | 100 | model-dev |
| strategy_retail_capture.py | +1.39 | +5.35 | -3.96 | 200 | retail-dev |
| strategy_bayesian.py | in progress | — | — | — | bayesian-dev |
| model_quoting_strategy.py | -3.04 | — | — | 100 | researcher |

## Key Techniques Discovered

1. **Vol-adaptive spread + sizing** (retail-dev): Compute per-step probability volatility; quote tight with big size at extremes, wide with small size near 0.5
2. **Multi-level quoting** (multilevel-dev): L1 + L2 (+3 ticks, 40%) + L3 (+6 ticks, 20%) + inside-spread sentinels
3. **Model-vol boost** (retail-dev): Boost base_size up to 2x when prob near extremes (low vol = safe to size up)
4. **Shock detection** (from round4): Detect large mid moves, pause losing side for N steps
5. **Fill bias tracking** (from v15): Infer direction from fill patterns
6. **Convex inventory management** (from round4): Hermite spline + cubic tail for smooth inventory skew
7. **Arb defense** (arb-hunter): Asymmetric sizing, wider spreads in high-vol, cooldowns after arb fills

## Reference Strategies (from main)

- `strategies/mean_edge_lab/submission/strategy_round4_best_standalone.py` — +13.21 (500 sims, very stable)
- `strategies/hunt_arch_best_v15.py` — +6.17 (50 sims)
