# Technique Catalog

Proven techniques discovered across all teams. New teams should study these before building.

## Core Architecture (from round4/v15)
Every top strategy builds on this foundation:
- **Fill bias tracking**: When buys fill -> bias -= hit; sells fill -> bias += hit; bias *= decay. Infers fair value direction. (v15: hit=1.0, decay=0.4)
- **Volatility EMA**: vol = decay * vol + (1-decay) * |mid_move|. Tracks market turbulence. (v15: decay=0.8)
- **Inventory skew**: Shift fair value by -coeff * net_inventory to flatten positions. (v15: coeff=0.06)
- **Fair value**: mid + fill_bias + inv_skew [+ trend]. Quote around this, not raw midpoint.

## Spread & Sizing
- **Base spread 2 ticks** from fair value works best. Wider loses retail, tighter loses to arb.
- **Large base size (10-15 shares)** captures more retail per fill. Size matters more than spread.
- **Vol-scaled sizing**: size *= max(floor, 1.0 - vol * coeff). Shrink in volatile markets.

## Advanced Techniques

### Shock Detection (+2-3 edge)
Detect large mid moves (> threshold). Pause quoting on losing side for N steps. Shrink size to 15%.
- Trigger: |move| > max(min_trigger, vol_mult * max(vol, vol_floor))
- Duration: 4 steps. Size mult: 0.15.

### Multi-Level Quoting (+2-4 edge)
Place orders at multiple price depths:
- L1: base spread, full size
- L2: +3 ticks wider, 40% size
- L3: +6 ticks wider, 20% size
- Inside-spread sentinels: 2.5 shares during calm periods with wide competitor spreads
Each level captures retail that doesn't reach inner levels.

### Trend Following (+1-2 edge)
EMA trend signal: trend = decay * trend + alpha * move. Add trend * weight to fair value.
- Captures drift direction. (round4: alpha=0.17, decay=0.65, weight=1.0)

### Convex Inventory Management (+1 edge)
Replace linear inventory skew with Hermite spline smoothing below soft cap + cubic tail penalty above.
Smoother transitions avoid abrupt quote jumps.

### Model-Vol Size Boost (+2-3 edge)
Compute per-step probability volatility: pvol = phi(probit(p)) / (sigma * sqrt(T)) * sigma * 100.
When pvol is low (prob near 0 or 1), boost size up to 2x. Safe because arb risk is minimal at extremes.

### Trend-Based Asymmetric L1 Sizing (+3 edge on arb)
When trend is strong, reduce L1 size on the side the arb will sweep. Keeps retail flat while cutting arb damage.

## What Doesn't Work
- Penny-ahead of competitor: too close to fair, arb eats all edge
- Wide quotes behind competitor: retail never reaches us
- Pure Bayesian inference: signals too noisy to beat simple fill bias
- Stopping quoting late game: misses free edge at locked extremes
- Excessive inventory penalty (>0.10): shifts quotes outside competitor, kills retail
