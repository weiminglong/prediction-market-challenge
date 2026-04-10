# Autoresearch Changelog: strategy_final

## Experiment 0 — baseline
**Score:** 6/18 (33.3%)
**Change:** None — original strategy
**Metrics:** edge=0.59, retail=1.87, arb=-1.28
**Passing:** Profitable 3/3, No failures 3/3
**Failing:** Strong edge, Very strong edge, Arb controlled, Strong retail

## Experiment 1 — keep
**Score:** 9/18 (50.0%)
**Change:** base_size 1.0 → 1.2
**Reasoning:** Retail (1.87) close to 2.0 threshold, bigger size captures more
**Result:** Retail jumped to 2.24, now passing "Strong retail". Arb worsened to -1.51

## Experiment 2 — keep
**Score:** 9/18 (50.0%)
**Change:** vol_scale coefficient 0.6 → 0.7
**Reasoning:** Compensate for larger size by scaling down more aggressively during vol
**Result:** Arb improved from -1.51 to -1.46. Marginal but in the right direction

## Experiment 3 — discard
**Score:** 9/18 (50.0%)
**Change:** fill_bias step 1.0→1.5, decay 0.5→0.4
**Reasoning:** Stronger fill signal should shift quotes further from fill side
**Result:** No improvement, slightly worse metrics. Fill bias was already tuned

## Experiment 4 — discard
**Score:** 9/18 (50.0%)
**Change:** cooldown spread ±3 → ±4
**Reasoning:** Wider post-fill spread reduces repeated arb damage
**Result:** Arb improved (-1.40) but retail dropped more (-0.13). Net negative

## Experiment 5 — keep
**Score:** 11/18 (61.1%)
**Change:** Asymmetric sizing — 1.3 on safe side, 0.8 on cooldown side
**Reasoning:** Capture more retail on the safe side, limit arb on the vulnerable side
**Result:** "Strong edge" now passes 2/3. Key architectural change

## Experiment 6 — discard
**Score:** 11/18 (61.1%)
**Change:** cooldown-side size 0.8 → 0.5
**Reasoning:** Further reduce arb exposure on vulnerable side
**Result:** Arb improved 0.04 but retail dropped 0.04. Net neutral

## Experiment 7 — keep
**Score:** 12/18 (66.7%)
**Change:** safe-side size 1.3 → 1.5
**Reasoning:** Push "Strong edge" to 3/3 consistency
**Result:** "Strong edge" 3/3. Edge jumped to 0.95 avg. Retail 2.68

## Experiment 8 — discard
**Score:** 11/18 (61.1%)
**Change:** Gap-based size scaling (reduce when competitor gap > 3)
**Reasoning:** Wide gap = more arb activity = reduce exposure
**Result:** Too aggressive — reduced both retail and edge. Different signal not additive

## Experiment 9 — discard
**Score:** 12/18 (66.7%)
**Change:** momentum coefficient 0.5 → 0.8
**Reasoning:** Better fair value tracking should improve quote accuracy
**Result:** Nearly identical to Exp 7. Momentum was already well-tuned

## Experiment 10 — discard
**Score:** 9/18 (50.0%)
**Change:** Late-game spread tightening (±1 in last 30%)
**Reasoning:** Less tick-space volatility late means can tighten safely
**Result:** DISASTER. Arb exploded to -2.07. Late game is NOT less volatile

## Experiment 11 — keep
**Score:** 12/18 (66.7%)
**Change:** safe-side size 1.5 → 1.7
**Reasoning:** Push edge from 0.95 toward 1.2 threshold
**Result:** Edge 1.07 avg. Two runs near 1.1. On track

## Experiment 12 — keep
**Score:** 14/18 (77.8%)
**Change:** safe-side size 1.7 → 2.0
**Reasoning:** Edge should scale linearly, targeting 1.25+
**Result:** "Very strong edge" passes 2/3! Edge 1.24 avg

## Experiment 13 — keep
**Score:** 15/18 (83.3%)
**Change:** safe-side size 2.0 → 2.2
**Reasoning:** Push weakest run above 1.2
**Result:** "Very strong edge" 3/3! Edge 1.35 avg. All runs above 1.2

## Experiment 14 — discard
**Score:** 12/18 (66.7%)
**Change:** inv_skew 0.06 → 0.10
**Reasoning:** Flatten inventory faster to improve retail/arb ratio
**Result:** Too aggressive — shifted quotes too much, reduced retail capture

## Experiment 15 — keep
**Score:** 15/18 (83.3%)
**Change:** Vol-adaptive spread widening (+1 tick when vol > 1.0)
**Reasoning:** Additional arb protection during volatile periods
**Result:** Edge 1.38 (up from 1.35), arb -2.38 (improved from -2.42). More robust

---

## Final Analysis

**Baseline:** 33.3% → **Final: 83.3%** (150% improvement)
**Experiments:** 16 total, 9 kept, 7 discarded (56% keep rate)

**Top changes that helped most:**
1. **Asymmetric sizing** (Exp 5) — bigger on safe side, smaller post-fill. Single biggest architectural improvement
2. **Size scaling** (Exp 7→13) — incrementally pushing safe-side from 1.0 to 2.2 unlocked "Strong edge" and "Very strong edge"
3. **Vol-adaptive spread widening** (Exp 15) — widening during volatile periods reduces arb without hurting retail

**Structural ceiling:** "Arb controlled" (>-1.2) is mathematically incompatible with "Very strong edge" (>1.2). The retail/arb ratio is ~1.6x, so edge >1.2 requires arb < -2.1. Satisfying both would require ratio >2.0, which isn't achievable with the available signals.
