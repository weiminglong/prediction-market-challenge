# Codex Strategy Lab Design

Date: 2026-04-09

## Goal

Build an isolated strategy development track that competes with the existing
Claude-authored strategies by maximizing `Mean Edge` under the repository's
default evaluation command:

```bash
uv run orderbook-pm run <strategy_path>
```

This work must not modify or depend on Claude's in-progress strategy files.

## Scope

Create a new strategy workspace under `strategies/codex-lab/` that contains:

- candidate strategy implementations
- local helper code shared only by Codex strategies
- a benchmark runner for comparing candidate strategies under repo defaults
- focused tests for deterministic strategy logic
- one promoted best strategy file for final evaluation

Out of scope:

- changes to simulator mechanics
- changes to challenge scoring
- edits to Claude's `strategies/strategy_final.py` or `strategies/iterations/*`
- speculative infrastructure outside the isolated Codex strategy workspace

## Constraints

- Benchmark target is highest `Mean Edge`
- Evaluation protocol is the repo default command with default simulation count,
  steps, and seed range
- Work must remain isolated from Claude's current uncommitted work
- Strategy code must use only the public participant API
- The final result should be reproducible from files checked into this repo

## Existing Baseline

Claude's current `strategies/strategy_final.py` is an adaptive market maker
with these core behaviors:

- competitor midpoint anchoring
- unconditional `CancelAll()` each step
- fill-based cooldown widening
- volatility-scaled quote size
- inventory skew in quote placement

This is a credible baseline, but the unconditional cancel-and-repost loop gives
up queue priority every step. The most promising avenue for improvement is to
preserve queue priority when quotes remain acceptable while still widening or
shutting off toxic sides during adverse-selection regimes.

## Strategy Hypothesis

The best path is a small strategy family rather than a single speculative
implementation. The lab will compare:

1. A conservative selective-replace market maker
2. A regime-switching market maker
3. A promoted best strategy chosen by benchmark results

The leading hypothesis is that a regime-switching strategy with selective quote
replacement will beat the current Claude baseline by improving the tradeoff
between retail capture and arbitrage loss.

## Architecture

### Folder Layout

```text
strategies/codex-lab/
  benchmark.py
  helpers.py
  strategy_conservative.py
  strategy_regime.py
  strategy_best.py
tests/
  test_codex_lab_helpers.py
```

The exact filenames may change during implementation, but the structure should
separate:

- benchmark orchestration
- reusable deterministic quote logic
- individual strategy variants
- unit tests for helper behavior

### Components

`helpers.py`
- deterministic logic only
- fair-value anchoring from visible competitor quotes
- inventory skew logic
- quote retention and replacement decisions
- side shutdown rules under inventory/collateral pressure
- regime classification from recent fills and competitor movement

`strategy_conservative.py`
- baseline improvement over Claude's approach
- selective cancel/replace
- modest inventory skew
- simpler rules and lower parameter count

`strategy_regime.py`
- full regime switching
- calm vs toxic market handling
- optional one-sided quoting when a side looks adverse
- stronger inventory and collateral discipline

`strategy_best.py`
- either a thin alias of the winning implementation or a final tuned copy
- the main strategy to use in head-to-head evaluation

`benchmark.py`
- runs candidate strategies with the repo-default CLI command
- parses `Mean Edge`, `Mean Retail Edge`, `Mean Arb Edge`, failures, and final
  wealth
- compares multiple candidates in a single run
- supports repeated runs over multiple seed starts when explicitly requested,
  but defaults to the repo-default benchmark

## Data Flow

Per simulation step, strategy logic will use only:

- `competitor_best_bid_ticks`
- `competitor_best_ask_ticks`
- `buy_filled_quantity`
- `sell_filled_quantity`
- `yes_inventory`
- `no_inventory`
- `free_cash`
- `own_orders`

The strategy will maintain small internal state across steps, such as:

- prior competitor midpoint
- short-horizon movement estimate
- recent one-sided fill pressure
- lightweight regime state or cooldown counters

The strategy will then:

1. Infer a fair anchor from visible competitor quotes
2. Adjust for inventory and recent fill toxicity
3. Decide whether the current regime is calm, uncertain, or toxic
4. Retain, cancel, or replace resting quotes selectively
5. Size each side subject to inventory and collateral limits

## Regime Logic

The regime-switching strategy should classify market conditions with simple,
robust rules rather than many parameters.

Proposed regimes:

- Calm
  - competitor midpoint stable
  - no recent one-sided fill shock
  - quote both sides with narrower offsets and normal size

- Uncertain
  - moderate competitor movement or mild one-sided fills
  - widen both sides modestly
  - reduce size

- Toxic
  - sharp competitor movement, repeated same-side fills, or large inventory
    imbalance
  - widen heavily or disable the toxic side
  - keep or favor the inventory-reducing side only

The strategy should degrade smoothly rather than oscillate wildly between
regimes.

## Quote Management Rules

The core improvement over Claude's baseline is selective replacement.

Rules:

- Keep an existing order if its price is within a small tolerance of the new
  target and its remaining quantity is still acceptable
- Cancel and replace only when the target moved materially
- Use stable client order ids so each side can be managed independently
- Avoid `CancelAll()` unless both sides must be reset

This design intentionally prioritizes queue retention when the environment is
stable.

## Inventory And Collateral Rules

The strategy must treat cash and position limits as first-class controls.

Rules:

- quote size must be computed from `free_cash`, not raw `cash`
- uncovered asks must be sized using their collateral cost
- YES and NO inventory should each have hard caps
- quote sizes should shrink as inventory becomes more imbalanced
- severe imbalance should disable the risk-increasing side

The strategy should reserve some free cash instead of fully deploying all
available collateral, preserving flexibility to reprice or continue quoting.

## Testing

Implementation will follow TDD for deterministic helper behavior.

Tests will cover:

- quote target selection from visible competitor quotes
- inventory skew directionality
- side disablement under high inventory imbalance
- selective replacement decisions
- size reduction under collateral pressure

These tests are not intended to prove profitability. They exist to lock down the
decision logic so strategy iterations do not silently regress.

## Benchmarking

Primary benchmark:

```bash
uv run orderbook-pm run strategies/codex-lab/strategy_best.py
```

Comparison benchmark:

- run the same command against Claude's `strategies/strategy_final.py`
- compare `Mean Edge` directly

During development, the benchmark runner may also compare:

- `Mean Retail Edge`
- `Mean Arb Edge`
- failure count

These are diagnostic metrics only. The winner is the strategy with the highest
`Mean Edge` under the repo-default command.

## Error Handling

Strategy code must avoid invalid participant actions that would fail a
simulation.

The implementation should:

- clamp quote prices into valid tick bounds
- avoid zero or negative post-quantization sizes
- avoid placing orders that exceed free collateral
- handle missing competitor bid or ask values gracefully
- remain valid when one side is intentionally disabled

## Success Criteria

The project is successful when:

- the repo contains an isolated `strategies/codex-lab/` workspace
- at least two candidate strategies are benchmarkable
- deterministic helper logic is covered by tests
- one promoted Codex strategy is runnable via the repo-default CLI command
- that promoted strategy achieves a higher default-command `Mean Edge` than
  Claude's current `strategies/strategy_final.py`

## Risks

- overfitting to one default seed range
- adding too many interacting parameters
- reducing arb losses at the expense of losing all retail flow
- preserving queue too aggressively and leaving stale quotes exposed

Mitigations:

- keep regime rules simple
- use a conservative candidate and a higher-upside candidate
- track retail edge and arb edge during development
- benchmark against Claude repeatedly before promoting a winner

## Implementation Notes

- The Codex strategy work should remain in its own folder
- Shared logic should be extracted only when it is reused by multiple candidate
  strategies
- Tests should target helper functions, not stochastic profitability
- The final promoted strategy should be easy to run without the benchmark helper
