from __future__ import annotations

import math

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _norm_ppf_approx(p: float) -> float:
    if p <= 0.001:
        return -3.09
    if p >= 0.999:
        return 3.09
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t: float) -> float:
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


class Strategy(BaseStrategy):
    """Vol-adaptive single-level market maker.

    Uses the accurate probability sensitivity formula:
    per_step_vol = phi(Phi^{-1}(p)) / (sigma*sqrt(T)) * sigma * 100

    Near probability extremes (0 or 1): low vol -> tight spread, big size.
    Near 0.5: high vol -> wide spread, small size.

    Parameters tuned via sweep:
    - hs_mult = 2.0 (half-spread = ceil(2.0 * per_step_vol), min 2)
    - size_base = 6.0 (size = 6.0 / max(1.0, vol), clamped 2-8)
    - inv_soft = 12, inv_range = 10
    - inv_skew = 0.06 * (1 + 5*t_frac)
    - end_cutoff = 0.92
    """

    def on_step(self, state: StepState):
        actions: list = [CancelAll()]

        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        if comp_bid is None and comp_ask is None:
            return actions

        if comp_bid is not None and comp_ask is not None:
            mid = (comp_bid + comp_ask) / 2.0
        elif comp_bid is not None:
            mid = comp_bid + 2.0
        else:
            mid = comp_ask - 2.0

        net_inv = state.yes_inventory - state.no_inventory
        total_steps = state.step + state.steps_remaining
        t_frac = state.step / total_steps if total_steps > 0 else 0.0
        steps_rem = max(1, state.steps_remaining)

        if t_frac > 0.92:
            return actions

        # Accurate per-step probability volatility
        prob_est = max(0.005, min(0.995, mid / 100.0))
        z = _norm_ppf_approx(prob_est)
        phi_z = _norm_pdf(z)
        total_sigma = 0.02 * math.sqrt(steps_rem)
        per_step_vol = phi_z / max(0.01, total_sigma) * 0.02 * 100.0
        per_step_vol = max(0.05, min(8.0, per_step_vol))

        # Inventory skew: aggressively flatten
        inv_skew = net_inv * 0.06 * (1.0 + 5.0 * t_frac)

        # Half-spread: 2 sigma safety from estimated fair
        hs = max(2, math.ceil(2.0 * per_step_vol))

        # Size: inversely proportional to vol
        size = max(2.0, min(8.0, 6.0 / max(1.0, per_step_vol)))

        bid = int(round(mid - hs - inv_skew))
        ask = int(round(mid + hs - inv_skew))
        bid = max(1, min(98, bid))
        ask = max(2, min(99, ask))
        if bid >= ask:
            return actions

        buy_size = size
        sell_size = size

        if net_inv > 12:
            buy_size *= max(0.0, 1.0 - (net_inv - 12) / 10.0)
        if net_inv < -12:
            sell_size *= max(0.0, 1.0 - (abs(net_inv) - 12) / 10.0)

        if buy_size >= 1.0:
            cost = bid / 100.0 * buy_size
            if state.free_cash > cost + 0.5:
                actions.append(
                    PlaceOrder(side=Side.BUY, price_ticks=bid, quantity=round(buy_size, 2))
                )

        if sell_size >= 1.0:
            actions.append(
                PlaceOrder(side=Side.SELL, price_ticks=ask, quantity=round(sell_size, 2))
            )

        return actions
