"""V4: Iterate on V3 — wider spread sweep to find optimal.

V3 used ±5 and got -0.42 edge. Let's try ±6 and ±4 to bracket the optimum.
Also add time-awareness: as steps decrease, true probability becomes more
extreme and less volatile in tick-space, so we can tighten the spread.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0

    def on_step(self, state: StepState):
        actions = [CancelAll()]

        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks

        if bid is None and ask is None:
            return actions
        if bid is None:
            bid = max(1, ask - 8)
        if ask is None:
            ask = min(99, bid + 8)

        mid = (bid + ask) / 2.0

        # Fill signal
        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
        self.fill_bias *= 0.5

        # Inventory skew
        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.05

        fair = mid + self.fill_bias + inv_skew

        # Time-adaptive spread: wider early (more volatile), tighter late
        total_steps = state.step + state.steps_remaining
        progress = state.step / max(1, total_steps)
        half_spread = 6 if progress < 0.7 else 4

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))

        if my_bid >= my_ask:
            return actions

        max_inv = 60
        bid_size = max(0.5, 4.0 * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.5, 4.0 * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
