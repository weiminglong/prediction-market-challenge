"""V16: Optimized V13 — stronger momentum, probability-aware spread.

Near 50%, dp/dz sensitivity is highest → more tick volatility → widen.
Near extremes, less volatility → can tighten.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.comp_momentum = 0.0

    def on_step(self, state: StepState):
        actions = [CancelAll()]

        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks

        if bid is None and ask is None:
            return actions
        if bid is None:
            bid = max(1, ask - 6)
        if ask is None:
            ask = min(99, bid + 6)

        mid = (bid + ask) / 2.0

        # Stronger momentum tracking
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            self.comp_momentum = 0.4 * self.comp_momentum + 0.6 * (mid - prev_mid)
        self.prev_bid = bid
        self.prev_ask = ask

        # Fill signal
        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
        self.fill_bias *= 0.5

        # Inventory skew
        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.7

        # Probability-aware spread: wider near 50, tighter at extremes
        distance_from_center = abs(mid - 50)
        if distance_from_center < 20:
            half_spread = 3
        else:
            half_spread = 2

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))

        if my_bid >= my_ask:
            return actions

        max_inv = 30
        bid_size = max(0.25, 1.0 * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.25, 1.0 * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
