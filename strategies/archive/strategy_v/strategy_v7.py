"""V7: Cooldown after fills.

Base spread ±4 (between V1's ±3 and V3's ±5). After any fill, widen to
±7 for a cooldown period, then return to base. This avoids repeated
arb damage while being tighter than V3 during calm periods.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.cooldown = 0

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

        # Fill signal and cooldown
        if state.buy_filled_quantity > 0 or state.sell_filled_quantity > 0:
            self.cooldown = 3
        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
        self.fill_bias *= 0.5

        if self.cooldown > 0:
            self.cooldown -= 1

        # Inventory skew
        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.05

        fair = mid + self.fill_bias + inv_skew

        # Spread: tighter normally, wider on cooldown
        half_spread = 7 if self.cooldown > 0 else 4

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))

        if my_bid >= my_ask:
            return actions

        max_inv = 50
        bid_size = max(0.5, 3.5 * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.5, 3.5 * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
