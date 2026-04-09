"""V1: Wider spread + smaller size + inventory management.

Key changes from starter:
- Half-spread of 3 (vs 2) to reduce arb adverse selection
- Size 3 (vs 5) to limit per-fill losses
- Inventory skewing to avoid directional risk accumulation
- Fill-based momentum signal to detect arb activity and fade
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
            bid = max(1, ask - 6)
        if ask is None:
            ask = min(99, bid + 6)

        mid = (bid + ask) / 2.0

        # Fill signal: if we got filled, the arb likely moved the price
        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
        self.fill_bias *= 0.5  # decay

        # Inventory skew: push quotes away from our position
        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.04

        fair = mid + self.fill_bias + inv_skew

        # Wider spread than starter
        half_spread = 3
        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))

        if my_bid >= my_ask:
            return actions

        # Smaller size, inventory-adjusted
        max_inv = 50
        bid_size = max(0.5, 3.0 * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.5, 3.0 * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
