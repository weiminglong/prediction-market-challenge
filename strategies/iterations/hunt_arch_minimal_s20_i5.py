"""Minimal strategy: CancelAll + bid at mid-2, ask at mid+2. Size=20, inv_cap=5. No signals."""

from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
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
        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + inv_skew
        my_bid = max(1, int(round(fair - 2)))
        my_ask = min(99, int(round(fair + 2)))

        if my_bid >= my_ask:
            return actions

        size = 20.0
        max_inv = 5
        bid_size = max(0.2, size * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.2, size * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
