"""V5: Two-tier quoting.

Place a small order close to fair (captures retail) and a larger order
further out (safety net, captures big moves). The close order takes arb
hits but is small. The far order rarely gets hit by arb.
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

        # Tier 1: close to fair, small size (retail capture)
        t1_half = 3
        t1_bid = max(1, int(round(fair - t1_half)))
        t1_ask = min(99, int(round(fair + t1_half)))

        # Tier 2: far from fair, larger size (safety)
        t2_half = 6
        t2_bid = max(1, int(round(fair - t2_half)))
        t2_ask = min(99, int(round(fair + t2_half)))

        max_inv = 50

        # Tier 1 orders (small)
        t1_bid_size = max(0.5, 1.5 * max(0.0, 1.0 - net_inv / max_inv))
        t1_ask_size = max(0.5, 1.5 * max(0.0, 1.0 + net_inv / max_inv))

        if t1_bid < t1_ask:
            cost = t1_bid * 0.01 * t1_bid_size
            if cost <= state.free_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=t1_bid, quantity=t1_bid_size))
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=t1_ask, quantity=t1_ask_size))

        # Tier 2 orders (larger, further out) — skip if overlapping with tier 1
        t2_bid_size = max(0.5, 3.0 * max(0.0, 1.0 - net_inv / max_inv))
        t2_ask_size = max(0.5, 3.0 * max(0.0, 1.0 + net_inv / max_inv))

        if t2_bid < t1_bid:
            cost = t2_bid * 0.01 * t2_bid_size
            if cost <= state.free_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=t2_bid, quantity=t2_bid_size))
        if t2_ask > t1_ask:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=t2_ask, quantity=t2_ask_size))

        return actions
