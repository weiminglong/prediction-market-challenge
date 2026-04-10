"""No-cooldown v8: v5 + reduce vol_scale sensitivity.

V5 uses vol_scale with 0.7 multiplier. During vol, this reduces size a lot.
Try 0.3 multiplier (less vol sensitivity) since we're already no-cooldown.
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
        self.vol_estimate = 0.0

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

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.comp_momentum = 0.5 * self.comp_momentum + 0.5 * move
            self.vol_estimate = 0.9 * self.vol_estimate + 0.1 * abs(move)
        self.prev_bid = bid
        self.prev_ask = ask

        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.5
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.5
        self.fill_bias *= 0.4

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        # Less vol sensitivity on size
        vol_scale = max(0.4, 1.0 - self.vol_estimate * 0.3)
        max_inv = 10

        bid_inv_scale = max(0.0, 1.0 - net_inv / max_inv)
        ask_inv_scale = max(0.0, 1.0 + net_inv / max_inv)

        cash_remaining = state.free_cash

        # Inner ±1 during calm
        if self.vol_estimate < 0.5:
            inner_bid = max(1, int(round(fair - 1)))
            inner_ask = min(99, int(round(fair + 1)))
            if inner_bid < inner_ask:
                cost = inner_bid * 0.01 * 0.5
                if cost <= cash_remaining:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=inner_bid, quantity=0.5))
                    cash_remaining -= cost
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=inner_ask, quantity=0.5))

        # Outer ±2
        spread = 2 + vol_widen
        my_bid = max(1, int(round(fair - spread)))
        my_ask = min(99, int(round(fair + spread)))

        if my_bid < my_ask:
            bid_size = max(0.5, round(10.0 * vol_scale * bid_inv_scale, 1))
            ask_size = max(0.5, round(10.0 * vol_scale * ask_inv_scale, 1))

            cost = my_bid * 0.01 * bid_size
            if cost <= cash_remaining:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
