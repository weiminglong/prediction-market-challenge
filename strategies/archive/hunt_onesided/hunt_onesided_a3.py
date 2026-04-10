"""arch_v29 base + REDUCE (not zero) filled side to 2.0 for 1 step, safe stays 10.
Maybe zeroing is too aggressive - try just reducing to a small amount."""

from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_estimate = 0.0
        self.reduce_bid = False
        self.reduce_ask = False

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
            self.vol_estimate = 0.9 * self.vol_estimate + 0.1 * abs(move)
        self.prev_bid = bid
        self.prev_ask = ask

        new_reduce_bid = False
        new_reduce_ask = False
        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.5
            new_reduce_bid = True
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.5
            new_reduce_ask = True
        self.fill_bias *= 0.4

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew

        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        my_bid = max(1, int(round(fair - 2 - vol_widen)))
        my_ask = min(99, int(round(fair + 2 + vol_widen)))

        if my_bid >= my_ask:
            self.reduce_bid = new_reduce_bid
            self.reduce_ask = new_reduce_ask
            return actions

        vol_scale = max(0.1, 1.0 - self.vol_estimate * 1.5)
        base_size = 10.0
        reduced_size = 2.0
        max_inv = 10

        if self.reduce_bid:
            bid_size = reduced_size * vol_scale
        else:
            bid_size = max(0.2, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))

        if self.reduce_ask:
            ask_size = reduced_size * vol_scale
        else:
            ask_size = max(0.2, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        self.reduce_bid = new_reduce_bid
        self.reduce_ask = new_reduce_ask

        if bid_size > 0.05:
            cost = my_bid * 0.01 * bid_size
            if cost <= state.free_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        if ask_size > 0.05:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
