"""Zero filled side + boost opposite: after fill, zero filled side, boost opposite to 15."""

from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_estimate = 0.0
        self.bid_boost = 0
        self.ask_boost = 0

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

        buy_filled = state.buy_filled_quantity > 0
        sell_filled = state.sell_filled_quantity > 0

        if buy_filled:
            self.fill_bias -= 1.0
            self.bid_boost = 0     # zero bid side
            self.ask_boost = 2     # boost ask side for 2 steps
        if sell_filled:
            self.fill_bias += 1.0
            self.ask_boost = 0     # zero ask side
            self.bid_boost = 2     # boost bid side for 2 steps
        self.fill_bias *= 0.4

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew

        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        my_bid = max(1, int(round(fair - 2 - vol_widen)))
        my_ask = min(99, int(round(fair + 2 + vol_widen)))

        if my_bid >= my_ask:
            return actions

        vol_scale = max(0.1, 1.0 - self.vol_estimate * 1.5)
        base_size = 10.0
        max_inv = 10

        # Boost opposite side after fill
        bid_mult = 15.0 if self.bid_boost > 0 else base_size
        ask_mult = 15.0 if self.ask_boost > 0 else base_size

        bid_size = max(0.2, bid_mult * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.2, ask_mult * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if self.bid_boost > 0:
            self.bid_boost -= 1
        if self.ask_boost > 0:
            self.ask_boost -= 1

        # Place orders (skip filled side for 1 step - the boost/zero already handles sizing)
        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
