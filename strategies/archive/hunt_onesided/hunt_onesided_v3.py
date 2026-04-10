"""One-sided quoting v3: Ultra aggressive - threshold=1, reducing_size=100, accum_size=0.

Go one-sided as soon as |inv| > 1. Massive size on reducing side.
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
        self.bid_cooldown = 0
        self.ask_cooldown = 0
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
            self.fill_bias -= 1.0
            self.bid_cooldown = 2
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
            self.ask_cooldown = 2
        self.fill_bias *= 0.5

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.10

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        vol_widen = 1 if self.vol_estimate > 1.0 else 0

        threshold = 1
        flat_size = 20.0
        reducing_size = 100.0

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        if net_inv > threshold:
            # Long YES: only sell with massive size, tight spread
            ask_spread = 1 + vol_widen
            my_ask = min(99, int(round(fair + ask_spread)))
            if my_ask >= 2:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=reducing_size))

        elif net_inv < -threshold:
            # Short: only buy with massive size, tight spread
            bid_spread = 1 + vol_widen
            my_bid = max(1, int(round(fair - bid_spread)))
            cost = my_bid * 0.01 * reducing_size
            if cost <= state.free_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=reducing_size))

        else:
            # Flat: both sides, big size, spread ±2
            bid_spread = (3 if self.bid_cooldown > 0 else 2) + vol_widen
            ask_spread = (3 if self.ask_cooldown > 0 else 2) + vol_widen

            my_bid = max(1, int(round(fair - bid_spread)))
            my_ask = min(99, int(round(fair + ask_spread)))

            if my_bid >= my_ask:
                return actions

            bid_size = flat_size if self.bid_cooldown == 0 else 0.8
            ask_size = flat_size if self.ask_cooldown == 0 else 0.8

            cost = my_bid * 0.01 * bid_size
            if cost <= state.free_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
