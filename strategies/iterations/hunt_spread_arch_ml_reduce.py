"""Arch base + inner ±1 only on inventory-reducing side + ±2 outer."""

from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
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
            self.vol_estimate = 0.9 * self.vol_estimate + 0.1 * abs(move)
        self.prev_bid = bid
        self.prev_ask = ask

        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
        self.fill_bias *= 0.4

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew

        vol_scale = max(0.1, 1.0 - self.vol_estimate * 1.5)
        max_inv = 10

        cash_remaining = state.free_cash

        # Inner ±1: ONLY on reducing side
        inner_bid = max(1, int(round(fair - 1)))
        inner_ask = min(99, int(round(fair + 1)))

        if inner_bid < inner_ask:
            if net_inv > 1:
                # Long: inner ask to sell, size proportional to inventory
                inner_size = round(min(3.0, net_inv * 0.3) * vol_scale, 1)
                if inner_size >= 0.2:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=inner_ask, quantity=inner_size))
            elif net_inv < -1:
                # Short: inner bid to buy
                inner_size = round(min(3.0, abs(net_inv) * 0.3) * vol_scale, 1)
                if inner_size >= 0.2:
                    cost = inner_bid * 0.01 * inner_size
                    if cost <= cash_remaining:
                        actions.append(PlaceOrder(side=Side.BUY, price_ticks=inner_bid, quantity=inner_size))
                        cash_remaining -= cost

        # Outer ±2, size 10
        my_bid = max(1, int(round(fair - 2)))
        my_ask = min(99, int(round(fair + 2)))

        if my_bid < my_ask:
            bid_size = max(0.2, 10.0 * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
            ask_size = max(0.2, 10.0 * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

            cost = my_bid * 0.01 * bid_size
            if cost <= cash_remaining:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
