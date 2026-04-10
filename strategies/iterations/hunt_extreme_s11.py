"""v21 base, NO vol scaling at all. Size=12, inv=10."""
from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None

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
        my_bid = max(1, int(round(fair - 2)))
        my_ask = min(99, int(round(fair + 2)))
        if my_bid >= my_ask:
            return actions
        base_size = 12.0
        max_inv = 10
        bid_size = max(0.2, base_size * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.2, base_size * max(0.0, 1.0 + net_inv / max_inv))
        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions
