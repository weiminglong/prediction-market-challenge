"""V17: V13 base + asymmetric fill response.

After a buy fill (arb likely sold, price dropped), widen bid spread
but keep ask tight to capture the retail that follows arb. Vice versa.
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
        self.last_buy_fill = False
        self.last_sell_fill = False

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

        # Competitor momentum
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            self.comp_momentum = 0.5 * self.comp_momentum + 0.5 * (mid - prev_mid)
        self.prev_bid = bid
        self.prev_ask = ask

        # Fill signals
        self.last_buy_fill = state.buy_filled_quantity > 0
        self.last_sell_fill = state.sell_filled_quantity > 0

        if self.last_buy_fill:
            self.fill_bias -= 1.0
        if self.last_sell_fill:
            self.fill_bias += 1.0
        self.fill_bias *= 0.5

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        # Asymmetric spread: widen the side that just got hit
        bid_spread = 3 if self.last_buy_fill else 2
        ask_spread = 3 if self.last_sell_fill else 2

        my_bid = max(1, int(round(fair - bid_spread)))
        my_ask = min(99, int(round(fair + ask_spread)))

        if my_bid >= my_ask:
            return actions

        max_inv = 30
        bid_size = max(0.25, 1.0 * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.25, 1.0 * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
