"""V8: Competitor momentum signal for directional quoting.

Track changes in competitor best bid/ask to infer arb direction.
When arb was buying (asks moved up), lean into selling.
When arb was selling (bids moved down), lean into buying.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.prev_bid = None
        self.prev_ask = None
        self.fill_bias = 0.0
        self.comp_momentum = 0.0

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

        # Competitor movement momentum
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            self.comp_momentum = 0.6 * self.comp_momentum + 0.4 * (mid - prev_mid)
        self.prev_bid = bid
        self.prev_ask = ask

        # Fill signal
        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
        self.fill_bias *= 0.5

        # Inventory skew
        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.05

        # Fair value with momentum
        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        half_spread = 5
        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))

        if my_bid >= my_ask:
            return actions

        max_inv = 60
        bid_size = max(0.5, 4.0 * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.5, 4.0 * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
