"""Vol-adaptive spread: ±1 during calm markets, ±3 during volatile.

Hypothesis: In calm markets, the arb is less active and retail fills dominate.
Use ±1 spread with huge size to capture maximum flow. In volatile markets,
widen to ±3 and shrink size to avoid arb losses.
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
        inv_skew = -net_inv * 0.08

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        # Vol-adaptive: calm = tight + big, volatile = wide + small
        is_calm = self.vol_estimate < 0.5

        if is_calm:
            base_spread = 1
            base_size = 25.0
            cooldown_spread = 2
            cooldown_size = 1.0
            max_inv = 6
        else:
            base_spread = 3
            base_size = 5.0
            cooldown_spread = 4
            cooldown_size = 0.5
            max_inv = 10

        bid_spread = cooldown_spread if self.bid_cooldown > 0 else base_spread
        ask_spread = cooldown_spread if self.ask_cooldown > 0 else base_spread

        my_bid = max(1, int(round(fair - bid_spread)))
        my_ask = min(99, int(round(fair + ask_spread)))

        if my_bid >= my_ask:
            return actions

        bid_base = cooldown_size if self.bid_cooldown > 0 else base_size
        ask_base = cooldown_size if self.ask_cooldown > 0 else base_size

        bid_size = max(0.2, bid_base * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.2, ask_base * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
