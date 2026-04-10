"""Edge-price aware: tighter spread near boundaries, wider near 50.

Near price boundaries (1-20 or 80-99), the true value has less room
to move, so arb sweeps are smaller. We can be more aggressive there.
Near 50, maximum uncertainty = maximum arb risk = wider spread.

Also: near boundaries, inventory risk is lower because settlement
value is more predictable.
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
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        # Price-dependent spread: tighter near boundaries
        dist_from_edge = min(mid, 100 - mid)  # distance from nearest boundary
        if dist_from_edge < 15:
            base_spread = 1  # Very tight near boundaries
            base_size = 15.0
        elif dist_from_edge < 30:
            base_spread = 2  # Standard
            base_size = 10.0
        else:
            base_spread = 2  # Near 50, standard spread but could widen
            base_size = 10.0

        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        bid_spread = (base_spread + 1 if self.bid_cooldown > 0 else base_spread) + vol_widen
        ask_spread = (base_spread + 1 if self.ask_cooldown > 0 else base_spread) + vol_widen

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        my_bid = max(1, int(round(fair - bid_spread)))
        my_ask = min(99, int(round(fair + ask_spread)))

        if my_bid >= my_ask:
            return actions

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)

        bid_base = 0.8 if self.bid_cooldown > 0 else base_size
        ask_base = 0.8 if self.ask_cooldown > 0 else base_size

        max_inv = 10
        bid_size = max(0.5, round(bid_base * vol_scale * max(0.0, 1.0 - net_inv / max_inv), 1))
        ask_size = max(0.5, round(ask_base * vol_scale * max(0.0, 1.0 + net_inv / max_inv), 1))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
