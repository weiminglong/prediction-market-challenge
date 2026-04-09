"""Multi-level quoting: orders at BOTH ±1 and ±2 simultaneously.

Hypothesis: Inner level (±1, small size) captures high-frequency retail.
Outer level (±2, larger size) provides depth and captures more flow.
Combined, more total retail fills with manageable arb exposure.
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

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)
        max_inv = 8

        bid_inv_scale = max(0.0, 1.0 - net_inv / max_inv)
        ask_inv_scale = max(0.0, 1.0 + net_inv / max_inv)

        # --- Inner level: ±1 spread, small size ---
        inner_bid = max(1, int(round(fair - 1)))
        inner_ask = min(99, int(round(fair + 1)))

        inner_bid_size = 3.0 * vol_scale * bid_inv_scale if self.bid_cooldown == 0 else 0.0
        inner_ask_size = 3.0 * vol_scale * ask_inv_scale if self.ask_cooldown == 0 else 0.0

        # --- Outer level: ±2 spread, larger size ---
        outer_bid = max(1, int(round(fair - 2)))
        outer_ask = min(99, int(round(fair + 2)))

        outer_bid_size = max(0.2, 10.0 * vol_scale * bid_inv_scale)
        outer_ask_size = max(0.2, 10.0 * vol_scale * ask_inv_scale)

        # On cooldown, only quote outer level with reduced size
        if self.bid_cooldown > 0:
            outer_bid_size = max(0.2, 0.8 * vol_scale * bid_inv_scale)
        if self.ask_cooldown > 0:
            outer_ask_size = max(0.2, 0.8 * vol_scale * ask_inv_scale)

        # Place inner orders (if they don't cross)
        if inner_bid < inner_ask:
            if inner_bid_size > 0:
                cost = inner_bid * 0.01 * inner_bid_size
                if cost <= state.free_cash:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=inner_bid, quantity=inner_bid_size))

            if inner_ask_size > 0:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=inner_ask, quantity=inner_ask_size))

        # Place outer orders (if they don't cross)
        if outer_bid < outer_ask:
            remaining_cash = state.free_cash - (inner_bid * 0.01 * inner_bid_size if inner_bid < inner_ask and inner_bid_size > 0 else 0)
            cost = outer_bid * 0.01 * outer_bid_size
            if cost <= remaining_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=outer_bid, quantity=outer_bid_size))

            actions.append(PlaceOrder(side=Side.SELL, price_ticks=outer_ask, quantity=outer_ask_size))

        return actions
