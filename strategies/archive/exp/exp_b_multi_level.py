"""Exp B: Multi-level quoting on safe side.

Place orders at ±2 AND ±1 from fair when not on cooldown.
The ±1 order is tighter (more retail, more arb risk) but small.
The ±2 order is the main workhorse. Net effect: capture more retail
at two price points.
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

        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        if self.bid_cooldown > 0: self.bid_cooldown -= 1
        if self.ask_cooldown > 0: self.ask_cooldown -= 1

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)
        max_inv = 30
        inv_bid_scale = max(0.0, 1.0 - net_inv / max_inv)
        inv_ask_scale = max(0.0, 1.0 + net_inv / max_inv)

        on_bid_cooldown = self.bid_cooldown > 0
        on_ask_cooldown = self.ask_cooldown > 0

        # Level 1: main orders at ±2 (or ±3 on cooldown)
        l1_bid_spread = (3 if on_bid_cooldown else 2) + vol_widen
        l1_ask_spread = (3 if on_ask_cooldown else 2) + vol_widen
        l1_bid = max(1, int(round(fair - l1_bid_spread)))
        l1_ask = min(99, int(round(fair + l1_ask_spread)))

        l1_bid_base = 0.8 if on_bid_cooldown else 2.0
        l1_ask_base = 0.8 if on_ask_cooldown else 2.0
        l1_bid_size = max(0.2, l1_bid_base * vol_scale * inv_bid_scale)
        l1_ask_size = max(0.2, l1_ask_base * vol_scale * inv_ask_scale)

        if l1_bid < l1_ask:
            cost = l1_bid * 0.01 * l1_bid_size
            if cost <= state.free_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=l1_bid, quantity=l1_bid_size))
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=l1_ask, quantity=l1_ask_size))

        # Level 2: tight orders at ±1 (only when not on cooldown and low vol)
        if not on_bid_cooldown and not on_ask_cooldown and vol_widen == 0:
            l2_bid = max(1, int(round(fair - 1)))
            l2_ask = min(99, int(round(fair + 1)))
            l2_size = 0.8 * vol_scale

            if l2_bid < l2_ask and l2_bid != l1_bid and l2_ask != l1_ask:
                l2_bid_size = max(0.2, l2_size * inv_bid_scale)
                l2_ask_size = max(0.2, l2_size * inv_ask_scale)
                cost = l2_bid * 0.01 * l2_bid_size
                if cost <= state.free_cash:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=l2_bid, quantity=l2_bid_size))
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=l2_ask, quantity=l2_ask_size))

        return actions
