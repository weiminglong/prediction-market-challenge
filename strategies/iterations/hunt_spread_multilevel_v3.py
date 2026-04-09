"""Multi-level v3: Tiny inner size (1.0), strong arb protection.

V2 got retail edge 16.6 but arb edge -12.6. The inner ±1 level feeds arb.
Fix: inner size 1.0 (just enough to attract retail), outer ±2 with size 10.
Also: stronger fill_bias decay (0.3 instead of 0.5) and longer cooldown (3).
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
            self.bid_cooldown = 3
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
            self.ask_cooldown = 3
        self.fill_bias *= 0.3

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        vol_scale = max(0.3, 1.0 - self.vol_estimate * 0.5)
        max_inv = 10

        bid_inv_scale = max(0.0, 1.0 - net_inv / max_inv)
        ask_inv_scale = max(0.0, 1.0 + net_inv / max_inv)

        cash_remaining = state.free_cash

        # --- Inner level: ±1 spread, tiny size (only when no cooldown and calm) ---
        if self.bid_cooldown == 0 and self.ask_cooldown == 0 and self.vol_estimate < 0.8:
            inner_bid = max(1, int(round(fair - 1)))
            inner_ask = min(99, int(round(fair + 1)))

            if inner_bid < inner_ask:
                inner_bid_size = round(1.0 * bid_inv_scale, 1)
                inner_ask_size = round(1.0 * ask_inv_scale, 1)

                if inner_bid_size >= 0.5:
                    cost = inner_bid * 0.01 * inner_bid_size
                    if cost <= cash_remaining:
                        actions.append(PlaceOrder(side=Side.BUY, price_ticks=inner_bid, quantity=inner_bid_size))
                        cash_remaining -= cost

                if inner_ask_size >= 0.5:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=inner_ask, quantity=inner_ask_size))

        # --- Outer level: ±2 spread (±3 on cooldown) ---
        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        outer_bid_spread = (3 if self.bid_cooldown > 0 else 2) + vol_widen
        outer_ask_spread = (3 if self.ask_cooldown > 0 else 2) + vol_widen

        outer_bid = max(1, int(round(fair - outer_bid_spread)))
        outer_ask = min(99, int(round(fair + outer_ask_spread)))

        if outer_bid < outer_ask:
            outer_bid_base = 0.8 if self.bid_cooldown > 0 else 10.0
            outer_ask_base = 0.8 if self.ask_cooldown > 0 else 10.0

            outer_bid_size = max(0.5, round(outer_bid_base * vol_scale * bid_inv_scale, 1))
            outer_ask_size = max(0.5, round(outer_ask_base * vol_scale * ask_inv_scale, 1))

            cost = outer_bid * 0.01 * outer_bid_size
            if cost <= cash_remaining:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=outer_bid, quantity=outer_bid_size))

            actions.append(PlaceOrder(side=Side.SELL, price_ticks=outer_ask, quantity=outer_ask_size))

        return actions
