"""Multi-level + asymmetric: inner tight spread on reducing side, outer levels always.

Combines multi-level quoting with inventory-driven asymmetry.
When long, inner ask at ±1 (eager to sell), inner bid suppressed.
Outer levels at ±2 always present. Very tight inv cap.
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

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)
        max_inv = 6
        bid_inv_scale = max(0.0, 1.0 - net_inv / max_inv)
        ask_inv_scale = max(0.0, 1.0 + net_inv / max_inv)

        cash_used = 0.0

        # --- Inner level: only on inventory-REDUCING side ---
        # Long (net_inv > 0): inner ask at ±1 to sell quickly
        # Short (net_inv < 0): inner bid at ±1 to buy quickly
        inner_bid_price = max(1, int(round(fair - 1)))
        inner_ask_price = min(99, int(round(fair + 1)))

        if net_inv < -1 and self.bid_cooldown == 0:
            # Short: eager to buy, place inner bid
            inner_bid_size = min(5.0, abs(net_inv)) * vol_scale
            if inner_bid_size > 0.2 and inner_bid_price < inner_ask_price:
                cost = inner_bid_price * 0.01 * inner_bid_size
                if cost <= state.free_cash - cash_used:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=inner_bid_price, quantity=inner_bid_size))
                    cash_used += cost

        if net_inv > 1 and self.ask_cooldown == 0:
            # Long: eager to sell, place inner ask
            inner_ask_size = min(5.0, abs(net_inv)) * vol_scale
            if inner_ask_size > 0.2 and inner_bid_price < inner_ask_price:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=inner_ask_price, quantity=inner_ask_size))

        # --- Outer level: ±2 always present ---
        outer_bid = max(1, int(round(fair - 2)))
        outer_ask = min(99, int(round(fair + 2)))

        if outer_bid < outer_ask:
            outer_bid_base = 0.8 if self.bid_cooldown > 0 else 12.0
            outer_ask_base = 0.8 if self.ask_cooldown > 0 else 12.0

            outer_bid_size = max(0.2, outer_bid_base * vol_scale * bid_inv_scale)
            outer_ask_size = max(0.2, outer_ask_base * vol_scale * ask_inv_scale)

            cost = outer_bid * 0.01 * outer_bid_size
            if cost <= state.free_cash - cash_used:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=outer_bid, quantity=outer_bid_size))

            actions.append(PlaceOrder(side=Side.SELL, price_ticks=outer_ask, quantity=outer_ask_size))

        return actions
