"""Combo: arb_dodge + multi-level inner reduce + baseline outer.

Combine the best ideas:
1. Arb dodge (blackout on consecutive fills) to reduce arb losses
2. Inner ±1 only on reducing side (from inner_reduce)
3. Standard ±2/±3 outer (proven baseline)
4. Size 10, inv_cap 10 (proven baseline params)
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
        self.consecutive_buy_fills = 0
        self.consecutive_sell_fills = 0
        self.blackout = 0

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

        # Track consecutive fills
        if state.buy_filled_quantity > 0:
            self.consecutive_buy_fills += 1
            self.fill_bias -= 1.0
            self.bid_cooldown = 2
        else:
            self.consecutive_buy_fills = 0

        if state.sell_filled_quantity > 0:
            self.consecutive_sell_fills += 1
            self.fill_bias += 1.0
            self.ask_cooldown = 2
        else:
            self.consecutive_sell_fills = 0

        self.fill_bias *= 0.5

        # Blackout on arb detection
        if self.consecutive_buy_fills >= 2 or self.consecutive_sell_fills >= 2:
            self.blackout = 2

        if self.blackout > 0:
            self.blackout -= 1
            if self.bid_cooldown > 0:
                self.bid_cooldown -= 1
            if self.ask_cooldown > 0:
                self.ask_cooldown -= 1
            return actions

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * 0.5

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)
        max_inv = 10
        bid_inv_scale = max(0.0, 1.0 - net_inv / max_inv)
        ask_inv_scale = max(0.0, 1.0 + net_inv / max_inv)

        cash_remaining = state.free_cash

        # --- Inner ±1: only on inventory-reducing side, small size ---
        if self.vol_estimate < 0.8 and self.bid_cooldown == 0 and self.ask_cooldown == 0:
            inner_bid = max(1, int(round(fair - 1)))
            inner_ask = min(99, int(round(fair + 1)))

            if inner_bid < inner_ask:
                if net_inv > 2:
                    # Long: inner ask to sell
                    size = round(min(2.0, net_inv * 0.3), 1)
                    if size >= 0.5:
                        actions.append(PlaceOrder(side=Side.SELL, price_ticks=inner_ask, quantity=size))
                elif net_inv < -2:
                    # Short: inner bid to buy
                    size = round(min(2.0, abs(net_inv) * 0.3), 1)
                    if size >= 0.5:
                        cost = inner_bid * 0.01 * size
                        if cost <= cash_remaining:
                            actions.append(PlaceOrder(side=Side.BUY, price_ticks=inner_bid, quantity=size))
                            cash_remaining -= cost

        # --- Outer ±2/±3 (standard baseline) ---
        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        bid_spread = (3 if self.bid_cooldown > 0 else 2) + vol_widen
        ask_spread = (3 if self.ask_cooldown > 0 else 2) + vol_widen

        outer_bid = max(1, int(round(fair - bid_spread)))
        outer_ask = min(99, int(round(fair + ask_spread)))

        if outer_bid < outer_ask:
            bid_base = 0.8 if self.bid_cooldown > 0 else 10.0
            ask_base = 0.8 if self.ask_cooldown > 0 else 10.0

            bid_size = max(0.5, round(bid_base * vol_scale * bid_inv_scale, 1))
            ask_size = max(0.5, round(ask_base * vol_scale * ask_inv_scale, 1))

            cost = outer_bid * 0.01 * bid_size
            if cost <= cash_remaining:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=outer_bid, quantity=bid_size))

            actions.append(PlaceOrder(side=Side.SELL, price_ticks=outer_ask, quantity=ask_size))

        return actions
