"""Inner-reduce: ±1 only on inventory-reducing side, ±2 on both sides.

Key insight: The inner ±1 level generates both retail and arb fills.
But if we only place it on the reducing side, we:
1. Get retail fills that reduce inventory (good)
2. Get arb fills that reduce inventory (less bad - we wanted to reduce anyway)
3. Never accumulate more inventory at tight spread

Combined with aggressive outer ±2 for normal market-making.
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

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)
        max_inv = 10
        bid_inv_scale = max(0.0, 1.0 - net_inv / max_inv)
        ask_inv_scale = max(0.0, 1.0 + net_inv / max_inv)

        cash_remaining = state.free_cash

        # --- Inner ±1 level: ONLY on inventory-reducing side ---
        if self.vol_estimate < 1.0:
            inner_bid = max(1, int(round(fair - 1)))
            inner_ask = min(99, int(round(fair + 1)))

            if inner_bid < inner_ask:
                # Long YES (net_inv > 2): place inner ASK to sell
                if net_inv > 2 and self.ask_cooldown == 0:
                    inner_ask_size = round(min(3.0, net_inv * 0.5) * vol_scale, 1)
                    if inner_ask_size >= 0.5:
                        actions.append(PlaceOrder(side=Side.SELL, price_ticks=inner_ask, quantity=inner_ask_size))

                # Short YES (net_inv < -2): place inner BID to buy
                if net_inv < -2 and self.bid_cooldown == 0:
                    inner_bid_size = round(min(3.0, abs(net_inv) * 0.5) * vol_scale, 1)
                    if inner_bid_size >= 0.5:
                        cost = inner_bid * 0.01 * inner_bid_size
                        if cost <= cash_remaining:
                            actions.append(PlaceOrder(side=Side.BUY, price_ticks=inner_bid, quantity=inner_bid_size))
                            cash_remaining -= cost

        # --- Outer ±2 level (±3 on cooldown) ---
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
