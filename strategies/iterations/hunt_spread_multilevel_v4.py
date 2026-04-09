"""Multi-level v4: Three levels ±1/±2/±3 with graduated sizes.

Hypothesis: More levels = more price discovery. Tiny at ±1, medium at ±2,
large at ±3. The ±3 is very safe from arb, while ±1 grabs aggressive retail.
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

        vol_scale = max(0.3, 1.0 - self.vol_estimate * 0.5)
        max_inv = 10

        bid_inv_scale = max(0.0, 1.0 - net_inv / max_inv)
        ask_inv_scale = max(0.0, 1.0 + net_inv / max_inv)

        cash_remaining = state.free_cash

        # Define levels: (spread, bid_size, ask_size, skip_on_cooldown, skip_on_vol)
        levels = [
            (1, 1.0, 1.0, True, True),    # ±1: tiny, skip on cooldown/vol
            (2, 6.0, 6.0, False, False),   # ±2: medium, always
            (3, 4.0, 4.0, False, False),   # ±3: moderate, always (safe from arb)
        ]

        for spread, bid_base, ask_base, skip_cooldown, skip_vol in levels:
            if skip_cooldown and (self.bid_cooldown > 0 or self.ask_cooldown > 0):
                continue
            if skip_vol and self.vol_estimate > 0.8:
                continue

            level_bid = max(1, int(round(fair - spread)))
            level_ask = min(99, int(round(fair + spread)))

            if level_bid >= level_ask:
                continue

            # On cooldown, reduce size for ±2 level
            if self.bid_cooldown > 0 and spread == 2:
                bid_base = 0.8
            if self.ask_cooldown > 0 and spread == 2:
                ask_base = 0.8

            b_size = round(bid_base * vol_scale * bid_inv_scale, 1)
            a_size = round(ask_base * vol_scale * ask_inv_scale, 1)

            if b_size >= 0.5:
                cost = level_bid * 0.01 * b_size
                if cost <= cash_remaining:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=level_bid, quantity=b_size))
                    cash_remaining -= cost

            if a_size >= 0.5:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=level_ask, quantity=a_size))

        return actions
