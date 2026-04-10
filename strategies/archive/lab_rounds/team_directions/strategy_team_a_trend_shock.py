"""Team A: trend-following fair tilt + shock regime (from hunt_arch_best_v23 core).

Extends the v23 pipeline (per-step CancelAll, vol-scaled size, fill bias, inventory skew)
with an EWM signed-momentum tilt on fair and a distinct shock state: large moves vs
recent volatility trigger wider half-spread, extra size haircut, and brief one-sided
suppression on the side most exposed to continuation (bid after sharp drops, ask after sharp pops).
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0  # -1 down shock, +1 up shock

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
        move = 0.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_estimate = 0.9 * self.vol_estimate + 0.1 * abs(move)
            self.trend = 0.82 * self.trend + 0.18 * move

            vol_floor = 0.35
            shock_trigger = max(2.2, 3.8 * max(self.vol_estimate, vol_floor))
            if abs(move) >= shock_trigger:
                self.shock_remaining = 4
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
        self.fill_bias *= 0.4

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        trend_tilt = self.trend * 0.62
        fair = mid + self.fill_bias + inv_skew + trend_tilt

        half_spread = 2
        if self.vol_estimate > 1.0:
            half_spread += 1
        if self.shock_remaining > 0:
            half_spread += 2

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))

        if my_bid >= my_ask:
            return actions

        vol_scale = max(0.1, 1.0 - self.vol_estimate * 1.5)
        if self.shock_remaining > 0:
            vol_scale *= 0.4

        base_size = 10.0
        max_inv = 10
        bid_size = max(0.2, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.2, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0

        if self.shock_remaining > 0:
            self.shock_remaining -= 1

        cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
