"""Parametric template for trend-following + shock-protection architecture."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

DEFAULT_PARAMS = {
    "fill_hit": 1.0,
    "fill_decay": 0.4,
    "inv_skew": 0.06,
    "vol_decay": 0.9,
    "trend_decay": 0.82,
    "trend_alpha": 0.18,
    "trend_weight": 0.62,
    "shock_vol_floor": 0.35,
    "shock_trigger_min": 2.2,
    "shock_trigger_vol_mult": 3.8,
    "shock_duration": 4,
    "spread_base": 2,
    "spread_vol_threshold": 1.0,
    "spread_vol_extra": 1,
    "spread_shock_extra": 2,
    "vol_floor": 0.1,
    "vol_coeff": 1.5,
    "shock_size_mult": 0.4,
    "base_size": 10.0,
    "min_size": 0.2,
    "max_inventory": 10.0,
}


class ParametricTrendShockStrategy(BaseStrategy):
    PARAMS = DEFAULT_PARAMS

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0

    def on_step(self, state: StepState):
        params = self.PARAMS
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

            vol_decay = float(params["vol_decay"])
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)

            trend_decay = float(params["trend_decay"])
            trend_alpha = float(params["trend_alpha"])
            self.trend = trend_decay * self.trend + trend_alpha * move

            shock_trigger = max(
                float(params["shock_trigger_min"]),
                float(params["shock_trigger_vol_mult"])
                * max(self.vol_estimate, float(params["shock_vol_floor"])),
            )
            if abs(move) >= shock_trigger:
                self.shock_remaining = int(params["shock_duration"])
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        fill_hit = float(params["fill_hit"])
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
        self.fill_bias *= float(params["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        fair = (
            mid
            + self.fill_bias
            - net_inv * float(params["inv_skew"])
            + self.trend * float(params["trend_weight"])
        )

        half_spread = int(params["spread_base"])
        if self.vol_estimate > float(params["spread_vol_threshold"]):
            half_spread += int(params["spread_vol_extra"])
        if self.shock_remaining > 0:
            half_spread += int(params["spread_shock_extra"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(params["vol_floor"]), 1.0 - self.vol_estimate * float(params["vol_coeff"]))
        if self.shock_remaining > 0:
            vol_scale *= float(params["shock_size_mult"])

        base_size = float(params["base_size"])
        max_inv = max(float(params["max_inventory"]), 1e-9)
        min_size = float(params["min_size"])
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions


class Strategy(ParametricTrendShockStrategy):
    PARAMS = dict(DEFAULT_PARAMS)
