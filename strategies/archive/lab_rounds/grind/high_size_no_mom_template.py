"""Exact no-cooldown/no-momentum template from the current best family."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

DEFAULT_PARAMS = {
    "fill_hit": 1.0,
    "fill_decay": 0.5,
    "inventory_skew": 0.06,
    "vol_decay": 0.9,
    "spread_base": 2,
    "vol_widen_threshold": 1.0,
    "vol_widen_extra": 1,
    "vol_floor": 0.2,
    "vol_coeff": 0.7,
    "base_size": 10.0,
    "min_size": 0.2,
    "max_inventory": 10.0,
}


class ParametricNoCooldownNoMomentumStrategy(BaseStrategy):
    PARAMS = DEFAULT_PARAMS

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_estimate = 0.0

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

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            vol_decay = float(params["vol_decay"])
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)
        self.prev_bid = bid
        self.prev_ask = ask

        fill_hit = float(params["fill_hit"])
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
        self.fill_bias *= float(params["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * float(params["inventory_skew"])
        fair = mid + self.fill_bias + inv_skew

        vol_widen = int(params["vol_widen_extra"]) if self.vol_estimate > float(params["vol_widen_threshold"]) else 0
        spread_base = int(params["spread_base"])
        my_bid = max(1, int(round(fair - spread_base - vol_widen)))
        my_ask = min(99, int(round(fair + spread_base + vol_widen)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(params["vol_floor"]), 1.0 - self.vol_estimate * float(params["vol_coeff"]))
        base_size = float(params["base_size"])
        max_inv = max(float(params["max_inventory"]), 1e-9)
        min_size = float(params["min_size"])
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions


class Strategy(ParametricNoCooldownNoMomentumStrategy):
    PARAMS = dict(DEFAULT_PARAMS)
