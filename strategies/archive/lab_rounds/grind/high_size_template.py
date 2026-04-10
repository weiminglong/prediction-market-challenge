"""Parametric high-size market making template."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

DEFAULT_PARAMS = {
    "fill_hit": 1.0,
    "fill_decay": 0.5,
    "inventory_skew": 0.06,
    "momentum_weight": 0.0,
    "momentum_decay": 0.5,
    "vol_decay": 0.9,
    "cooldown_steps": 0,
    "spread_base": 2,
    "cooldown_extra": 0,
    "vol_widen_threshold": 1.0,
    "vol_widen_extra": 1,
    "vol_floor": 0.2,
    "vol_coeff": 0.7,
    "base_size": 10.0,
    "min_size": 0.2,
    "max_inventory": 10.0,
    "max_inventory_hard": 16.0,
}


def _normalize_quotes(
    best_bid: int | None,
    best_ask: int | None,
) -> tuple[int, int]:
    if best_bid is None and best_ask is None:
        return 47, 53
    if best_bid is None:
        ask = int(best_ask)
        bid = max(1, ask - 6)
        if bid >= ask:
            ask = min(99, bid + 1)
        return bid, ask
    if best_ask is None:
        bid = int(best_bid)
        ask = min(99, bid + 6)
        if bid >= ask:
            bid = max(1, ask - 1)
        return bid, ask
    return int(best_bid), int(best_ask)


class ParametricHighSizeStrategy(BaseStrategy):
    PARAMS = DEFAULT_PARAMS

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.momentum = 0.0
        self.vol_estimate = 0.0
        self.bid_cooldown = 0
        self.ask_cooldown = 0

    def on_step(self, state: StepState):
        params = self.PARAMS
        actions = [CancelAll()]

        bid, ask = _normalize_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )
        mid = (bid + ask) / 2.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            momentum_decay = float(params["momentum_decay"])
            vol_decay = float(params["vol_decay"])
            self.momentum = momentum_decay * self.momentum + (1.0 - momentum_decay) * move
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)
        self.prev_bid = bid
        self.prev_ask = ask

        fill_hit = float(params["fill_hit"])
        cooldown_steps = int(params["cooldown_steps"])
        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= fill_hit
            self.bid_cooldown = cooldown_steps
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += fill_hit
            self.ask_cooldown = cooldown_steps
        self.fill_bias *= float(params["fill_decay"])

        net_inventory = state.yes_inventory - state.no_inventory
        fair = (
            mid
            + self.fill_bias
            - net_inventory * float(params["inventory_skew"])
            + self.momentum * float(params["momentum_weight"])
        )

        bid_spread = int(params["spread_base"]) + (int(params["cooldown_extra"]) if self.bid_cooldown > 0 else 0)
        ask_spread = int(params["spread_base"]) + (int(params["cooldown_extra"]) if self.ask_cooldown > 0 else 0)

        if self.vol_estimate >= float(params["vol_widen_threshold"]):
            bid_spread += int(params["vol_widen_extra"])
            ask_spread += int(params["vol_widen_extra"])

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        my_bid = max(1, int(round(fair - bid_spread)))
        my_ask = min(99, int(round(fair + ask_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(
            float(params["vol_floor"]),
            1.0 - self.vol_estimate * float(params["vol_coeff"]),
        )
        base_size = float(params["base_size"]) * vol_scale
        max_inventory = max(float(params["max_inventory"]), 1e-9)
        min_size = float(params["min_size"])
        bid_size = max(min_size, base_size * max(0.0, 1.0 - net_inventory / max_inventory))
        ask_size = max(min_size, base_size * max(0.0, 1.0 + net_inventory / max_inventory))

        hard = float(params["max_inventory_hard"])
        if net_inventory >= hard:
            bid_size = 0.0
        if net_inventory <= -hard:
            ask_size = 0.0

        available_cash = state.free_cash
        if bid_size > 0.0:
            bid_collateral = my_bid * 0.01 * bid_size
            if bid_collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
                available_cash -= bid_collateral

        if ask_size > 0.0:
            ask_price = my_ask * 0.01
            covered = min(max(0.0, state.yes_inventory), ask_size)
            uncovered = ask_size - covered
            ask_collateral = max(0.0, 1.0 - ask_price) * uncovered
            if ask_collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions


class Strategy(ParametricHighSizeStrategy):
    PARAMS = dict(DEFAULT_PARAMS)
