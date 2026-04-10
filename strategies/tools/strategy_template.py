"""Parametric strategy family used for mean-edge optimization."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

DEFAULT_PARAMS = {
    "fill_hit": 1.0,
    "fill_decay": 0.5,
    "inventory_skew": 0.06,
    "momentum_weight": 0.5,
    "momentum_decay": 0.5,
    "vol_decay": 0.9,
    "cooldown_steps": 2,
    "spread_base": 2,
    "cooldown_extra": 1,
    "vol_widen_threshold": 1.4,
    "vol_widen_extra": 0,
    "vol_size_floor": 0.2,
    "vol_size_coeff": 0.6,
    "base_size": 1.0,
    "min_size": 0.2,
    "max_inventory": 30.0,
    "max_inventory_hard": 45.0,
    "fill_vol_spike": 0.0,
    "cash_buffer": 0.0,
}


class ParametricStrategy(BaseStrategy):
    PARAMS = DEFAULT_PARAMS

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.comp_momentum = 0.0
        self.bid_cooldown = 0
        self.ask_cooldown = 0
        self.vol_estimate = 0.0

    @staticmethod
    def _normalized_competitor_quotes(
        bid: int | None,
        ask: int | None,
    ) -> tuple[int, int]:
        if bid is None and ask is None:
            return 47, 53
        if bid is None:
            assert ask is not None
            ask_tick = int(ask)
            bid_tick = max(1, ask_tick - 6)
            if bid_tick >= ask_tick:
                ask_tick = min(99, bid_tick + 1)
            return bid_tick, ask_tick
        if ask is None:
            bid_tick = int(bid)
            ask_tick = min(99, bid_tick + 6)
            if bid_tick >= ask_tick:
                bid_tick = max(1, ask_tick - 1)
            return bid_tick, ask_tick
        return int(bid), int(ask)

    def on_step(self, state: StepState):
        params = self.PARAMS
        actions = [CancelAll()]

        comp_bid, comp_ask = self._normalized_competitor_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )
        mid = (comp_bid + comp_ask) / 2.0

        if self.prev_bid is not None and self.prev_ask is not None:
            previous_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - previous_mid
            momentum_decay = float(params["momentum_decay"])
            vol_decay = float(params["vol_decay"])
            self.comp_momentum = momentum_decay * self.comp_momentum + (1.0 - momentum_decay) * move
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)
        self.prev_bid = comp_bid
        self.prev_ask = comp_ask

        cooldown_steps = int(params["cooldown_steps"])
        fill_hit = float(params["fill_hit"])
        fill_vol_spike = float(params["fill_vol_spike"])

        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= fill_hit
            self.bid_cooldown = cooldown_steps
            self.vol_estimate = max(self.vol_estimate, fill_vol_spike)
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += fill_hit
            self.ask_cooldown = cooldown_steps
            self.vol_estimate = max(self.vol_estimate, fill_vol_spike)
        self.fill_bias *= float(params["fill_decay"])

        net_inventory = state.yes_inventory - state.no_inventory
        inventory_skew = -net_inventory * float(params["inventory_skew"])

        fair = (
            mid
            + self.fill_bias
            + inventory_skew
            + self.comp_momentum * float(params["momentum_weight"])
        )

        spread_base = int(params["spread_base"])
        cooldown_extra = int(params["cooldown_extra"])
        bid_spread = spread_base + (cooldown_extra if self.bid_cooldown > 0 else 0)
        ask_spread = spread_base + (cooldown_extra if self.ask_cooldown > 0 else 0)

        if self.vol_estimate > float(params["vol_widen_threshold"]):
            extra = int(params["vol_widen_extra"])
            bid_spread += extra
            ask_spread += extra

        if self.bid_cooldown > 0:
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            self.ask_cooldown -= 1

        my_bid = max(1, int(round(fair - bid_spread)))
        my_ask = min(99, int(round(fair + ask_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(
            float(params["vol_size_floor"]),
            1.0 - self.vol_estimate * float(params["vol_size_coeff"]),
        )
        base_size = float(params["base_size"]) * vol_scale
        min_size = float(params["min_size"])
        max_inventory = max(float(params["max_inventory"]), 1e-9)
        max_inventory_hard = float(params["max_inventory_hard"])

        bid_size = max(min_size, base_size * max(0.0, 1.0 - net_inventory / max_inventory))
        ask_size = max(min_size, base_size * max(0.0, 1.0 + net_inventory / max_inventory))

        if net_inventory >= max_inventory_hard:
            bid_size = 0.0
        if net_inventory <= -max_inventory_hard:
            ask_size = 0.0

        available_cash = max(0.0, state.cash - float(params["cash_buffer"]))
        if bid_size > 0.0:
            buy_collateral = my_bid * 0.01 * bid_size
            if buy_collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
                available_cash -= buy_collateral

        if ask_size > 0.0:
            ask_price = my_ask * 0.01
            covered_size = min(max(0.0, state.yes_inventory), ask_size)
            uncovered_size = ask_size - covered_size
            ask_collateral = max(0.0, 1.0 - ask_price) * uncovered_size
            if ask_collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions


class Strategy(ParametricStrategy):
    PARAMS = dict(DEFAULT_PARAMS)
