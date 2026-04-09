"""Team E4: event-regime state machine on top of local-trend champion."""

from __future__ import annotations

from enum import IntEnum

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Regime(IntEnum):
    CALM = 0
    TRANSITION = 1
    SHOCK = 2
    RECOVERY = 3


class Strategy(BaseStrategy):
    PARAMS = {
        "base_size": 14.113052,
        "fill_decay": 0.590229,
        "fill_hit": 0.497805,
        "inv_skew": 0.027657,
        "max_inventory": 8.674922,
        "min_size": 0.550072,
        "shock_duration": 4,
        "shock_size_mult": 0.15,
        "shock_trigger_min": 1.846626,
        "shock_trigger_vol_mult": 3.044949,
        "shock_vol_floor": 0.354703,
        "spread_base": 2,
        "spread_shock_extra": 2,
        "spread_vol_extra": 1,
        "spread_vol_threshold": 1.227116,
        "trend_alpha": 0.173101,
        "trend_decay": 0.654128,
        "trend_weight": 0.996645,
        "vol_coeff": 2.402131,
        "vol_decay": 0.935142,
        "vol_floor": 0.064398,
        # Regime / anti-whipsaw
        "transition_vol_mult": 1.55,
        "transition_move_frac": 0.62,
        "transition_spread_add": 1,
        "transition_size_mult": 0.78,
        "transition_calm_steps": 3,
        "shock_arm_required": 2,
        "shock_single_bar_mult": 2.35,
        "recovery_duration": 3,
        "recovery_spread_add": 1,
        "recovery_size_mult": 0.72,
        "fill_mem_decay": 0.82,
        "fill_mem_skew": 0.018,
    }

    def __init__(self):
        self.regime = Regime.CALM
        self.fill_bias = 0.0
        self.fill_buy_mem = 0.0
        self.fill_sell_mem = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.recovery_remaining = 0
        self.transition_calm = 0
        self.shock_arm_sign = 0
        self.shock_arm_steps = 0

    def on_step(self, state: StepState):
        p = self.PARAMS
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

            vol_decay = float(p["vol_decay"])
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)

            trend_decay = float(p["trend_decay"])
            trend_alpha = float(p["trend_alpha"])
            self.trend = trend_decay * self.trend + trend_alpha * move

        self.prev_bid = bid
        self.prev_ask = ask

        shock_trigger = max(
            float(p["shock_trigger_min"]),
            float(p["shock_trigger_vol_mult"]) * max(self.vol_estimate, float(p["shock_vol_floor"])),
        )
        transition_vol = float(p["transition_vol_mult"]) * max(
            self.vol_estimate, float(p["shock_vol_floor"])
        )
        transition_move = float(p["transition_move_frac"]) * shock_trigger
        single_bar = float(p["shock_single_bar_mult"]) * shock_trigger

        abs_move = abs(move)
        move_sign = 1 if move > 0 else (-1 if move < 0 else 0)

        if abs_move >= shock_trigger and move_sign != 0:
            if move_sign == self.shock_arm_sign:
                self.shock_arm_steps += 1
            else:
                self.shock_arm_sign = move_sign
                self.shock_arm_steps = 1
        elif abs_move < 0.45 * transition_move:
            self.shock_arm_steps = max(0, self.shock_arm_steps - 1)
            if self.shock_arm_steps == 0:
                self.shock_arm_sign = 0

        shock_commit = (self.shock_arm_steps >= int(p["shock_arm_required"])) or (
            abs_move >= single_bar
        )

        calm_step = abs_move < 0.35 * transition_move and self.vol_estimate < 0.85 * transition_vol

        if self.regime == Regime.CALM:
            if self.shock_remaining > 0:
                self.regime = Regime.SHOCK
            elif shock_commit and abs_move >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = move_sign
                self.regime = Regime.SHOCK
                self.transition_calm = 0
                self.shock_arm_steps = 0
                self.shock_arm_sign = 0
            elif abs_move >= transition_move or self.vol_estimate >= transition_vol:
                self.regime = Regime.TRANSITION
                self.transition_calm = 0

        elif self.regime == Regime.TRANSITION:
            if shock_commit and abs_move >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = move_sign if move_sign != 0 else self.shock_arm_sign
                self.regime = Regime.SHOCK
                self.transition_calm = 0
                self.shock_arm_steps = 0
                self.shock_arm_sign = 0
            elif calm_step:
                self.transition_calm += 1
                if self.transition_calm >= int(p["transition_calm_steps"]):
                    self.regime = Regime.CALM
                    self.transition_calm = 0
            else:
                self.transition_calm = 0

        elif self.regime == Regime.SHOCK:
            if self.shock_remaining <= 0:
                self.recovery_remaining = int(p["recovery_duration"])
                self.regime = Regime.RECOVERY
                self.shock_arm_steps = 0
                self.shock_arm_sign = 0

        elif self.regime == Regime.RECOVERY:
            if self.recovery_remaining <= 0:
                self.regime = Regime.CALM
            elif shock_commit and abs_move >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = move_sign if move_sign != 0 else 1
                self.recovery_remaining = 0
                self.regime = Regime.SHOCK
                self.shock_arm_steps = 0
                self.shock_arm_sign = 0

        fill_hit = float(p["fill_hit"])
        mem_decay = float(p["fill_mem_decay"])
        self.fill_buy_mem *= mem_decay
        self.fill_sell_mem *= mem_decay
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
            self.fill_buy_mem += float(state.buy_filled_quantity)
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
            self.fill_sell_mem += float(state.sell_filled_quantity)
        self.fill_bias *= float(p["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        fill_mem_signal = float(p["fill_mem_skew"]) * (self.fill_sell_mem - self.fill_buy_mem)
        fair = (
            mid
            + self.fill_bias
            + fill_mem_signal
            - net_inv * float(p["inv_skew"])
            + self.trend * float(p["trend_weight"])
        )

        half_spread = int(p["spread_base"])
        if self.vol_estimate > float(p["spread_vol_threshold"]):
            half_spread += int(p["spread_vol_extra"])
        if self.regime == Regime.TRANSITION:
            half_spread += int(p["transition_spread_add"])
        if self.regime == Regime.SHOCK or self.shock_remaining > 0:
            half_spread += int(p["spread_shock_extra"])
        if self.regime == Regime.RECOVERY:
            half_spread += int(p["recovery_spread_add"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(p["vol_floor"]), 1.0 - self.vol_estimate * float(p["vol_coeff"]))
        if self.regime == Regime.TRANSITION:
            vol_scale *= float(p["transition_size_mult"])
        if self.regime == Regime.SHOCK or self.shock_remaining > 0:
            vol_scale *= float(p["shock_size_mult"])
        if self.regime == Regime.RECOVERY:
            vol_scale *= float(p["recovery_size_mult"])

        base_size = float(p["base_size"])
        max_inv = max(float(p["max_inventory"]), 1e-9)
        min_size = float(p["min_size"])
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if self.regime == Regime.SHOCK or self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        if self.regime == Regime.RECOVERY and self.recovery_remaining > 0:
            self.recovery_remaining -= 1

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions
