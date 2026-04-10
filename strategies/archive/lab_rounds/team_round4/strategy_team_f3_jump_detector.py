"""F3: gated move/vol ratio jumps + fill-asym memory skew; mild immediate post-jump tier."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    PARAMS = {
        "asym_fair_cap": 2.2,
        "asym_fair_scale": 0.0115,
        "asym_hit": 0.5,
        "asym_mem_decay": 0.84,
        "base_size": 14.113052,
        "fill_decay": 0.590229,
        "fill_hit": 0.497805,
        "imm_shock_size_mult": 0.9,
        "imm_shock_spread_extra": 1,
        "inv_skew": 0.027657,
        "jump_ratio_abs_min": 1.15,
        "jump_ratio_threshold": 2.78,
        "max_inventory": 8.674922,
        "min_size": 0.550072,
        "min_vol_for_ratio": 0.135,
        "ratio_vol_floor": 0.27,
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
        "trend_weight": 0.99735,
        "vol_coeff": 2.402131,
        "vol_decay": 0.935142,
        "vol_floor": 0.064398,
    }

    def __init__(self):
        self.fill_bias = 0.0
        self.fill_asym_mem = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0

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
        vol_before = self.vol_estimate

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid

            vol_decay = float(p["vol_decay"])
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)

            trend_decay = float(p["trend_decay"])
            trend_alpha = float(p["trend_alpha"])
            self.trend = trend_decay * self.trend + trend_alpha * move

            denom_r = max(vol_before, float(p["ratio_vol_floor"]))
            ratio_ok = vol_before >= float(p["min_vol_for_ratio"])
            ratio_jump = (
                ratio_ok
                and abs(move) >= float(p["jump_ratio_abs_min"])
                and abs(move) / denom_r >= float(p["jump_ratio_threshold"])
            )

            shock_trigger = max(
                float(p["shock_trigger_min"]),
                float(p["shock_trigger_vol_mult"])
                * max(self.vol_estimate, float(p["shock_vol_floor"])),
            )
            classic_jump = abs(move) >= shock_trigger

            if classic_jump or ratio_jump:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        fill_hit = float(p["fill_hit"])
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
        self.fill_bias *= float(p["fill_decay"])

        asym_hit = float(p["asym_hit"])
        asym_d = state.buy_filled_quantity - state.sell_filled_quantity
        if asym_d != 0.0:
            self.fill_asym_mem += asym_hit * asym_d
        self.fill_asym_mem *= float(p["asym_mem_decay"])

        cap = float(p["asym_fair_cap"])
        mem_c = max(-cap, min(cap, self.fill_asym_mem))
        asym_fair = float(p["asym_fair_scale"]) * mem_c

        net_inv = state.yes_inventory - state.no_inventory
        fair = (
            mid
            + self.fill_bias
            - net_inv * float(p["inv_skew"])
            + self.trend * float(p["trend_weight"])
            + asym_fair
        )

        half_spread = int(p["spread_base"])
        if self.vol_estimate > float(p["spread_vol_threshold"]):
            half_spread += int(p["spread_vol_extra"])
        dur = int(p["shock_duration"])
        imm = self.shock_remaining == dur
        if self.shock_remaining > 0:
            half_spread += int(p["spread_shock_extra"])
            if imm:
                half_spread += int(p["imm_shock_spread_extra"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(p["vol_floor"]), 1.0 - self.vol_estimate * float(p["vol_coeff"]))
        if self.shock_remaining > 0:
            vol_scale *= float(p["shock_size_mult"])
            if imm:
                vol_scale *= float(p["imm_shock_size_mult"])

        base_size = float(p["base_size"])
        max_inv = max(float(p["max_inventory"]), 1e-9)
        min_size = float(p["min_size"])
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
