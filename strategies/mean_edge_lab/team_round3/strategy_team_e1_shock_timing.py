"""Champion trend-shock with multi-tier triggers, hysteresis, asymmetric duration, suppression schedule."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    PARAMS = {
        "base_size": 14.113052,
        "fill_decay": 0.590229,
        "fill_hit": 0.497805,
        "inv_skew": 0.027657,
        "max_inventory": 8.674922,
        "min_size": 0.550072,
        "shock_duration_mild": 2,
        "shock_duration_severe": 4,
        "shock_mild_trigger_frac": 0.76,
        "shock_severe_trigger_frac": 1.0,
        "shock_release_frac": 0.4,
        "shock_dur_trend_align": 0.14,
        "shock_spread_extra_mild": 1,
        "shock_spread_extra_severe": 2,
        "shock_size_mult_mild": 0.2,
        "shock_size_mult_severe": 0.15,
        "shock_full_suppress_steps": 2,
        "shock_relax_suppress_mult": 0.32,
        "shock_upgrade_extra_steps": 2,
        "shock_trigger_min": 1.846626,
        "shock_trigger_vol_mult": 3.044949,
        "shock_vol_floor": 0.354703,
        "spread_base": 2,
        "spread_vol_extra": 1,
        "spread_vol_threshold": 1.227116,
        "trend_alpha": 0.173101,
        "trend_decay": 0.654128,
        "trend_weight": 0.996645,
        "vol_coeff": 2.402131,
        "vol_decay": 0.935142,
        "vol_floor": 0.064398,
    }

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.shock_tier = 0
        self.shock_start_duration = 0
        self.shock_armed = True

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
        base_trigger = 0.0
        mild_th = 0.0
        severe_th = 0.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid

            vol_decay = float(p["vol_decay"])
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)

            trend_decay = float(p["trend_decay"])
            trend_alpha = float(p["trend_alpha"])
            self.trend = trend_decay * self.trend + trend_alpha * move

            base_trigger = max(
                float(p["shock_trigger_min"]),
                float(p["shock_trigger_vol_mult"]) * max(self.vol_estimate, float(p["shock_vol_floor"])),
            )
            mild_th = base_trigger * float(p["shock_mild_trigger_frac"])
            severe_th = base_trigger * float(p["shock_severe_trigger_frac"])
            release_th = base_trigger * float(p["shock_release_frac"])

            if self.shock_remaining == 0:
                if not self.shock_armed and abs(move) < release_th:
                    self.shock_armed = True
                elif self.shock_armed:
                    tier = 0
                    if abs(move) >= severe_th:
                        tier = 2
                        dur = int(p["shock_duration_severe"])
                    elif abs(move) >= mild_th:
                        tier = 1
                        dur = int(p["shock_duration_mild"])
                    else:
                        dur = 0

                    if tier > 0:
                        asym = float(p["shock_dur_trend_align"])
                        if asym != 0.0 and abs(self.trend) > 1e-9:
                            aligned = 1.0 if (move * self.trend) > 0.0 else -1.0
                            dur = int(round(dur * (1.0 + asym * aligned)))
                        dur = max(1, dur)
                        self.shock_remaining = dur
                        self.shock_start_duration = dur
                        self.shock_tier = tier
                        self.shock_sign = 1 if move > 0.0 else -1
                        self.shock_armed = False

            elif self.shock_tier == 1 and abs(move) >= severe_th:
                self.shock_tier = 2
                bump = int(p["shock_upgrade_extra_steps"])
                self.shock_remaining += bump
                self.shock_start_duration += bump

        self.prev_bid = bid
        self.prev_ask = ask

        fill_hit = float(p["fill_hit"])
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
        self.fill_bias *= float(p["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        fair = (
            mid
            + self.fill_bias
            - net_inv * float(p["inv_skew"])
            + self.trend * float(p["trend_weight"])
        )

        half_spread = int(p["spread_base"])
        if self.vol_estimate > float(p["spread_vol_threshold"]):
            half_spread += int(p["spread_vol_extra"])
        if self.shock_remaining > 0:
            if self.shock_tier >= 2:
                half_spread += int(p["shock_spread_extra_severe"])
            else:
                half_spread += int(p["shock_spread_extra_mild"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(p["vol_floor"]), 1.0 - self.vol_estimate * float(p["vol_coeff"]))
        if self.shock_remaining > 0:
            if self.shock_tier >= 2:
                vol_scale *= float(p["shock_size_mult_severe"])
            else:
                vol_scale *= float(p["shock_size_mult_mild"])

        base_size = float(p["base_size"])
        max_inv = max(float(p["max_inventory"]), 1e-9)
        min_size = float(p["min_size"])
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if self.shock_remaining > 0:
            shock_age = self.shock_start_duration - self.shock_remaining
            full_sup = int(p["shock_full_suppress_steps"])
            relax = float(p["shock_relax_suppress_mult"])
            if self.shock_sign < 0:
                if shock_age < full_sup:
                    bid_size = 0.0
                else:
                    bid_size *= relax
            elif self.shock_sign > 0:
                if shock_age < full_sup:
                    ask_size = 0.0
                else:
                    ask_size *= relax
            self.shock_remaining -= 1
            if self.shock_remaining == 0:
                self.shock_tier = 0
                self.shock_sign = 0
                self.shock_start_duration = 0

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions
