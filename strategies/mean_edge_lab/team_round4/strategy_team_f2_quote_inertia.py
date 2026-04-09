"""Quote inertia: smooth fair-offset from mid (fill/inv/trend); mid tracks instantly."""

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
        # Inertia on offset = fair - mid (tick units); shock/mega loosen smoothing.
        "off_inertia_calm": 0.825,
        "off_inertia_shock": 0.48,
        "off_drift_cap_calm": 0.95,
        "off_drift_cap_shock": 7.5,
        "mega_move_mult": 2.25,
    }

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.anchor_offset: float | None = None
        self._mega_shock_steps = 0

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

            shock_trigger = max(
                float(p["shock_trigger_min"]),
                float(p["shock_trigger_vol_mult"]) * max(self.vol_estimate, float(p["shock_vol_floor"])),
            )
            mega_thr = shock_trigger * float(p["mega_move_mult"])
            if abs(move) >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = 1 if move > 0 else -1
                if abs(move) >= mega_thr:
                    self._mega_shock_steps = int(p["shock_duration"])
            elif self._mega_shock_steps > 0:
                self._mega_shock_steps -= 1

        self.prev_bid = bid
        self.prev_ask = ask

        fill_hit = float(p["fill_hit"])
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
        self.fill_bias *= float(p["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        raw_offset = (
            self.fill_bias
            - net_inv * float(p["inv_skew"])
            + self.trend * float(p["trend_weight"])
        )

        in_shock = self.shock_remaining > 0
        mega = self._mega_shock_steps > 0
        if self.anchor_offset is None:
            self.anchor_offset = raw_offset
        else:
            if in_shock or mega:
                inertia = float(p["off_inertia_shock"])
                cap_o = float(p["off_drift_cap_shock"])
                if mega:
                    inertia = min(inertia, 0.32)
                    cap_o = max(cap_o, float(p["off_drift_cap_shock"]) + 1.5)
            else:
                inertia = float(p["off_inertia_calm"])
                cap_o = float(p["off_drift_cap_calm"])

            blended = inertia * self.anchor_offset + (1.0 - inertia) * raw_offset
            delta = blended - self.anchor_offset
            if delta > cap_o:
                blended = self.anchor_offset + cap_o
            elif delta < -cap_o:
                blended = self.anchor_offset - cap_o
            self.anchor_offset = blended

        fair = mid + self.anchor_offset

        half_spread = int(p["spread_base"])
        if self.vol_estimate > float(p["spread_vol_threshold"]):
            half_spread += int(p["spread_vol_extra"])
        if self.shock_remaining > 0:
            half_spread += int(p["spread_shock_extra"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(p["vol_floor"]), 1.0 - self.vol_estimate * float(p["vol_coeff"]))
        if self.shock_remaining > 0:
            vol_scale *= float(p["shock_size_mult"])

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
