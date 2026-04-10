"""E3: champion trend-shock core + asymmetric dual-horizon trend quotes.

- Slow + fast EMAs on mid moves; fair uses weighted sum (stability + responsiveness).
- Trend sign widens the "fade" side and tightens the "follow" side (ticks).
- On sign flip (nonzero to opposite), temporary symmetric spread widening (pullback).
- Shock path unchanged: vol-based spread, one-sided size cut during shock window.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    PARAMS = {
        "asym_spread_ticks": 1,
        "base_size": 14.113052,
        "fill_decay": 0.590229,
        "fill_hit": 0.497805,
        "inv_skew": 0.027657,
        "max_inventory": 8.674922,
        "min_size": 0.550072,
        "pullback_duration": 2,
        "pullback_spread_extra": 1,
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
        "trend_fast_alpha": 0.42,
        "trend_fast_decay": 0.48,
        "trend_sign_eps": 0.08,
        "trend_weight_fast": 0.38,
        "trend_weight_slow": 0.62,
        "vol_coeff": 2.402131,
        "vol_decay": 0.935142,
        "vol_floor": 0.064398,
    }

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend_slow = 0.0
        self.trend_fast = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.prev_trend_sign = 0
        self.pullback_remaining = 0

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
            self.trend_slow = trend_decay * self.trend_slow + trend_alpha * move

            tfd = float(p["trend_fast_decay"])
            tfa = float(p["trend_fast_alpha"])
            self.trend_fast = tfd * self.trend_fast + tfa * move

            shock_trigger = max(
                float(p["shock_trigger_min"]),
                float(p["shock_trigger_vol_mult"]) * max(self.vol_estimate, float(p["shock_vol_floor"])),
            )
            if abs(move) >= shock_trigger:
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

        tws = float(p["trend_weight_slow"])
        twf = float(p["trend_weight_fast"])
        trend_blend = tws * self.trend_slow + twf * self.trend_fast

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias - net_inv * float(p["inv_skew"]) + trend_blend

        eps = float(p["trend_sign_eps"])
        sign = 0
        if trend_blend > eps:
            sign = 1
        elif trend_blend < -eps:
            sign = -1

        if self.prev_trend_sign != 0 and sign != 0 and sign != self.prev_trend_sign:
            self.pullback_remaining = int(p["pullback_duration"])
        self.prev_trend_sign = sign

        half = int(p["spread_base"])
        if self.vol_estimate > float(p["spread_vol_threshold"]):
            half += int(p["spread_vol_extra"])
        if self.shock_remaining > 0:
            half += int(p["spread_shock_extra"])

        asym = int(p["asym_spread_ticks"])
        bid_half = half
        ask_half = half
        if sign > 0:
            bid_half = max(1, bid_half - asym)
            ask_half = ask_half + asym
        elif sign < 0:
            bid_half = bid_half + asym
            ask_half = max(1, ask_half - asym)

        if self.pullback_remaining > 0:
            pb = int(p["pullback_spread_extra"])
            bid_half += pb
            ask_half += pb
            self.pullback_remaining -= 1

        my_bid = max(1, int(round(fair - bid_half)))
        my_ask = min(99, int(round(fair + ask_half)))
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
