"""Dual fair fusion: stable (slow) fair + reactive (directional) fair, confidence-weighted blend."""

from __future__ import annotations

import math

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _norm_ppf_approx(p: float) -> float:
    if p <= 0.001:
        return -3.09
    if p >= 0.999:
        return 3.09
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t: float) -> float:
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _tick_to_price(price_ticks: int) -> float:
    return price_ticks * 0.01


class Strategy(BaseStrategy):
    PARAMS = {
        "base_size": 13.2,
        "fill_decay": 0.59,
        "fill_hit": 0.50,
        "inv_skew": 0.028,
        "inv_soft": 4.6,
        "inv_edge_mult": 0.016,
        "max_inventory": 8.7,
        "min_size": 0.55,
        "side_damp_soft": 5.5,
        "damp_bid_when_long": 0.94,
        "damp_ask_when_short": 0.94,
        "uncovered_penalty": 0.12,
        "shock_duration": 4,
        "shock_size_mult": 0.15,
        "shock_trigger_min": 1.85,
        "shock_trigger_vol_mult": 3.0,
        "shock_vol_floor": 0.35,
        "spread_base": 2,
        "spread_shock_extra": 2,
        "spread_vol_extra": 1,
        "spread_vol_threshold": 1.23,
        "spread_disagree_scale": 2.8,
        "spread_disagree_coef": 0.42,
        "vol_decay": 0.935,
        "vol_coeff": 2.35,
        "vol_floor": 0.065,
        "reserve_cash_base": 0.0,
        "reserve_cash_vol": 5.0,
        # Slow vs fast fair
        "trend_slow_decay": 0.88,
        "trend_slow_alpha": 0.055,
        "trend_fast_decay": 0.65,
        "trend_fast_alpha": 0.17,
        "trend_slow_weight": 1.0,
        "trend_fast_weight": 1.0,
        "stable_fill_mult": 0.86,
        "reactive_dir_weight": 0.52,
        "dir_decay": 0.5,
        "dir_drift_weight": 0.38,
        "dir_fill_weight": 0.28,
        "dir_clamp": 1.45,
        # Confidence blending
        "conf_vol_k": 2.15,
        "conf_disagree_decay": 3.4,
        "conf_reactive_trend_cap": 1.85,
        "conf_reactive_dir_cap": 1.35,
        "conf_reactive_base": 0.22,
        "conf_stable_floor": 0.18,
        "flow_align_bonus": 0.14,
        "flow_misalign_penalty": 0.11,
        "size_uncertainty_k": 0.19,
        "dir_safe_mult": 0.95,
        "dir_risky_mult": 0.78,
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
        self.arb_direction = 0.0

    def _convex_inv_shift(self, net_inv: float) -> float:
        p = self.PARAMS
        if net_inv == 0.0:
            return 0.0
        sig = 1.0 if net_inv > 0 else -1.0
        max_inv = max(p["max_inventory"], 1e-6)
        a = min(abs(net_inv), max_inv * 1.4)
        skew = p["inv_skew"]
        soft = max(p["inv_soft"], 1e-6)
        em = p["inv_edge_mult"]
        if a <= soft:
            t = a / soft
            w = t * t * (3.0 - 2.0 * t)
            return -sig * skew * a * w
        excess = a - soft
        base = skew * soft
        tail = skew * excess * (1.0 + em * excess * excess)
        return -sig * (base + tail)

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
        bid_drift = 0.0
        ask_drift = 0.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_estimate = p["vol_decay"] * self.vol_estimate + (1.0 - p["vol_decay"]) * abs(move)
            self.trend_slow = p["trend_slow_decay"] * self.trend_slow + p["trend_slow_alpha"] * move
            self.trend_fast = p["trend_fast_decay"] * self.trend_fast + p["trend_fast_alpha"] * move
            shock_trigger = max(
                p["shock_trigger_min"],
                p["shock_trigger_vol_mult"] * max(self.vol_estimate, p["shock_vol_floor"]),
            )
            if abs(move) >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = 1 if move > 0 else -1
            bid_drift = bid - self.prev_bid
            ask_drift = ask - self.prev_ask

        self.prev_bid = bid
        self.prev_ask = ask

        drift_signal = (bid_drift + ask_drift) / 2.0
        self.arb_direction = p["dir_decay"] * self.arb_direction + p["dir_drift_weight"] * drift_signal

        fill_hit = p["fill_hit"]
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
            self.arb_direction -= state.buy_filled_quantity * p["dir_fill_weight"]
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
            self.arb_direction += state.sell_filled_quantity * p["dir_fill_weight"]
        self.fill_bias *= p["fill_decay"]

        net_inv = state.yes_inventory - state.no_inventory
        inv_shift = self._convex_inv_shift(net_inv)

        fb_stable = self.fill_bias * p["stable_fill_mult"]
        fair_stable = mid + fb_stable + inv_shift + self.trend_slow * p["trend_slow_weight"]

        ad = max(-p["dir_clamp"], min(p["dir_clamp"], self.arb_direction))
        fair_reactive = (
            mid
            + self.fill_bias
            + inv_shift
            + self.trend_fast * p["trend_fast_weight"]
            + p["reactive_dir_weight"] * ad
        )

        disagree_ticks = abs(fair_stable - fair_reactive)
        vol_ref = max(self.vol_estimate, p["vol_floor"])

        conf_stable = (1.0 / (1.0 + p["conf_vol_k"] * self.vol_estimate)) * math.exp(
            -disagree_ticks / max(p["conf_disagree_decay"], 1e-6)
        )
        conf_stable = max(p["conf_stable_floor"], conf_stable)

        tr_n = min(1.0, abs(self.trend_fast) / max(p["conf_reactive_trend_cap"], 1e-6))
        dir_n = min(1.0, abs(ad) / max(p["conf_reactive_dir_cap"], 1e-6))
        flow_align = 1.0
        if abs(self.fill_bias) > 0.08 and abs(self.trend_fast) > 0.06:
            same = (self.fill_bias > 0) == (self.trend_fast > 0)
            flow_align = 1.0 + p["flow_align_bonus"] if same else 1.0 - p["flow_misalign_penalty"]
        conf_reactive = p["conf_reactive_base"] + 0.38 * tr_n + 0.40 * dir_n
        conf_reactive *= max(0.55, flow_align)
        conf_reactive = max(1e-6, conf_reactive)

        w_stable = conf_stable / (conf_stable + conf_reactive)
        fair = w_stable * fair_stable + (1.0 - w_stable) * fair_reactive

        blend_uncertainty = disagree_ticks + 0.55 * self.vol_estimate

        half_spread = float(p["spread_base"])
        if self.vol_estimate > p["spread_vol_threshold"]:
            half_spread += float(p["spread_vol_extra"])
        if self.shock_remaining > 0:
            half_spread += float(p["spread_shock_extra"])
        half_spread += p["spread_disagree_coef"] * (disagree_ticks / max(p["spread_disagree_scale"], 1e-6))
        half_spread_i = int(max(1, round(half_spread)))

        my_bid = max(1, int(round(fair - half_spread_i)))
        my_ask = min(99, int(round(fair + half_spread_i)))
        if my_bid >= my_ask:
            return actions

        steps_rem = max(1, state.steps_remaining)
        prob_est = max(0.005, min(0.995, mid / 100.0))
        z = _norm_ppf_approx(prob_est)
        phi_z = _norm_pdf(z)
        total_sigma = 0.02 * math.sqrt(steps_rem)
        model_vol = phi_z / max(0.01, total_sigma) * 0.02 * 100.0
        model_vol = max(0.05, min(8.0, model_vol))
        model_boost = max(0.5, min(3.0, 1.0 / max(0.3, model_vol)))

        vol_scale = max(p["vol_floor"], 1.0 - self.vol_estimate * p["vol_coeff"])
        if self.shock_remaining > 0:
            vol_scale *= p["shock_size_mult"]

        unc_scale = 1.0 / (1.0 + p["size_uncertainty_k"] * blend_uncertainty)
        unc_scale = max(0.42, min(1.12, unc_scale))

        base_size = p["base_size"] * model_boost * unc_scale
        max_inv = max(p["max_inventory"], 1e-9)
        min_size = p["min_size"]
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        s_soft = p["side_damp_soft"]
        if net_inv > s_soft:
            bid_size *= p["damp_bid_when_long"]
        elif net_inv < -s_soft:
            ask_size *= p["damp_ask_when_short"]

        react_share = 1.0 - w_stable
        ds = ad
        abs_ds = abs(ds)
        if abs_ds > 0.06 and react_share > 0.35:
            safe_boost = 1.0 + react_share * abs_ds * p["dir_safe_mult"]
            risky_shrink = max(0.18, 1.0 - react_share * abs_ds * p["dir_risky_mult"])
            if ds > 0.0:
                bid_size *= safe_boost
                ask_size *= risky_shrink
            else:
                ask_size *= safe_boost
                bid_size *= risky_shrink

        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        ask_px = _tick_to_price(my_ask)
        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            pen = p["uncovered_penalty"]
            ask_size = max(min_size, avail_yes + max(0.0, uncovered * (1.0 - pen)))

        reserve_need = p["reserve_cash_base"] + p["reserve_cash_vol"] * vol_ref
        spendable = max(0.0, state.free_cash - reserve_need)

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            scale = spendable / max(buy_cost, 1e-12)
            bid_size *= scale
            if bid_size < min_size:
                bid_size = 0.0
            else:
                bid_size = max(min_size, bid_size)
            buy_cost = my_bid * 0.01 * bid_size

        free_after_bid = state.free_cash - buy_cost
        one_m_ask = max(1e-9, 1.0 - ask_px)
        if ask_size > 0.0:
            cov = min(ask_size, avail_yes)
            unc = max(0.0, ask_size - cov)
            coll = one_m_ask * unc
            if coll > free_after_bid + 1e-9:
                max_unc = max(0.0, free_after_bid / one_m_ask)
                new_ask = cov + max_unc
                if new_ask < min_size - 1e-9:
                    ask_size = 0.0
                else:
                    ask_size = max(min_size, new_ask)

        if bid_size >= 0.01 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size >= 0.01:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions
