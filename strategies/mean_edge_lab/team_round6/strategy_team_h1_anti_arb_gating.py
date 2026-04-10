"""Round 6 H1: G1 + anti-arb gating from adverse move bursts (one-sided pull)."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _tick_to_price(price_ticks: int) -> float:
    return price_ticks * 0.01


class Strategy(BaseStrategy):
    PARAMS = {
        "base_size": 14.113052,
        "fill_decay": 0.590229,
        "fill_hit": 0.497805,
        "inv_skew": 0.027657,
        "inv_soft": 4.6,
        "inv_edge_mult": 0.0155,
        "max_inventory": 8.674922,
        "min_size": 0.550072,
        "side_damp_soft": 5.5,
        "damp_bid_when_long": 0.94,
        "damp_ask_when_short": 0.94,
        "uncovered_penalty": 0.12,
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
        "reserve_cash_base": 0.0,
        "reserve_cash_vol": 5.0,
        "regime_memory_decay": 0.988,
        "regime_vol_norm": 2.2,
        "regime_trend_norm": 4.5,
        "regime_shift_vol_scale": 1.8,
        "regime_shift_trend_scale": 2.4,
        "regime_hyst_enter": 0.58,
        "regime_hyst_exit": 0.38,
        "regime_spread_extra": 1,
        "regime_fair_damp": 0.22,
        "regime_size_damp": 0.12,
        # Anti-arb: adverse mid bursts -> suppress toxic side for a few steps
        "arb_trig_min": 1.48,
        "arb_trig_vol_mult": 2.55,
        "arb_score_decay": 0.805,
        "arb_score_hit": 1.22,
        "arb_gate_enter": 2.28,
        "arb_gate_steps": 5,
        "arb_after_burst_damp": 0.38,
        "arb_inv_weight": 0.44,
    }

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.regime_memory = 0.0
        self.regime_latched = False
        self._prev_vol_for_shift: float | None = None
        self._prev_trend_for_shift: float | None = None
        self.tox_bid_score = 0.0
        self.tox_ask_score = 0.0
        self.arb_suppress_bid = 0
        self.arb_suppress_ask = 0

    def _convex_inv_shift(self, net_inv: float) -> float:
        p = self.PARAMS
        if net_inv == 0.0:
            return 0.0
        sig = 1.0 if net_inv > 0 else -1.0
        max_inv = max(float(p["max_inventory"]), 1e-6)
        a = min(abs(net_inv), max_inv * 1.4)
        skew = float(p["inv_skew"])
        soft = max(float(p["inv_soft"]), 1e-6)
        em = float(p["inv_edge_mult"])
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
        had_prev = self.prev_bid is not None and self.prev_ask is not None
        if had_prev:
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
            if abs(move) >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        v_ref = max(self.vol_estimate, float(p["vol_floor"]))
        vol_level = min(1.0, self.vol_estimate / max(float(p["regime_vol_norm"]), 1e-6))
        trend_level = min(1.0, abs(self.trend) / max(float(p["regime_trend_norm"]), 1e-6))
        shift_vol = 0.0
        shift_tr = 0.0
        if self._prev_vol_for_shift is not None and self._prev_trend_for_shift is not None:
            dv = abs(self.vol_estimate - self._prev_vol_for_shift)
            dt = abs(self.trend - self._prev_trend_for_shift)
            shift_vol = min(1.0, dv * float(p["regime_shift_vol_scale"]) / v_ref)
            shift_tr = min(1.0, dt * float(p["regime_shift_trend_scale"]))
        self._prev_vol_for_shift = self.vol_estimate
        self._prev_trend_for_shift = self.trend
        instant = max(vol_level, trend_level, 0.55 * shift_vol + 0.55 * shift_tr)
        mem_decay = float(p["regime_memory_decay"])
        self.regime_memory = mem_decay * self.regime_memory + (1.0 - mem_decay) * instant
        hi = float(p["regime_hyst_enter"])
        lo = float(p["regime_hyst_exit"])
        if not self.regime_latched and self.regime_memory >= hi:
            self.regime_latched = True
        elif self.regime_latched and self.regime_memory <= lo:
            self.regime_latched = False

        fill_hit = float(p["fill_hit"])
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
        self.fill_bias *= float(p["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        max_inv = max(float(p["max_inventory"]), 1e-9)

        # Adverse-move toxicity: down-moves poison bids; up-moves poison asks
        if had_prev:
            arb_trig = max(
                float(p["arb_trig_min"]),
                float(p["arb_trig_vol_mult"]) * max(self.vol_estimate, float(p["vol_floor"])),
            )
            d_arb = float(p["arb_score_decay"])
            self.tox_bid_score *= d_arb
            self.tox_ask_score *= d_arb
            hit = float(p["arb_score_hit"])
            if move <= -arb_trig:
                self.tox_bid_score += hit
            elif move >= arb_trig:
                self.tox_ask_score += hit

        iw = float(p["arb_inv_weight"])
        inv_b = 1.0 + iw * max(0.0, min(1.0, net_inv / max_inv))
        inv_a = 1.0 + iw * max(0.0, min(1.0, -net_inv / max_inv))
        eff_bid_tox = self.tox_bid_score * inv_b
        eff_ask_tox = self.tox_ask_score * inv_a
        enter = float(p["arb_gate_enter"])
        steps = int(p["arb_gate_steps"])
        damp = float(p["arb_after_burst_damp"])
        if eff_bid_tox >= enter:
            self.arb_suppress_bid = max(self.arb_suppress_bid, steps)
            self.tox_bid_score *= damp
        if eff_ask_tox >= enter:
            self.arb_suppress_ask = max(self.arb_suppress_ask, steps)
            self.tox_ask_score *= damp

        trend_w = float(p["trend_weight"])
        fair_trend = self.trend * trend_w
        if self.regime_latched:
            damp_f = float(p["regime_fair_damp"])
            fair_trend *= 1.0 - damp_f
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + fair_trend

        half_spread = int(p["spread_base"])
        if self.vol_estimate > float(p["spread_vol_threshold"]):
            half_spread += int(p["spread_vol_extra"])
        if self.shock_remaining > 0:
            half_spread += int(p["spread_shock_extra"])
        if self.regime_latched:
            half_spread += int(p["regime_spread_extra"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(p["vol_floor"]), 1.0 - self.vol_estimate * float(p["vol_coeff"]))
        if self.shock_remaining > 0:
            vol_scale *= float(p["shock_size_mult"])
        if self.regime_latched:
            vol_scale *= max(0.55, 1.0 - float(p["regime_size_damp"]))

        base_size = float(p["base_size"])
        min_size = float(p["min_size"])
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        s_soft = float(p["side_damp_soft"])
        if net_inv > s_soft:
            bid_size *= float(p["damp_bid_when_long"])
        elif net_inv < -s_soft:
            ask_size *= float(p["damp_ask_when_short"])

        ask_px = _tick_to_price(my_ask)
        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            pen = float(p["uncovered_penalty"])
            ask_size = max(min_size, avail_yes + max(0.0, uncovered * (1.0 - pen)))

        vol_ref = max(self.vol_estimate, float(p["vol_floor"]))
        reserve_need = float(p["reserve_cash_base"]) + float(p["reserve_cash_vol"]) * vol_ref
        spendable = max(0.0, state.free_cash - reserve_need)

        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        # Anti-arb: drop toxic side entirely while suppression timer active
        if self.arb_suppress_bid > 0:
            bid_size = 0.0
        if self.arb_suppress_ask > 0:
            ask_size = 0.0

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            scale = spendable / max(buy_cost, 1e-12)
            bid_size = bid_size * scale
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

        if bid_size > 0.0 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        if self.arb_suppress_bid > 0:
            self.arb_suppress_bid -= 1
        if self.arb_suppress_ask > 0:
            self.arb_suppress_ask -= 1
        return actions
