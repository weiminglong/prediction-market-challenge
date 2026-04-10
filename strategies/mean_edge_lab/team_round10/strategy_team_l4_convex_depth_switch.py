"""L4: convex depth switch — quote *depth* (ladder vs flat vs shallow) follows risk state.

Three-state depth policy machine with hysteresis:
  CALM: sparse touch + deep passive levels (retail capture away from NBBO).
  INTERMEDIATE: standard single-level two-sided core (combined-style).
  DANGER: shallow L1 only + one-sided defensive inventory/trend bias.

Strict per-order cash/collateral feasibility like multilevel.
"""
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


# Depth state constants (avoid Enum for minimal surface)
_DEPTH_CALM = 0
_DEPTH_MID = 1
_DEPTH_DANGER = 2


class Strategy(BaseStrategy):
    # Risk score -> depth state (hysteresis on boundaries)
    _W_VOL = 0.30
    _W_TREND = 0.20
    _W_SHOCK = 0.30
    _W_TOX = 0.20
    _VOL_CAP = 2.05
    _TREND_CAP = 1.78
    _TOX_CAP = 2.32
    _UP_CALM = 0.40
    _DN_MID = 0.24
    _UP_DANGER = 0.58
    _DN_DANGER = 0.46

    _MIN_SIZE = 0.5

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_ema = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.depth_state = _DEPTH_MID

    def _convex_inv_shift(self, net_inv: float) -> float:
        if net_inv == 0.0:
            return 0.0
        sign = 1.0 if net_inv > 0 else -1.0
        a = min(abs(net_inv), 12.0)
        soft = 4.6
        skew = 0.028
        edge_mult = 0.016
        if a <= soft:
            t = a / soft
            w = t * t * (3.0 - 2.0 * t)
            return -sign * skew * a * w
        excess = a - soft
        return -sign * (skew * soft + skew * excess * (1.0 + edge_mult * excess * excess))

    def _risk_raw(self) -> float:
        vol_n = min(1.0, self.vol_ema / self._VOL_CAP)
        tr_n = min(1.0, abs(self.trend) / self._TREND_CAP)
        shock_n = 1.0 if self.shock_remaining > 0 else 0.0
        tox = min(1.0, abs(self.fill_bias) / self._TOX_CAP)
        return (
            self._W_VOL * vol_n
            + self._W_TREND * tr_n
            + self._W_SHOCK * shock_n
            + self._W_TOX * tox
        )

    def _update_depth_state(self, raw: float) -> int:
        if self.depth_state == _DEPTH_CALM:
            if raw >= self._UP_CALM:
                self.depth_state = _DEPTH_MID
        elif self.depth_state == _DEPTH_MID:
            if raw <= self._DN_MID:
                self.depth_state = _DEPTH_CALM
            elif raw >= self._UP_DANGER:
                self.depth_state = _DEPTH_DANGER
        else:
            if raw <= self._DN_DANGER:
                self.depth_state = _DEPTH_MID
        return self.depth_state

    def _place_deep_level(
        self,
        actions: list,
        side: Side,
        price_ticks: int,
        qty: float,
        avail_yes: float,
        spendable: float,
    ) -> tuple[float, float]:
        """Return (new_avail_yes, new_spendable) after optional place."""
        min_sz = self._MIN_SIZE
        if qty < min_sz - 1e-9:
            return avail_yes, spendable
        px = price_ticks * 0.01
        if side == Side.BUY:
            cost = px * qty
            if cost > spendable + 1e-9:
                return avail_yes, spendable
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=price_ticks, quantity=qty))
            return avail_yes, spendable - cost
        cov = min(qty, avail_yes)
        unc = max(0.0, qty - cov)
        one_m = max(1e-9, 1.0 - px)
        coll = one_m * unc
        if coll > spendable + 1e-9:
            return avail_yes, spendable
        actions.append(PlaceOrder(side=Side.SELL, price_ticks=price_ticks, quantity=qty))
        return max(0.0, avail_yes - qty), spendable - coll

    def on_step(self, state: StepState) -> list:
        actions: list = [CancelAll()]
        bid_t = state.competitor_best_bid_ticks
        ask_t = state.competitor_best_ask_ticks
        if bid_t is None and ask_t is None:
            return actions
        if bid_t is None:
            bid_t = max(1, ask_t - 6)
        if ask_t is None:
            ask_t = min(99, bid_t + 6)
        mid = (bid_t + ask_t) / 2.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_ema = 0.93 * self.vol_ema + 0.07 * abs(move)
            self.trend = 0.65 * self.trend + 0.17 * move
            shock_trigger = max(1.85, 3.0 * max(self.vol_ema, 0.35))
            if abs(move) >= shock_trigger:
                self.shock_remaining = 4
                self.shock_sign = 1 if move > 0 else -1
        self.prev_bid = bid_t
        self.prev_ask = ask_t

        if state.buy_filled_quantity > 0:
            self.fill_bias -= 0.5
        if state.sell_filled_quantity > 0:
            self.fill_bias += 0.5
        self.fill_bias *= 0.59

        raw = self._risk_raw()
        dstate = self._update_depth_state(raw)

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend

        steps_rem = max(1, state.steps_remaining)
        prob_est = max(0.005, min(0.995, mid / 100.0))
        z = _norm_ppf_approx(prob_est)
        phi_z = _norm_pdf(z)
        total_sigma = 0.02 * math.sqrt(steps_rem)
        model_vol = phi_z / max(0.01, total_sigma) * 0.02 * 100.0
        model_vol = max(0.05, min(8.0, model_vol))

        half_spread = 2
        if self.vol_ema > 1.2:
            half_spread += 1
        if self.shock_remaining > 0:
            half_spread += 2
        if dstate == _DEPTH_DANGER:
            half_spread += 1

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        model_boost = max(0.5, min(3.0, 1.0 / max(0.3, model_vol)))
        base_size = 10.0 * model_boost
        if dstate == _DEPTH_CALM:
            base_size *= 1.02
        elif dstate == _DEPTH_DANGER:
            base_size *= 0.94

        vol_scale = max(0.06, 1.0 - self.vol_ema * 2.4)
        if self.shock_remaining > 0:
            vol_scale *= 0.15
        if dstate == _DEPTH_DANGER:
            vol_scale *= 0.92

        max_inv = 8.7
        min_sz = self._MIN_SIZE
        bid_size = max(min_sz, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_sz, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if dstate == _DEPTH_CALM:
            bid_size *= 0.88
            ask_size *= 0.88

        if net_inv > 5.5:
            bid_size *= 0.94
        elif net_inv < -5.5:
            ask_size *= 0.94

        # DANGER: one-sided defensive (inventory-first, then trend)
        if dstate == _DEPTH_DANGER and self.shock_remaining == 0:
            inv_gate = 2.8
            tr_gate = 0.42
            if net_inv > inv_gate:
                bid_size *= 0.35
                if raw >= 0.68:
                    bid_size = 0.0
            elif net_inv < -inv_gate:
                ask_size *= 0.35
                if raw >= 0.68:
                    ask_size = 0.0
            else:
                if self.trend >= tr_gate:
                    ask_size *= 0.42
                elif self.trend <= -tr_gate:
                    bid_size *= 0.42

        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(min_sz, avail_yes + uncovered * 0.88)

        vol_ref = max(self.vol_ema, 0.06)
        spendable = max(0.0, state.free_cash - 5.0 * vol_ref)

        shock_rem_after = self.shock_remaining
        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            bid_size *= spendable / max(buy_cost, 1e-12)
            if bid_size < min_sz:
                bid_size = 0.0
            buy_cost = my_bid * 0.01 * bid_size

        free_after_bid = state.free_cash - buy_cost
        one_m_ask = max(1e-9, 1.0 - my_ask * 0.01)
        if ask_size > 0.0:
            cov = min(ask_size, avail_yes)
            unc = max(0.0, ask_size - cov)
            if one_m_ask * unc > free_after_bid + 1e-9:
                new_ask = cov + max(0.0, free_after_bid / one_m_ask)
                ask_size = max(min_sz, new_ask) if new_ask >= min_sz else 0.0

        # L1 placement + deep ladder only in CALM
        if bid_size > 0.0 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
            spend_work = spendable - buy_cost
        else:
            spend_work = spendable

        ay = avail_yes
        if ask_size > 0.0:
            cov_l1 = min(ask_size, ay)
            unc_l1 = max(0.0, ask_size - cov_l1)
            coll_l1 = one_m_ask * unc_l1
            if coll_l1 <= spend_work + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
                ay = max(0.0, ay - ask_size)
                spend_work -= coll_l1

        if dstate == _DEPTH_CALM and shock_rem_after == 0:
            l2_ex, l3_ex = 6, 12
            l2f, l3f = 0.46, 0.28
            bid_l2 = max(1, my_bid - l2_ex)
            ask_l2 = min(99, my_ask + l2_ex)
            bid_l3 = max(1, my_bid - l3_ex)
            ask_l3 = min(99, my_ask + l3_ex)
            b2 = max(min_sz, bid_size * l2f) if bid_size > 0 else 0.0
            a2 = max(min_sz, ask_size * l2f) if ask_size > 0 else 0.0
            b3 = max(min_sz, bid_size * l3f) if bid_size > 0 else 0.0
            a3 = max(min_sz, ask_size * l3f) if ask_size > 0 else 0.0
            ay, spend_work = self._place_deep_level(actions, Side.BUY, bid_l2, b2, ay, spend_work)
            ay, spend_work = self._place_deep_level(actions, Side.SELL, ask_l2, a2, ay, spend_work)
            ay, spend_work = self._place_deep_level(actions, Side.BUY, bid_l3, b3, ay, spend_work)
            ay, spend_work = self._place_deep_level(actions, Side.SELL, ask_l3, a3, ay, spend_work)

        return actions
