"""j3 regime-switch core + small passive L2/L3 only in strong calm risk-on."""
from __future__ import annotations

import math

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _norm_ppf_approx(p):
    if p <= 0.001:
        return -3.09
    if p >= 0.999:
        return 3.09
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t):
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


class Strategy(BaseStrategy):
    _REGIME_HIGH = 0.58
    _REGIME_LOW = 0.40
    _W_VOL = 0.28
    _W_TREND = 0.22
    _W_SHOCK = 0.32
    _W_TOX = 0.18
    _VOL_CAP = 2.0
    _TREND_CAP = 1.75
    _TOX_CAP = 2.35
    _DANGER_EXTRA_SPREAD = 0.70
    # Passive depth: only in calm risk-on (multilevel-style extras, smaller than round4)
    _DEPTH_REGIME_MAX = 0.33
    _DEPTH_VOL_MAX = 0.58
    _DEPTH_TOX_ABS_MAX = 0.90
    _DEPTH_TREND_ABS_MAX = 0.58
    _L2_EXTRA = 4
    _L3_EXTRA = 7
    _L2_FRAC = 0.36
    _L3_FRAC = 0.21
    _MIN_DEPTH = 0.5

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_ema = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.regime_risk_off = False

    def _convex_inv_shift(self, net_inv):
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

    def _regime_raw(self) -> float:
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

    def _update_regime(self, raw: float) -> bool:
        if self.regime_risk_off:
            if raw <= self._REGIME_LOW:
                self.regime_risk_off = False
        else:
            if raw >= self._REGIME_HIGH:
                self.regime_risk_off = True
        return self.regime_risk_off

    def _depth_allowed(self, risk_off: bool, shock_active: bool, regime_raw: float) -> bool:
        if risk_off or shock_active:
            return False
        if regime_raw > self._DEPTH_REGIME_MAX:
            return False
        if self.vol_ema >= self._DEPTH_VOL_MAX:
            return False
        if abs(self.fill_bias) >= self._DEPTH_TOX_ABS_MAX:
            return False
        if abs(self.trend) >= self._DEPTH_TREND_ABS_MAX:
            return False
        return True

    def on_step(self, state: StepState):
        actions = [CancelAll()]
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

        regime_raw = self._regime_raw()
        risk_off = self._update_regime(regime_raw)

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

        if risk_off:
            half_spread += 1
            if regime_raw >= self._DANGER_EXTRA_SPREAD:
                half_spread += 1

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        model_boost = max(0.5, min(3.0, 1.0 / max(0.3, model_vol)))
        base_size = 10.0 * model_boost
        if risk_off:
            base_size *= 0.88
            if regime_raw >= self._DANGER_EXTRA_SPREAD:
                base_size *= 0.93
        else:
            base_size *= 1.035
            if self.shock_remaining == 0 and self.vol_ema < 0.72 and regime_raw < 0.38:
                base_size *= 1.025

        vol_scale = max(0.06, 1.0 - self.vol_ema * 2.4)
        if self.shock_remaining > 0:
            vol_scale *= 0.15
        if risk_off:
            vol_scale *= 0.86
        else:
            vol_scale = min(1.08, vol_scale * 1.03)

        max_inv = 8.7
        bid_size = max(0.5, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.5, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if net_inv > 5.5:
            bid_size *= 0.94
        elif net_inv < -5.5:
            ask_size *= 0.94

        if risk_off and self.shock_remaining == 0 and regime_raw >= 0.68:
            if self.trend <= -0.48:
                bid_size *= 0.45
                if regime_raw >= 0.78:
                    bid_size = 0.0
            elif self.trend >= 0.48:
                ask_size *= 0.45
                if regime_raw >= 0.78:
                    ask_size = 0.0

        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(0.5, avail_yes + uncovered * 0.88)

        vol_ref = max(self.vol_ema, 0.06)
        spendable = max(0.0, state.free_cash - 5.0 * vol_ref)

        shock_active = self.shock_remaining > 0
        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            bid_size *= spendable / max(buy_cost, 1e-12)
            if bid_size < 0.5:
                bid_size = 0.0
            buy_cost = my_bid * 0.01 * bid_size

        free_after_bid = state.free_cash - buy_cost
        one_m_ask = max(1e-9, 1.0 - my_ask * 0.01)
        if ask_size > 0.0:
            cov = min(ask_size, avail_yes)
            unc = max(0.0, ask_size - cov)
            if one_m_ask * unc > free_after_bid + 1e-9:
                new_ask = cov + max(0.0, free_after_bid / one_m_ask)
                ask_size = max(0.5, new_ask) if new_ask >= 0.5 else 0.0

        depth_ok = self._depth_allowed(risk_off, shock_active, regime_raw)

        buy_cost = my_bid * 0.01 * bid_size
        spendable_rem = spendable
        avail_sim = avail_yes
        if bid_size > 0.0 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
            spendable_rem = max(0.0, spendable_rem - buy_cost)

        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
            cov_l1 = min(ask_size, avail_sim)
            unc_l1 = max(0.0, ask_size - cov_l1)
            coll_l1 = one_m_ask * unc_l1
            avail_sim = max(0.0, avail_sim - ask_size)
            if coll_l1 > 0:
                spendable_rem = max(0.0, spendable_rem - coll_l1)

        if not depth_ok:
            return actions

        l2e = self._L2_EXTRA
        l3e = self._L3_EXTRA
        ms = self._MIN_DEPTH

        bid_L2 = max(1, my_bid - l2e)
        ask_L2 = min(99, my_ask + l2e)
        bid_sz2 = max(ms, bid_size * self._L2_FRAC) if bid_size > 0 else 0.0
        ask_sz2 = max(ms, ask_size * self._L2_FRAC) if ask_size > 0 else 0.0

        if bid_sz2 > 0:
            c2 = bid_L2 * 0.01 * bid_sz2
            if c2 <= spendable_rem + 1e-9:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_L2, quantity=bid_sz2))
                spendable_rem = max(0.0, spendable_rem - c2)

        if ask_sz2 > 0:
            om2 = max(1e-9, 1.0 - ask_L2 * 0.01)
            cv2 = min(ask_sz2, avail_sim)
            uc2 = max(0.0, ask_sz2 - cv2)
            cl2 = om2 * uc2
            if cl2 <= spendable_rem + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_L2, quantity=ask_sz2))
                avail_sim = max(0.0, avail_sim - ask_sz2)
                if cl2 > 0:
                    spendable_rem = max(0.0, spendable_rem - cl2)

        bid_L3 = max(1, my_bid - l3e)
        ask_L3 = min(99, my_ask + l3e)
        bid_sz3 = max(ms, bid_size * self._L3_FRAC) if bid_size > 0 else 0.0
        ask_sz3 = max(ms, ask_size * self._L3_FRAC) if ask_size > 0 else 0.0

        if bid_sz3 > 0:
            c3 = bid_L3 * 0.01 * bid_sz3
            if c3 <= spendable_rem + 1e-9:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_L3, quantity=bid_sz3))
                spendable_rem = max(0.0, spendable_rem - c3)

        if ask_sz3 > 0:
            om3 = max(1e-9, 1.0 - ask_L3 * 0.01)
            cv3 = min(ask_sz3, avail_sim)
            uc3 = max(0.0, ask_sz3 - cv3)
            cl3 = om3 * uc3
            if cl3 <= spendable_rem + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_L3, quantity=ask_sz3))

        return actions
