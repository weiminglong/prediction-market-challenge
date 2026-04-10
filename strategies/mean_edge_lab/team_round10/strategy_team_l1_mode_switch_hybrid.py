"""L1 mode-switch hybrid: calm = multi-level retail capture (arb_hunter-style ladder);
toxic/shock = single-level model-vol MM (combined-style) with wider spreads and no outer bait."""

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
    _REGIME_TOXIC_ENTER = 0.57
    _REGIME_CALM_EXIT = 0.41
    _W_VOL = 0.26
    _W_TREND = 0.2
    _W_SHOCK = 0.34
    _W_FILL_TOX = 0.12
    _W_ARB_EMA = 0.18
    _VOL_CAP = 2.05
    _TREND_CAP = 1.78
    _TOX_CAP = 2.32
    _DANGER_SPREAD_RAW = 0.66

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_ema = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.quiet_steps = 0
        self.arb_fill_ema = 0.0
        self.regime_toxic = False

    def _convex_inv_shift(self, net_inv):
        if net_inv == 0.0:
            return 0.0
        sig = 1.0 if net_inv > 0 else -1.0
        max_inv = 8.7
        a = min(abs(net_inv), max_inv * 1.35)
        skew = 0.028
        soft = 4.6
        em = 0.016
        if a <= soft:
            t = a / soft
            w = t * t * (3.0 - 2.0 * t)
            return -sig * skew * a * w
        excess = a - soft
        base = skew * soft
        tail = skew * excess * (1.0 + em * excess * excess)
        return -sig * (base + tail)

    def _regime_raw(self) -> float:
        vol_n = min(1.0, self.vol_ema / self._VOL_CAP)
        tr_n = min(1.0, abs(self.trend) / self._TREND_CAP)
        shock_n = 1.0 if self.shock_remaining > 0 else 0.0
        fill_tox = min(1.0, abs(self.fill_bias) / self._TOX_CAP)
        return (
            self._W_VOL * vol_n
            + self._W_TREND * tr_n
            + self._W_SHOCK * shock_n
            + self._W_FILL_TOX * fill_tox
            + self._W_ARB_EMA * min(1.0, self.arb_fill_ema)
        )

    def _update_regime(self, raw: float) -> bool:
        if self.regime_toxic:
            if raw <= self._REGIME_CALM_EXIT:
                self.regime_toxic = False
        else:
            if raw >= self._REGIME_TOXIC_ENTER:
                self.regime_toxic = True
        return self.regime_toxic

    def _model_vol(self, mid: float, steps_rem: int) -> float:
        prob_est = max(0.005, min(0.995, mid / 100.0))
        z = _norm_ppf_approx(prob_est)
        phi_z = _norm_pdf(z)
        total_sigma = 0.02 * math.sqrt(max(1, steps_rem))
        model_vol = phi_z / max(0.01, total_sigma) * 0.02 * 100.0
        return max(0.05, min(8.0, model_vol))

    def on_step(self, state: StepState):
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
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_ema = 0.93 * self.vol_ema + 0.07 * abs(move)
            self.trend = 0.65 * self.trend + 0.17 * move
            shock_trigger = max(1.85, 3.0 * max(self.vol_ema, 0.35))
            if abs(move) >= shock_trigger:
                self.shock_remaining = 4
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        buy_qty = state.buy_filled_quantity
        sell_qty = state.sell_filled_quantity
        if buy_qty > 0:
            self.fill_bias -= 0.5
        if sell_qty > 0:
            self.fill_bias += 0.5
        self.fill_bias *= 0.59

        arb_like = (buy_qty > 1.0 and sell_qty < 0.5) or (sell_qty > 1.0 and buy_qty < 0.5)
        self.arb_fill_ema = 0.88 * self.arb_fill_ema + 0.12 * (1.0 if arb_like else 0.0)

        if not arb_like and buy_qty + sell_qty < 0.5:
            self.quiet_steps += 1
        else:
            self.quiet_steps = 0

        raw = self._regime_raw()
        toxic = self._update_regime(raw)

        net_inv = state.yes_inventory - state.no_inventory
        max_inv = 8.7
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend

        half_spread = 2
        if self.vol_ema > 1.2:
            half_spread += 1
        if self.shock_remaining > 0:
            half_spread += 2

        if toxic:
            half_spread += 1
            if raw >= self._DANGER_SPREAD_RAW:
                half_spread += 1

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(0.062, 1.0 - self.vol_ema * 2.38)
        if self.shock_remaining > 0:
            vol_scale *= 0.15
        if toxic:
            vol_scale *= 0.84

        if toxic:
            steps_rem = max(1, state.steps_remaining)
            mv = self._model_vol(mid, steps_rem)
            model_boost = max(0.5, min(3.0, 1.0 / max(0.3, mv)))
            base_size = 10.0 * model_boost
            if raw >= self._DANGER_SPREAD_RAW:
                base_size *= 0.9
        else:
            base_size = 15.2
            if self.vol_ema < 0.68 and raw < 0.36:
                base_size *= 1.04

        min_sz = 0.55
        bid_size = max(min_sz, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_sz, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if net_inv > 5.5:
            bid_size *= 0.94
        elif net_inv < -5.5:
            ask_size *= 0.94

        if not toxic:
            direction = self.trend + self.fill_bias * 0.24
            da = abs(direction)
            if da > 0.14:
                dm = max(0.38, 1.0 - da * 0.88)
                if direction > 0:
                    ask_size *= dm
                else:
                    bid_size *= dm
        else:
            if self.shock_remaining == 0 and raw >= 0.7:
                if self.trend <= -0.46:
                    bid_size *= 0.48
                    if raw >= 0.8:
                        bid_size = 0.0
                elif self.trend >= 0.46:
                    ask_size *= 0.48
                    if raw >= 0.8:
                        ask_size = 0.0

        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(min_sz, avail_yes + uncovered * 0.88)

        vol_ref = max(self.vol_ema, 0.064)
        spendable = max(0.0, state.free_cash - 5.0 * vol_ref)

        shock_rem = self.shock_remaining
        if shock_rem > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        if toxic:
            return self._execute_combined_single(
                actions,
                state,
                my_bid,
                my_ask,
                bid_size,
                ask_size,
                spendable,
                avail_yes,
            )

        return self._execute_calm_ladder(
            actions,
            state,
            bid,
            ask,
            my_bid,
            my_ask,
            bid_size,
            ask_size,
            spendable,
            avail_yes,
            net_inv,
            max_inv,
            vol_scale,
        )

    def _execute_combined_single(
        self,
        actions,
        state,
        my_bid,
        my_ask,
        bid_size,
        ask_size,
        spendable,
        avail_yes,
    ):
        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            bid_size *= spendable / max(buy_cost, 1e-12)
            if bid_size < 0.5:
                bid_size = 0.0
            buy_cost = my_bid * 0.01 * bid_size

        free_after_bid = state.free_cash - buy_cost
        if ask_size > 0.0:
            one_m_ask = max(1e-9, 1.0 - my_ask * 0.01)
            cov = min(ask_size, avail_yes)
            unc = max(0.0, ask_size - cov)
            if one_m_ask * unc > free_after_bid + 1e-9:
                new_ask = cov + max(0.0, free_after_bid / one_m_ask)
                ask_size = max(0.5, new_ask) if new_ask >= 0.5 else 0.0

        if bid_size > 0.0 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions

    def _execute_calm_ladder(
        self,
        actions,
        state,
        comp_bid,
        comp_ask,
        my_bid,
        my_ask,
        bid_size,
        ask_size,
        spendable,
        avail_yes,
        net_inv,
        max_inv,
        vol_scale,
    ):
        min_sz = 0.55
        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            scale = spendable / max(buy_cost, 1e-12)
            bid_size = max(min_sz, bid_size * scale) if bid_size * scale >= min_sz else 0.0
            buy_cost = my_bid * 0.01 * bid_size

        ask_px = my_ask * 0.01
        one_m_ask = max(1e-9, 1.0 - ask_px)
        free_after_bid = state.free_cash - buy_cost
        if ask_size > 0.0:
            cov = min(ask_size, avail_yes)
            unc = max(0.0, ask_size - cov)
            coll = one_m_ask * unc
            if coll > free_after_bid + 1e-9:
                max_unc = max(0.0, free_after_bid / one_m_ask)
                new_ask = cov + max_unc
                ask_size = max(min_sz, new_ask) if new_ask >= min_sz - 1e-9 else 0.0

        if bid_size > 0.0 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
            spendable -= buy_cost
        if ask_size > 0.0:
            cov_l1 = min(ask_size, avail_yes)
            unc_l1 = max(0.0, ask_size - cov_l1)
            coll_l1 = one_m_ask * unc_l1
            if coll_l1 <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
                avail_yes = max(0.0, avail_yes - ask_size)
                if coll_l1 > 0:
                    spendable -= coll_l1

        def add_level(extra: int, frac: float):
            nonlocal spendable, avail_yes
            b2 = max(min_sz, bid_size * frac) if bid_size > 0 else 0.0
            a2 = max(min_sz, ask_size * frac) if ask_size > 0 else 0.0
            bb = max(1, my_bid - extra)
            aa = min(99, my_ask + extra)
            if b2 > 0:
                c2 = bb * 0.01 * b2
                if c2 <= spendable:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=bb, quantity=b2))
                    spendable -= c2
            if a2 > 0:
                cv = min(a2, avail_yes)
                uc = max(0.0, a2 - cv)
                cl = (1.0 - aa * 0.01) * uc
                if cl <= spendable + 1e-9:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=aa, quantity=a2))
                    avail_yes = max(0.0, avail_yes - a2)
                    if cl > 0:
                        spendable -= cl

        add_level(5, 0.48)
        add_level(9, 0.28)
        add_level(14, 0.14)

        comp_spread = comp_ask - comp_bid
        if comp_spread >= 3 and self.quiet_steps >= 2 and self.shock_remaining <= 0:
            sent_sz = round(3.4 * vol_scale, 2)
            tick_b = max(1, comp_bid + 1)
            sz_b = round(max(0.1, sent_sz * max(0.0, 1.0 - net_inv / max_inv)), 2)
            cost_b = tick_b * 0.01 * sz_b
            if sz_b > 0.1 and cost_b <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=tick_b, quantity=sz_b))
                spendable -= cost_b
            tick_a = min(99, comp_ask - 1)
            sz_a = round(max(0.1, sent_sz * max(0.0, 1.0 + net_inv / max_inv)), 2)
            cov_s = min(sz_a, avail_yes)
            unc_s = max(0.0, sz_a - cov_s)
            coll_s = (1.0 - tick_a * 0.01) * unc_s
            if sz_a > 0.1 and coll_s <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=tick_a, quantity=sz_a))

        return actions
