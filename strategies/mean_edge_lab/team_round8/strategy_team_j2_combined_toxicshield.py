"""J2: strategy_combined + toxicity memory (adverse moves, one-sided fills).

EWMA bid/ask toxicity scores; inventory boosts the score on the vulnerable side.
Above a hard threshold the toxic side is fully disabled; elevated toxicity also
refreshes a short global half-spread widen on both sides. Model-based sizing,
cash/spendable checks, shock gating, and fill_bias path match the combined baseline.
"""

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
    # Toxicity / shield — tuned for robust mean edge vs baseline combined
    TOX_DECAY = 0.865
    TOX_MOVE_VOL_MULT = 1.08
    TOX_MOVE_EPS = 0.065
    TOX_ONE_SIDED_RATIO = 1.95
    TOX_FILL_WEIGHT = 0.52
    TOX_CAP = 3.15
    TOX_INV_WEIGHT = 0.22
    TOX_HARD = 1.12
    TOX_WIDEN = 0.58
    TOX_SHIELD_STEPS = 5
    TOX_SHIELD_EXTRA_SPREAD = 1

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_ema = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.tox_bid = 0.0
        self.tox_ask = 0.0
        self.shield_left = 0

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

    def _update_toxicity(self, move: float, buy_qty: float, sell_qty: float) -> None:
        vol_denom = max(self.vol_ema, 0.06, abs(move) + 1e-6)
        mvm = max(self.TOX_MOVE_VOL_MULT, 1e-6)
        down_stress = max(0.0, -move) / (vol_denom * mvm)
        up_stress = max(0.0, move) / (vol_denom * mvm)

        ratio = max(self.TOX_ONE_SIDED_RATIO, 1.01)
        eps = self.TOX_MOVE_EPS
        buy_heavy = buy_qty > 0.35 and buy_qty > sell_qty * ratio
        sell_heavy = sell_qty > 0.35 and sell_qty > buy_qty * ratio

        inst_bid = down_stress
        if buy_heavy and move < -eps:
            inst_bid += self.TOX_FILL_WEIGHT * min(buy_qty, 8.0) / 8.0

        inst_ask = up_stress
        if sell_heavy and move > eps:
            inst_ask += self.TOX_FILL_WEIGHT * min(sell_qty, 8.0) / 8.0

        d = self.TOX_DECAY
        cap = self.TOX_CAP
        self.tox_bid = min(d * self.tox_bid + (1.0 - d) * inst_bid, cap)
        self.tox_ask = min(d * self.tox_ask + (1.0 - d) * inst_ask, cap)

    def _tox_effective(self, net_inv: float) -> tuple[float, float]:
        max_inv = 8.7
        iw = self.TOX_INV_WEIGHT
        eff_bid = self.tox_bid * (1.0 + iw * max(0.0, min(1.0, net_inv / max_inv)))
        eff_ask = self.tox_ask * (1.0 + iw * max(0.0, min(1.0, -net_inv / max_inv)))
        return eff_bid, eff_ask

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

        move = 0.0
        had_prev = self.prev_bid is not None and self.prev_ask is not None
        if had_prev:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_ema = 0.93 * self.vol_ema + 0.07 * abs(move)
            self.trend = 0.65 * self.trend + 0.17 * move
            shock_trigger = max(1.85, 3.0 * max(self.vol_ema, 0.35))
            if abs(move) >= shock_trigger:
                self.shock_remaining = 4
                self.shock_sign = 1 if move > 0 else -1

        if had_prev:
            self._update_toxicity(move, state.buy_filled_quantity, state.sell_filled_quantity)

        self.prev_bid = bid_t
        self.prev_ask = ask_t

        if state.buy_filled_quantity > 0:
            self.fill_bias -= 0.5
        if state.sell_filled_quantity > 0:
            self.fill_bias += 0.5
        self.fill_bias *= 0.59

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend

        eff_bid, eff_ask = self._tox_effective(net_inv)
        max_eff = max(eff_bid, eff_ask)
        if max_eff >= self.TOX_WIDEN:
            self.shield_left = max(self.shield_left, self.TOX_SHIELD_STEPS)

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
        if self.shield_left > 0:
            half_spread += self.TOX_SHIELD_EXTRA_SPREAD
            self.shield_left -= 1

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        model_boost = max(0.5, min(3.0, 1.0 / max(0.3, model_vol)))
        base_size = 10.0 * model_boost
        vol_scale = max(0.06, 1.0 - self.vol_ema * 2.4)
        if self.shock_remaining > 0:
            vol_scale *= 0.15

        max_inv = 8.7
        bid_size = max(0.5, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.5, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if net_inv > 5.5:
            bid_size *= 0.94
        elif net_inv < -5.5:
            ask_size *= 0.94

        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(0.5, avail_yes + uncovered * 0.88)

        vol_ref = max(self.vol_ema, 0.06)
        spendable = max(0.0, state.free_cash - 5.0 * vol_ref)

        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        if eff_bid >= self.TOX_HARD:
            bid_size = 0.0
        if eff_ask >= self.TOX_HARD:
            ask_size = 0.0

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
