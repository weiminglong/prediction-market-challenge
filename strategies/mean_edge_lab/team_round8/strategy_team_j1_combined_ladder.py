"""J1: strategy_combined core + conditional deeper bid/ask ladder (L2–L4).

Keeps combined Gaussian-ish sizing, convex inventory skew, shock handling, and
cash/collateral checks. When gates indicate benign microstructure (low vol_ema,
no active shock window, low adverse-fill toxicity proxy, moderate trend), total
bid/ask size is split across multiple price rungs; otherwise behavior matches
base L1-only combined.
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


def _sell_collateral(ask_tick: int, ask_size: float, yes_inventory: float) -> float:
    price = ask_tick * 0.01
    covered = min(max(0.0, yes_inventory), ask_size)
    uncovered = max(0.0, ask_size - covered)
    return max(0.0, 1.0 - price) * uncovered


class Strategy(BaseStrategy):
    # Ladder gating (lightweight tune)
    VOL_LADDER_MAX = 0.40
    TOX_LADDER_MAX = 0.30
    TREND_LADDER_MAX = 0.72
    TOX_MOVE_TH = 0.34
    TOX_EMA = 0.88
    TOX_CAP = 2.4

    # L4 only in very calm tape
    VOL_L4_MAX = 0.28
    TOX_L4_MAX = 0.16
    TREND_L4_MAX = 0.42

    # Extra rungs: fractions of base L1 size (additive, not split from L1)
    DEPTH_STEPS = (2, 4, 6)
    EXTRA_FRACS_2 = (0.10, 0.06)
    EXTRA_FRACS_3 = (0.09, 0.06, 0.04)

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_ema = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.toxic_ema = 0.0

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

    def _merge_price_size(self, prices: list[int], sizes: list[float], min_leg: float):
        out_p: list[int] = []
        out_s: list[float] = []
        for p, s in zip(prices, sizes):
            if s < min_leg:
                continue
            if out_p and p == out_p[-1]:
                out_s[-1] = round(out_s[-1] + s, 4)
            else:
                out_p.append(p)
                out_s.append(s)
        return out_p, out_s

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
            toxic_bump = 0.0
            if state.buy_filled_quantity > 0.0 and move < -self.TOX_MOVE_TH:
                toxic_bump += min(self.TOX_CAP, abs(move))
            if state.sell_filled_quantity > 0.0 and move > self.TOX_MOVE_TH:
                toxic_bump += min(self.TOX_CAP, abs(move))
            self.toxic_ema = self.TOX_EMA * self.toxic_ema + (1.0 - self.TOX_EMA) * toxic_bump
        self.prev_bid = bid_t
        self.prev_ask = ask_t

        if state.buy_filled_quantity > 0:
            self.fill_bias -= 0.5
        if state.sell_filled_quantity > 0:
            self.fill_bias += 0.5
        self.fill_bias *= 0.59

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

        shock_sign = self.shock_sign
        shock_rem_before = self.shock_remaining
        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        buy_cost_l1 = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost_l1 > spendable:
            bid_size *= spendable / max(buy_cost_l1, 1e-12)
            if bid_size < 0.5:
                bid_size = 0.0
            buy_cost_l1 = my_bid * 0.01 * bid_size

        free_after_bid = state.free_cash - buy_cost_l1
        if ask_size > 0.0:
            one_m_ask = max(1e-9, 1.0 - my_ask * 0.01)
            cov = min(ask_size, avail_yes)
            unc = max(0.0, ask_size - cov)
            if one_m_ask * unc > free_after_bid + 1e-9:
                new_ask = cov + max(0.0, free_after_bid / one_m_ask)
                ask_size = max(0.5, new_ask) if new_ask >= 0.5 else 0.0

        ladder_ok = (
            shock_rem_before == 0
            and self.vol_ema <= self.VOL_LADDER_MAX
            and self.toxic_ema <= self.TOX_LADDER_MAX
            and abs(self.trend) <= self.TREND_LADDER_MAX
            and abs(net_inv) < 4.6
        )
        use_l4 = (
            ladder_ok
            and self.vol_ema <= self.VOL_L4_MAX
            and self.toxic_ema <= self.TOX_L4_MAX
            and abs(self.trend) <= self.TREND_L4_MAX
        )
        n_extra = 3 if use_l4 else 2

        if not ladder_ok or (bid_size <= 0.0 and ask_size <= 0.0):
            if bid_size > 0.0 and buy_cost_l1 <= state.free_cash and buy_cost_l1 <= spendable + 1e-6:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
            if ask_size > 0.0:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
            return actions

        fracs = list(self.EXTRA_FRACS_3 if n_extra == 3 else self.EXTRA_FRACS_2)
        steps = list(self.DEPTH_STEPS[:n_extra])

        raw_eb = [round(bid_size * f, 4) for f in fracs]
        raw_ea = [round(ask_size * f, 4) for f in fracs]
        if net_inv > 1.15:
            raw_eb = [round(x * 0.22, 4) for x in raw_eb]
        elif net_inv < -1.15:
            raw_ea = [round(x * 0.22, 4) for x in raw_ea]
        raw_ep = [max(1, my_bid - s) for s in steps]
        raw_ap = [min(99, my_ask + s) for s in steps]

        l1_buy = my_bid * 0.01 * bid_size if bid_size > 0.0 else 0.0
        coll_l1 = _sell_collateral(my_ask, ask_size, state.yes_inventory) if ask_size >= 0.5 else 0.0

        k = 1.0
        ep: list[int] = []
        eb: list[float] = []
        ap: list[int] = []
        ea: list[float] = []
        for _ in range(28):
            sb = [max(0.0, round(x * k, 4)) for x in raw_eb]
            sa = [max(0.0, round(x * k, 4)) for x in raw_ea]
            ep, eb = self._merge_price_size(list(raw_ep), sb, 0.5)
            ap, ea = self._merge_price_size(list(raw_ap), sa, 0.5)
            extra_buy_cost = sum(p * 0.01 * s for p, s in zip(ep, eb))
            tot_buy = l1_buy + extra_buy_cost
            if bid_size > 0.0 and tot_buy > spendable + 1e-9:
                k *= 0.88
                continue
            if tot_buy > state.free_cash + 1e-9:
                k *= 0.88
                continue
            ry = state.yes_inventory
            if ask_size >= 0.5:
                ry = max(0.0, ry - ask_size)
            cex = 0.0
            for p, s in zip(ap, ea):
                if s < 0.5:
                    continue
                cex += _sell_collateral(p, s, ry)
                ry = max(0.0, ry - s)
            if tot_buy + coll_l1 + cex > state.free_cash + 1e-9:
                k *= 0.88
                continue
            break

        cash_left = state.free_cash
        spend_left = spendable + 1e-6
        if bid_size >= 0.5 and buy_cost_l1 <= cash_left + 1e-9 and buy_cost_l1 <= spend_left + 1e-9:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
            cash_left -= buy_cost_l1
            spend_left -= buy_cost_l1
        for p, s in zip(ep, eb):
            if s < 0.5:
                continue
            cost = p * 0.01 * s
            if cost <= cash_left + 1e-9 and cost <= spend_left + 1e-9:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=p, quantity=s))
                cash_left -= cost
                spend_left -= cost

        rem_yes = state.yes_inventory
        if ask_size >= 0.5:
            c = _sell_collateral(my_ask, ask_size, rem_yes)
            if c <= cash_left + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
                cash_left -= c
                rem_yes = max(0.0, rem_yes - ask_size)
        for p, s in zip(ap, ea):
            if s < 0.5:
                continue
            c = _sell_collateral(p, s, rem_yes)
            if c <= cash_left + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=p, quantity=s))
                cash_left -= c
                rem_yes = max(0.0, rem_yes - s)

        return actions
