"""K1: persistent client_order_ids, selective CancelOrder-only updates, breakout lean."""
from __future__ import annotations

import math
from typing import Sequence

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import Action, CancelOrder, PlaceOrder, Side, StepState

# Stable slots — no CancelAll; engine frees id after cancel for same-step replace.
OID_B_IN = "k1-b-in"
OID_B_OUT = "k1-b-out"
OID_S_IN = "k1-s-in"
OID_S_OUT = "k1-s-out"


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


class Strategy(BaseStrategy):
    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_ema = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0

    @staticmethod
    def _inv_skew(net_inv: float) -> float:
        if net_inv == 0.0:
            return 0.0
        sign = 1.0 if net_inv > 0 else -1.0
        a = min(abs(net_inv), 12.0)
        soft, skew = 4.6, 0.028
        edge_mult = 0.016
        if a <= soft:
            t = a / soft
            w = t * t * (3.0 - 2.0 * t)
            return -sign * skew * a * w
        excess = a - soft
        return -sign * (skew * soft + skew * excess * (1.0 + edge_mult * excess * excess))

    def _needs_replace(
        self,
        cur_px: int | None,
        cur_rem: float | None,
        want_px: int,
        want_qty: float,
    ) -> bool:
        if cur_px is None:
            return True
        if abs(cur_px - want_px) >= 1:
            return True
        if cur_rem is None or cur_rem <= 0.0:
            return True
        if want_qty <= 0.0:
            return cur_rem > 1e-9
        rel = abs(cur_rem - want_qty) / max(want_qty, cur_rem, 0.05)
        return rel > 0.18 or abs(cur_rem - want_qty) > 0.85

    def _sync_slot(
        self,
        actions: list[Action],
        own_by_id: dict[str, tuple[int, float]],
        oid: str,
        want_px: int | None,
        want_qty: float,
    ) -> None:
        cur = own_by_id.get(oid)
        cur_px = cur[0] if cur else None
        cur_rem = cur[1] if cur else None

        if want_qty <= 0.0 or want_px is None:
            if oid in own_by_id:
                actions.append(CancelOrder(order_id=oid))
            return

        if not self._needs_replace(cur_px, cur_rem, want_px, want_qty):
            return

        if oid in own_by_id:
            actions.append(CancelOrder(order_id=oid))
        actions.append(
            PlaceOrder(
                side=Side.BUY if oid.startswith("k1-b") else Side.SELL,
                price_ticks=want_px,
                quantity=want_qty,
                client_order_id=oid,
            )
        )

    def on_step(self, state: StepState) -> Sequence[Action]:
        actions: list[Action] = []
        bid_t = state.competitor_best_bid_ticks
        ask_t = state.competitor_best_ask_ticks
        if bid_t is None and ask_t is None:
            return actions
        if bid_t is None:
            bid_t = max(1, ask_t - 6)  # type: ignore[operator]
        if ask_t is None:
            ask_t = min(99, bid_t + 6)  # type: ignore[operator]

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

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self._inv_skew(net_inv) + self.trend

        # Light breakout tilt: nudge fair with trend×vol (keeps structure vs combined’s pure stack).
        mom = max(-1.0, min(1.0, self.trend / max(self.vol_ema, 0.18)))
        fair += 0.18 * mom * min(self.vol_ema, 1.6)

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

        my_bid_in = max(1, int(round(fair - half_spread)))
        my_ask_in = min(99, int(round(fair + half_spread)))
        outer_extra = 2
        my_bid_out = max(1, my_bid_in - outer_extra)
        my_ask_out = min(99, my_ask_in + outer_extra)

        if my_bid_in >= my_ask_in:
            for oid in (OID_B_IN, OID_B_OUT, OID_S_IN, OID_S_OUT):
                if any(o.order_id == oid for o in state.own_orders):
                    actions.append(CancelOrder(order_id=oid))
            return actions

        model_boost = max(0.5, min(3.0, 1.0 / max(0.3, model_vol)))
        base_size = 10.0 * model_boost
        vol_scale = max(0.06, 1.0 - self.vol_ema * 2.4)
        if self.shock_remaining > 0:
            vol_scale *= 0.15

        max_inv = 8.7
        bid_size_in = max(0.5, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size_in = max(0.5, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))
        # Outer quantities stay at 0: four stable IDs + selective sync still cancel stale OUT legs.
        bid_size_out = 0.0
        ask_size_out = 0.0

        if net_inv > 5.5:
            bid_size_in *= 0.94
            bid_size_out *= 0.94
        elif net_inv < -5.5:
            ask_size_in *= 0.94
            ask_size_out *= 0.94

        avail_yes = max(0.0, state.yes_inventory)

        def cap_ask(sz: float, px: int) -> float:
            if sz <= 0.0:
                return 0.0
            if sz > avail_yes:
                uncovered = sz - avail_yes
                sz = max(0.5, avail_yes + uncovered * 0.88)
            return sz

        ask_size_in = cap_ask(ask_size_in, my_ask_in)
        ask_size_out = cap_ask(ask_size_out, my_ask_out)

        vol_ref = max(self.vol_ema, 0.06)
        spendable = max(0.0, state.free_cash - 5.0 * vol_ref)

        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size_in = 0.0
                bid_size_out = 0.0
            elif self.shock_sign > 0:
                ask_size_in = 0.0
                ask_size_out = 0.0
            self.shock_remaining -= 1

        def cap_bid_budget(sz: float, px: int, budget: float) -> tuple[float, float]:
            if sz <= 0.0 or budget <= 0.0:
                return 0.0, budget
            cost = px * 0.01 * sz
            if cost > budget:
                sz *= budget / max(cost, 1e-12)
            if sz < 0.5:
                return 0.0, budget
            spent = px * 0.01 * sz
            return sz, max(0.0, budget - spent)

        bid_size_in, b_rem = cap_bid_budget(bid_size_in, my_bid_in, spendable)
        bid_size_out, _ = cap_bid_budget(bid_size_out, my_bid_out, b_rem)

        buy_cost_in = my_bid_in * 0.01 * bid_size_in if bid_size_in > 0.0 else 0.0
        buy_cost_out = my_bid_out * 0.01 * bid_size_out if bid_size_out > 0.0 else 0.0
        cash_left = max(0.0, state.free_cash - buy_cost_in - buy_cost_out)
        av_yes = avail_yes

        def fit_sell(sz: float, px: int) -> float:
            nonlocal cash_left, av_yes
            if sz <= 0.0 or cash_left <= 0.0:
                return 0.0
            one_m = max(1e-9, 1.0 - px * 0.01)
            cov = min(sz, av_yes)
            unc = max(0.0, sz - cov)
            need = one_m * unc
            if need > cash_left + 1e-9:
                unc = max(0.0, cash_left / one_m)
                sz = cov + unc
                if sz < 0.5:
                    return 0.0
                cov = min(sz, av_yes)
                unc = max(0.0, sz - cov)
                need = one_m * unc
            cash_left = max(0.0, cash_left - need)
            av_yes = max(0.0, av_yes - min(sz, av_yes))
            return sz

        ask_size_in = fit_sell(ask_size_in, my_ask_in)
        ask_size_out = fit_sell(ask_size_out, my_ask_out)

        if bid_size_in > 0.0 and my_bid_in * 0.01 * bid_size_in > state.free_cash + 1e-6:
            bid_size_in = 0.0
        if bid_size_out > 0.0 and my_bid_out * 0.01 * bid_size_out > state.free_cash + 1e-6:
            bid_size_out = 0.0

        own_by_id = {o.order_id: (o.price_ticks, o.remaining_quantity) for o in state.own_orders}

        # Outer level disabled if it would cross inner on the same side ladder (keep book sane).
        b_out_px = my_bid_out if my_bid_out < my_bid_in else None
        b_out_sz = bid_size_out if b_out_px is not None else 0.0
        a_out_px = my_ask_out if my_ask_out > my_ask_in else None
        a_out_sz = ask_size_out if a_out_px is not None else 0.0

        self._sync_slot(actions, own_by_id, OID_B_IN, my_bid_in, bid_size_in)
        self._sync_slot(actions, own_by_id, OID_B_OUT, b_out_px, b_out_sz)
        self._sync_slot(actions, own_by_id, OID_S_IN, my_ask_in, ask_size_in)
        self._sync_slot(actions, own_by_id, OID_S_OUT, a_out_px, a_out_sz)

        return actions
