"""I1: multilevel quoting + anti-arb toxicity (bursts + one-sided adverse fills).

Same core as strat2_team/strategy_multilevel; adds decaying bid/ask toxicity scores,
inventory-weighted effective scores, optional L1 soft-pull, and L2–L4 + sentinel
suppression on the toxic side. Scaling applies only when scale != 1.0 so a disabled
config matches the baseline numerically.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    PARAMS = {
        "base_size": 16.0,
        "fill_decay": 0.59,
        "fill_hit": 0.5,
        "inv_skew": 0.028,
        "inv_soft": 4.6,
        "inv_edge_mult": 0.016,
        "max_inventory": 10.0,
        "min_size": 0.55,
        "side_damp_soft": 5.5,
        "damp_bid_when_long": 0.94,
        "damp_ask_when_short": 0.94,
        "uncovered_penalty": 0.12,
        "shock_duration": 4,
        "shock_size_mult": 0.15,
        "shock_trigger_min": 1.85,
        "shock_trigger_vol_mult": 3.05,
        "shock_vol_floor": 0.355,
        "spread_base": 2,
        "spread_shock_extra": 2,
        "spread_vol_extra": 1,
        "spread_vol_threshold": 1.23,
        "trend_alpha": 0.173,
        "trend_decay": 0.654,
        "trend_weight": 1.0,
        "vol_coeff": 2.4,
        "vol_decay": 0.935,
        "vol_floor": 0.064,
        "reserve_cash_base": 0.0,
        "reserve_cash_vol": 5.0,
        # Multi-level params
        "l2_spread_extra": 5,
        "l2_size_frac": 0.5,
        "l3_spread_extra": 9,
        "l3_size_frac": 0.3,
        "l4_spread_extra": 14,
        "l4_size_frac": 0.15,
        "sentinel_size": 3.5,
        "sentinel_min_spread": 3,
        "sentinel_quiet_steps": 2,
        # Toxicity overlay
        "tox_decay": 0.88,
        "tox_inv_weight": 0.26,
        "tox_burst_move_min": 0.52,
        "tox_burst_count": 4,
        "tox_burst_hit": 0.42,
        "tox_onesided_min": 1.08,
        "tox_onesided_other_max": 0.34,
        "tox_fill_move_eps": 0.36,
        "tox_onesided_hit": 0.58,
        "tox_l1_scale_k": 0.0,
        "tox_l1_ref_eff": 1.9,
        "tox_hard_eff": 99.0,
        "tox_deep_eff": 1.22,
        "tox_l1_pull_eff": 1.88,
        "tox_l1_pull_mult": 0.78,
    }

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.quiet_steps = 0
        self.tox_bid = 0.0
        self.tox_ask = 0.0
        self._burst_dir = 0
        self._burst_len = 0

    def _convex_inv_shift(self, net_inv):
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

    def _tox_side_scale(self, eff: float) -> float:
        p = self.PARAMS
        if eff >= float(p["tox_hard_eff"]):
            return 0.0
        k = float(p["tox_l1_scale_k"])
        ref = max(float(p["tox_l1_ref_eff"]), 1e-6)
        s = max(0.0, 1.0 - k * min(1.0, eff / ref))
        pull_thr = float(p["tox_l1_pull_eff"])
        if eff >= pull_thr:
            s *= float(p["tox_l1_pull_mult"])
        return s

    def _tox_accumulate(self, p, had_prev: bool, move: float, buy_qty: float, sell_qty: float) -> None:
        if not had_prev:
            return
        d_tox = float(p["tox_decay"])
        self.tox_bid *= d_tox
        self.tox_ask *= d_tox

        if self._burst_len >= int(p["tox_burst_count"]):
            bh = float(p["tox_burst_hit"])
            if self._burst_dir < 0:
                self.tox_bid += bh
            elif self._burst_dir > 0:
                self.tox_ask += bh

        omin = float(p["tox_onesided_min"])
        oother = float(p["tox_onesided_other_max"])
        feps = float(p["tox_fill_move_eps"])
        oh = float(p["tox_onesided_hit"])
        if buy_qty >= omin and sell_qty <= oother and move <= -feps:
            self.tox_bid += oh
        if sell_qty >= omin and buy_qty <= oother and move >= feps:
            self.tox_ask += oh

    def _tox_quoting_effects(self, p, net_inv: float):
        max_inv = max(float(p["max_inventory"]), 1e-9)
        iw = float(p["tox_inv_weight"])
        eff_bid = self.tox_bid * (1.0 + iw * max(0.0, min(1.0, net_inv / max_inv)))
        eff_ask = self.tox_ask * (1.0 + iw * max(0.0, min(1.0, -net_inv / max_inv)))
        deep_thr = float(p["tox_deep_eff"])
        allow_deep_bid = eff_bid < deep_thr
        allow_deep_ask = eff_ask < deep_thr
        scale_bid = self._tox_side_scale(eff_bid)
        scale_ask = self._tox_side_scale(eff_ask)
        return allow_deep_bid, allow_deep_ask, scale_bid, scale_ask

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

            self.vol_estimate = p["vol_decay"] * self.vol_estimate + (1.0 - p["vol_decay"]) * abs(move)
            self.trend = p["trend_decay"] * self.trend + p["trend_alpha"] * move

            shock_trigger = max(
                p["shock_trigger_min"],
                p["shock_trigger_vol_mult"] * max(self.vol_estimate, p["shock_vol_floor"]),
            )
            if abs(move) >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = 1 if move > 0 else -1

            bm = float(p["tox_burst_move_min"])
            if abs(move) >= bm:
                d = 1 if move > 0 else -1
                if d == self._burst_dir:
                    self._burst_len += 1
                else:
                    self._burst_dir = d
                    self._burst_len = 1
            else:
                self._burst_len = 0
                self._burst_dir = 0

        self.prev_bid = bid
        self.prev_ask = ask

        # Fill-reactive bias
        buy_qty = state.buy_filled_quantity
        sell_qty = state.sell_filled_quantity
        total_fills = buy_qty + sell_qty

        if buy_qty > 0:
            self.fill_bias -= p["fill_hit"]
        if sell_qty > 0:
            self.fill_bias += p["fill_hit"]
        self.fill_bias *= p["fill_decay"]

        # Track quiet periods
        arb_like = (buy_qty > 1.0 and sell_qty < 0.5) or (sell_qty > 1.0 and buy_qty < 0.5)
        if not arb_like and total_fills < 0.5:
            self.quiet_steps += 1
        else:
            self.quiet_steps = 0

        net_inv = state.yes_inventory - state.no_inventory
        self._tox_accumulate(p, had_prev, move, buy_qty, sell_qty)

        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend * p["trend_weight"]

        # Spread
        half_spread = int(p["spread_base"])
        if self.vol_estimate > p["spread_vol_threshold"]:
            half_spread += int(p["spread_vol_extra"])
        if self.shock_remaining > 0:
            half_spread += int(p["spread_shock_extra"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        allow_deep_bid, allow_deep_ask, scale_bid, scale_ask = self._tox_quoting_effects(p, net_inv)

        # Sizing
        vol_scale = max(p["vol_floor"], 1.0 - self.vol_estimate * p["vol_coeff"])
        if self.shock_remaining > 0:
            vol_scale *= p["shock_size_mult"]

        base_size = p["base_size"]
        max_inv = max(p["max_inventory"], 1e-9)
        min_size = p["min_size"]
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        # Side damping
        if net_inv > p["side_damp_soft"]:
            bid_size *= p["damp_bid_when_long"]
        elif net_inv < -p["side_damp_soft"]:
            ask_size *= p["damp_ask_when_short"]

        # Uncovered sell penalty
        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(min_size, avail_yes + max(0.0, uncovered * (1.0 - p["uncovered_penalty"])))

        # Cash reserve
        vol_ref = max(self.vol_estimate, p["vol_floor"])
        reserve_need = p["reserve_cash_base"] + p["reserve_cash_vol"] * vol_ref
        spendable = max(0.0, state.free_cash - reserve_need)

        # Shock: suppress losing side
        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        if scale_bid != 1.0:
            bid_size *= scale_bid
            if bid_size > 0.0 and bid_size < min_size:
                bid_size = 0.0
        if scale_ask != 1.0:
            ask_size *= scale_ask
            if ask_size > 0.0 and ask_size < min_size:
                ask_size = 0.0

        # Cash feasibility for bids
        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            scale = spendable / max(buy_cost, 1e-12)
            bid_size = bid_size * scale
            if bid_size < min_size:
                bid_size = 0.0
            else:
                bid_size = max(min_size, bid_size)
            buy_cost = my_bid * 0.01 * bid_size

        # Cash feasibility for asks
        free_after_bid = state.free_cash - buy_cost
        ask_px = my_ask * 0.01
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

        # L1 orders
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

        # L2 orders: wider spread, smaller size
        # During shock, only place orders on the safe side
        l2_extra = int(p["l2_spread_extra"])
        l2_frac = p["l2_size_frac"]
        bid_L2 = max(1, my_bid - l2_extra)
        ask_L2 = min(99, my_ask + l2_extra)
        bid_sz2 = max(min_size, bid_size * l2_frac) if bid_size > 0 and allow_deep_bid else 0.0
        ask_sz2 = max(min_size, ask_size * l2_frac) if ask_size > 0 and allow_deep_ask else 0.0

        if bid_sz2 > 0:
            cost2 = bid_L2 * 0.01 * bid_sz2
            if cost2 <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_L2, quantity=bid_sz2))
                spendable -= cost2

        if ask_sz2 > 0:
            cov2 = min(ask_sz2, avail_yes)
            unc2 = max(0.0, ask_sz2 - cov2)
            coll2 = (1.0 - ask_L2 * 0.01) * unc2
            if coll2 <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_L2, quantity=ask_sz2))
                avail_yes = max(0.0, avail_yes - ask_sz2)
                if coll2 > 0:
                    spendable -= coll2

        # L3 orders: even wider, very small
        l3_extra = int(p["l3_spread_extra"])
        l3_frac = p["l3_size_frac"]
        bid_L3 = max(1, my_bid - l3_extra)
        ask_L3 = min(99, my_ask + l3_extra)
        bid_sz3 = max(min_size, bid_size * l3_frac) if bid_size > 0 and allow_deep_bid else 0.0
        ask_sz3 = max(min_size, ask_size * l3_frac) if ask_size > 0 and allow_deep_ask else 0.0

        if bid_sz3 > 0:
            cost3 = bid_L3 * 0.01 * bid_sz3
            if cost3 <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_L3, quantity=bid_sz3))
                spendable -= cost3

        if ask_sz3 > 0:
            cov3 = min(ask_sz3, avail_yes)
            unc3 = max(0.0, ask_sz3 - cov3)
            coll3 = (1.0 - ask_L3 * 0.01) * unc3
            if coll3 <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_L3, quantity=ask_sz3))
                avail_yes = max(0.0, avail_yes - ask_sz3)
                if coll3 > 0:
                    spendable -= coll3

        # L4 orders: widest level
        l4_extra = int(p["l4_spread_extra"])
        l4_frac = p["l4_size_frac"]
        bid_L4 = max(1, my_bid - l4_extra)
        ask_L4 = min(99, my_ask + l4_extra)
        bid_sz4 = max(min_size, bid_size * l4_frac) if bid_size > 0 and allow_deep_bid else 0.0
        ask_sz4 = max(min_size, ask_size * l4_frac) if ask_size > 0 and allow_deep_ask else 0.0

        if bid_sz4 > 0:
            cost4 = bid_L4 * 0.01 * bid_sz4
            if cost4 <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_L4, quantity=bid_sz4))
                spendable -= cost4

        if ask_sz4 > 0:
            cov4 = min(ask_sz4, avail_yes)
            unc4 = max(0.0, ask_sz4 - cov4)
            coll4 = (1.0 - ask_L4 * 0.01) * unc4
            if coll4 <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_L4, quantity=ask_sz4))
                avail_yes = max(0.0, avail_yes - ask_sz4)
                if coll4 > 0:
                    spendable -= coll4

        # Sentinel orders inside competitor spread during calm periods
        comp_spread = ask - bid
        if (comp_spread >= int(p["sentinel_min_spread"])
                and self.quiet_steps >= int(p["sentinel_quiet_steps"])
                and self.shock_remaining <= 0):
            sent_sz = round(p["sentinel_size"] * vol_scale, 2)
            if allow_deep_bid:
                tick_b = max(1, bid + 1)
                sz_b = round(max(0.1, sent_sz * max(0.0, 1.0 - net_inv / max_inv)), 2)
                cost_b = tick_b * 0.01 * sz_b
                if sz_b > 0.1 and cost_b <= spendable:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=tick_b, quantity=sz_b))
                    spendable -= cost_b
            if allow_deep_ask:
                tick_a = min(99, ask - 1)
                sz_a = round(max(0.1, sent_sz * max(0.0, 1.0 + net_inv / max_inv)), 2)
                cov_s = min(sz_a, avail_yes)
                unc_s = max(0.0, sz_a - cov_s)
                coll_s = (1.0 - tick_a * 0.01) * unc_s
                if sz_a > 0.1 and coll_s <= spendable + 1e-9:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=tick_a, quantity=sz_a))

        return actions
