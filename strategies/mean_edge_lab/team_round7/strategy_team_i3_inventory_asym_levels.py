"""Round 7 I3: multilevel quotes with inventory-aware asymmetry by depth (L2–L4).

Deeper ladder levels tilt harder toward inventory-reducing sides as |net_inv| grows:
- size fractions scale per level (long → more ask / less bid on deeper rungs)
- spread offsets widen the adding-inventory side and tighten the reducing side on deeper rungs

Feasibility paths match strategy_multilevel (cash, collateral, min_size, shock gating).
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
        "l2_spread_extra": 5,
        "l2_size_frac": 0.5,
        "l3_spread_extra": 9,
        "l3_size_frac": 0.3,
        "l4_spread_extra": 14,
        "l4_size_frac": 0.15,
        "sentinel_size": 3.5,
        "sentinel_min_spread": 3,
        "sentinel_quiet_steps": 2,
        # --- I3: inventory × depth asymmetry (level 2,3,4 index 0,1,2) ---
        "inv_level_asym_strength": 0.17,
        "inv_level_depth_step": 0.42,
        "inv_level_frac_lo": 0.38,
        "inv_level_frac_hi": 1.42,
        "inv_level_spread_scale": 1.15,
        "inv_level_spread_cap": 3,
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

    def _level_inv_asym(
        self, net_inv: float, level: int
    ) -> tuple[float, float, int, int]:
        """Return (bid_frac_mult, ask_frac_mult, bid_extra_ticks, ask_extra_ticks) for L2–L4."""
        p = self.PARAMS
        max_inv = max(float(p["max_inventory"]), 1e-9)
        if abs(net_inv) < 1e-9:
            return 1.0, 1.0, 0, 0
        depth = min(abs(net_inv) / max_inv, 1.0)
        sig = 1.0 if net_inv > 0 else -1.0
        li = max(0, level - 2)
        level_scale = 1.0 + float(p["inv_level_depth_step"]) * float(li)
        w = depth * level_scale * float(p["inv_level_asym_strength"])
        lo = float(p["inv_level_frac_lo"])
        hi = float(p["inv_level_frac_hi"])
        # Long: shrink deep bids, grow deep asks.
        bid_m = max(lo, min(hi, 1.0 - sig * w))
        ask_m = max(lo, min(hi, 1.0 + sig * w))
        st = float(p["inv_level_spread_scale"]) * depth * level_scale
        dt = int(round(st))
        dt = min(dt, int(p["inv_level_spread_cap"]))
        # Long: widen bid ladder (more negative distance from mid → larger extra below my_bid).
        bid_extra = int(round(sig * float(dt)))
        # Long: tighten ask ladder (smaller extra above my_ask).
        ask_extra = int(round(-sig * float(dt)))
        return bid_m, ask_m, bid_extra, ask_extra

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

            self.vol_estimate = p["vol_decay"] * self.vol_estimate + (1.0 - p["vol_decay"]) * abs(move)
            self.trend = p["trend_decay"] * self.trend + p["trend_alpha"] * move

            shock_trigger = max(
                p["shock_trigger_min"],
                p["shock_trigger_vol_mult"] * max(self.vol_estimate, p["shock_vol_floor"]),
            )
            if abs(move) >= shock_trigger:
                self.shock_remaining = int(p["shock_duration"])
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        buy_qty = state.buy_filled_quantity
        sell_qty = state.sell_filled_quantity
        total_fills = buy_qty + sell_qty

        if buy_qty > 0:
            self.fill_bias -= p["fill_hit"]
        if sell_qty > 0:
            self.fill_bias += p["fill_hit"]
        self.fill_bias *= p["fill_decay"]

        arb_like = (buy_qty > 1.0 and sell_qty < 0.5) or (sell_qty > 1.0 and buy_qty < 0.5)
        if not arb_like and total_fills < 0.5:
            self.quiet_steps += 1
        else:
            self.quiet_steps = 0

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend * p["trend_weight"]

        half_spread = int(p["spread_base"])
        if self.vol_estimate > p["spread_vol_threshold"]:
            half_spread += int(p["spread_vol_extra"])
        if self.shock_remaining > 0:
            half_spread += int(p["spread_shock_extra"])

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(p["vol_floor"], 1.0 - self.vol_estimate * p["vol_coeff"])
        if self.shock_remaining > 0:
            vol_scale *= p["shock_size_mult"]

        base_size = p["base_size"]
        max_inv = max(p["max_inventory"], 1e-9)
        min_size = p["min_size"]
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if net_inv > p["side_damp_soft"]:
            bid_size *= p["damp_bid_when_long"]
        elif net_inv < -p["side_damp_soft"]:
            ask_size *= p["damp_ask_when_short"]

        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(min_size, avail_yes + max(0.0, uncovered * (1.0 - p["uncovered_penalty"])))

        vol_ref = max(self.vol_estimate, p["vol_floor"])
        reserve_need = p["reserve_cash_base"] + p["reserve_cash_vol"] * vol_ref
        spendable = max(0.0, state.free_cash - reserve_need)

        shock_rem = self.shock_remaining
        shock_sign = self.shock_sign
        if shock_rem > 0:
            if shock_sign < 0:
                bid_size = 0.0
            elif shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

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

        def place_deep(
            level: int,
            base_bid_extra: int,
            base_ask_extra: int,
            base_bid_frac: float,
            base_ask_frac: float,
        ) -> None:
            nonlocal spendable, avail_yes
            bm, am, bd, ad = self._level_inv_asym(net_inv, level)
            bid_extra = max(0, int(base_bid_extra) + bd)
            ask_extra = max(0, int(base_ask_extra) + ad)
            bid_L = max(1, my_bid - bid_extra)
            ask_L = min(99, my_ask + ask_extra)
            bid_sz = max(min_size, bid_size * base_bid_frac * bm) if bid_size > 0 else 0.0
            ask_sz = max(min_size, ask_size * base_ask_frac * am) if ask_size > 0 else 0.0
            if shock_rem > 0:
                if shock_sign < 0:
                    bid_sz = 0.0
                elif shock_sign > 0:
                    ask_sz = 0.0
            if bid_sz > 0:
                c = bid_L * 0.01 * bid_sz
                if c <= spendable:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_L, quantity=bid_sz))
                    spendable -= c
            if ask_sz > 0:
                cv = min(ask_sz, avail_yes)
                uc = max(0.0, ask_sz - cv)
                cl = (1.0 - ask_L * 0.01) * uc
                if cl <= spendable + 1e-9:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_L, quantity=ask_sz))
                    avail_yes = max(0.0, avail_yes - ask_sz)
                    if cl > 0:
                        spendable -= cl

        place_deep(2, int(p["l2_spread_extra"]), int(p["l2_spread_extra"]), float(p["l2_size_frac"]), float(p["l2_size_frac"]))
        place_deep(3, int(p["l3_spread_extra"]), int(p["l3_spread_extra"]), float(p["l3_size_frac"]), float(p["l3_size_frac"]))
        place_deep(4, int(p["l4_spread_extra"]), int(p["l4_spread_extra"]), float(p["l4_size_frac"]), float(p["l4_size_frac"]))

        comp_spread = ask - bid
        if (
            comp_spread >= int(p["sentinel_min_spread"])
            and self.quiet_steps >= int(p["sentinel_quiet_steps"])
            and shock_rem <= 0
        ):
            sent_sz = round(p["sentinel_size"] * vol_scale, 2)
            tick_b = max(1, bid + 1)
            sz_b = round(max(0.1, sent_sz * max(0.0, 1.0 - net_inv / max_inv)), 2)
            cost_b = tick_b * 0.01 * sz_b
            if sz_b > 0.1 and cost_b <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=tick_b, quantity=sz_b))
                spendable -= cost_b
            tick_a = min(99, ask - 1)
            sz_a = round(max(0.1, sent_sz * max(0.0, 1.0 + net_inv / max_inv)), 2)
            cov_s = min(sz_a, avail_yes)
            unc_s = max(0.0, sz_a - cov_s)
            coll_s = (1.0 - tick_a * 0.01) * unc_s
            if sz_a > 0.1 and coll_s <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=tick_a, quantity=sz_a))

        return actions
