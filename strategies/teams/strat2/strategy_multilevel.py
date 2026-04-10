"""Enhanced round4: multi-level quoting + fill-enhanced signals.

Builds on round4_best (edge +10.73) with:
1. Multi-level orders (L2 at wider spread, smaller size) for more retail capture
2. Inside-spread sentinels during calm periods with wide spreads
3. All round4 core mechanics preserved exactly
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
        "max_inventory": 8.7,
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
        "trend_weight": 1.3,
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
        bid_sz2 = max(min_size, bid_size * l2_frac) if bid_size > 0 else 0.0
        ask_sz2 = max(min_size, ask_size * l2_frac) if ask_size > 0 else 0.0

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
        bid_sz3 = max(min_size, bid_size * l3_frac) if bid_size > 0 else 0.0
        ask_sz3 = max(min_size, ask_size * l3_frac) if ask_size > 0 else 0.0

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
        bid_sz4 = max(min_size, bid_size * l4_frac) if bid_size > 0 else 0.0
        ask_sz4 = max(min_size, ask_size * l4_frac) if ask_size > 0 else 0.0

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
            # Buy sentinel 1 tick inside competitor bid
            tick_b = max(1, bid + 1)
            sz_b = round(max(0.1, sent_sz * max(0.0, 1.0 - net_inv / max_inv)), 2)
            cost_b = tick_b * 0.01 * sz_b
            if sz_b > 0.1 and cost_b <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=tick_b, quantity=sz_b))
                spendable -= cost_b
            # Sell sentinel 1 tick inside competitor ask
            tick_a = min(99, ask - 1)
            sz_a = round(max(0.1, sent_sz * max(0.0, 1.0 + net_inv / max_inv)), 2)
            cov_s = min(sz_a, avail_yes)
            unc_s = max(0.0, sz_a - cov_s)
            coll_s = (1.0 - tick_a * 0.01) * unc_s
            if sz_a > 0.1 and coll_s <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=tick_a, quantity=sz_a))

        return actions
