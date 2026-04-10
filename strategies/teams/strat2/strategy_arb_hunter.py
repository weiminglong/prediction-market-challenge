"""Arb-hunter v11: Multilevel (no model-boost) + stronger arb defense
on L1 and L2, with enhanced shock detection."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_ema = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.quiet_steps = 0
        self.prev_move = 0.0

    def _convex_inv_shift(self, net_inv):
        if net_inv == 0.0:
            return 0.0
        sig = 1.0 if net_inv > 0 else -1.0
        max_inv = 8.7
        a = min(abs(net_inv), max_inv * 1.4)
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
        move = 0.0
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_ema = 0.935 * self.vol_ema + 0.065 * abs(move)
            self.trend = 0.654 * self.trend + 0.173 * move

            shock_trigger = max(1.85, 3.05 * max(self.vol_ema, 0.355))
            if abs(move) >= shock_trigger:
                self.shock_remaining = 4
                self.shock_sign = 1 if move > 0 else -1

            # Early shock: consecutive same-direction accelerating moves
            if (move * self.prev_move > 0 and abs(move) > 1.2 and
                abs(move) > abs(self.prev_move) + 0.3):
                self.shock_remaining = max(self.shock_remaining, 3)
                self.shock_sign = 1 if move > 0 else -1

            self.prev_move = move

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
        if not arb_like and buy_qty + sell_qty < 0.5:
            self.quiet_steps += 1
        else:
            self.quiet_steps = 0

        net_inv = state.yes_inventory - state.no_inventory
        max_inv = 8.7
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend * 1.0

        # Spread
        half_spread = 2
        if self.vol_ema > 1.23:
            half_spread += 1
        if self.shock_remaining > 0:
            half_spread += 2

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        # Sizing
        vol_scale = max(0.064, 1.0 - self.vol_ema * 2.4)
        if self.shock_remaining > 0:
            vol_scale *= 0.15

        base_size = 16.0
        min_size = 0.55
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        if net_inv > 5.5:
            bid_size *= 0.94
        elif net_inv < -5.5:
            ask_size *= 0.94

        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(min_size, avail_yes + max(0.0, uncovered * 0.88))

        # === ARB DEFENSE on L1: stronger than v9 ===
        direction = self.trend + self.fill_bias * 0.3
        dir_abs = abs(direction)
        if dir_abs > 0.12:
            # Stronger: dir_abs=0.12->0.87, dir_abs=0.3->0.67, dir_abs=0.5->0.45
            danger_mult = max(0.25, 1.0 - dir_abs * 1.1)
            if direction > 0:
                ask_size *= danger_mult
            else:
                bid_size *= danger_mult

        # Cash
        vol_ref = max(self.vol_ema, 0.064)
        spendable = max(0.0, state.free_cash - 5.0 * vol_ref)

        # Shock: suppress dangerous side
        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        # L1
        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            bid_size = max(min_size, bid_size * spendable / max(buy_cost, 1e-12)) if bid_size * spendable / max(buy_cost, 1e-12) >= min_size else 0.0
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
                ask_size = max(min_size, new_ask) if new_ask >= min_size - 1e-9 else 0.0

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

        # L2 with arb defense
        l2_extra = 5
        l2_frac = 0.5
        bid_L2 = max(1, my_bid - l2_extra)
        ask_L2 = min(99, my_ask + l2_extra)
        bid_sz2 = max(min_size, bid_size * l2_frac) if bid_size > 0 else 0.0
        ask_sz2 = max(min_size, ask_size * l2_frac) if ask_size > 0 else 0.0

        if dir_abs > 0.20:
            l2_danger = max(0.45, 1.0 - dir_abs * 0.7)
            if direction > 0:
                ask_sz2 *= l2_danger
            else:
                bid_sz2 *= l2_danger

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

        # L3
        l3_extra = 9
        l3_frac = 0.3
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

        # L4
        l4_extra = 14
        l4_frac = 0.15
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

        # Sentinels in calm periods
        comp_spread = ask - bid
        if comp_spread >= 3 and self.quiet_steps >= 2 and self.shock_remaining <= 0:
            sent_sz = round(3.5 * vol_scale, 2)
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
