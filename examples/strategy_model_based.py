"""Model-based optimal quoting with jump-diffusion analytics.

Combines proven techniques from round4, multilevel, and arb-hunter:
1. Tight 2-tick L1 spread with trend+shock risk controls
2. 4-level quoting ladder (L1-L4) for deep retail capture
3. Trend-based arb defense: reduce dangerous side on L1
4. Inside-spread sentinels during calm periods
5. Fill-aware shock detection (large fills reveal arb sweeps)
6. Convex inventory management with Hermite spline smoothing
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import Action, CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0
        self.quiet_steps = 0

    def _convex_inv_shift(self, net_inv: float) -> float:
        if net_inv == 0.0:
            return 0.0
        sig = 1.0 if net_inv > 0 else -1.0
        a = min(abs(net_inv), 12.2)
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

    def on_step(self, state: StepState) -> list[Action]:
        actions: list[Action] = [CancelAll()]

        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks
        if bid is None and ask is None:
            return actions
        if bid is None:
            bid = max(1, ask - 6)
        if ask is None:
            ask = min(99, bid + 6)

        mid = (bid + ask) / 2.0

        # --- Volatility and trend tracking ---
        move = 0.0
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_estimate = 0.935 * self.vol_estimate + 0.065 * abs(move)
            self.trend = 0.654 * self.trend + 0.173 * move

            shock_trigger = max(1.85, 3.05 * max(self.vol_estimate, 0.355))
            if abs(move) >= shock_trigger:
                self.shock_remaining = 4
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        # --- Fill tracking ---
        buy_qty = state.buy_filled_quantity
        sell_qty = state.sell_filled_quantity

        if buy_qty > 0:
            self.fill_bias -= 0.5
        if sell_qty > 0:
            self.fill_bias += 0.5
        self.fill_bias *= 0.59

        # Fill-based shock detection
        if buy_qty > 8.0 or sell_qty > 8.0:
            if self.shock_remaining < 2:
                self.shock_remaining = 2
                self.shock_sign = 1 if buy_qty > sell_qty else -1

        # Track quiet periods (no fills, no arb-like activity)
        arb_like = (buy_qty > 1.0 and sell_qty < 0.5) or (sell_qty > 1.0 and buy_qty < 0.5)
        if not arb_like and buy_qty + sell_qty < 0.5:
            self.quiet_steps += 1
        else:
            self.quiet_steps = 0

        # --- Fair value ---
        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend * 1.0

        # --- L1 Spread ---
        half_spread = 2
        if self.vol_estimate > 1.23:
            half_spread += 1
        if self.shock_remaining > 0:
            half_spread += 2

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        # --- L1 Sizing ---
        vol_scale = max(0.064, 1.0 - self.vol_estimate * 2.4)
        if self.shock_remaining > 0:
            vol_scale *= 0.15

        base_size = 16.0
        max_inv = 8.7
        min_size = 0.55

        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        # Side damping
        if net_inv > 5.5:
            bid_size *= 0.94
        elif net_inv < -5.5:
            ask_size *= 0.94

        # Trend-based arb defense on L1
        direction = self.trend + self.fill_bias * 0.25
        dir_abs = abs(direction)
        if dir_abs > 0.15:
            danger_mult = max(0.35, 1.0 - dir_abs * 0.9)
            if direction > 0:
                ask_size *= danger_mult
            else:
                bid_size *= danger_mult

        # Uncovered sell penalty
        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(min_size, avail_yes + max(0.0, uncovered * 0.88))

        # Cash reserve
        vol_ref = max(self.vol_estimate, 0.064)
        reserve = 5.0 * vol_ref
        spendable = max(0.0, state.free_cash - reserve)

        # Shock: suppress losing side
        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        # --- L1 Cash constraints ---
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

        # --- Place L1 orders ---
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

        # --- L2: wider spread, smaller size ---
        l2_bid = max(1, my_bid - 5)
        l2_ask = min(99, my_ask + 5)
        l2_bid_sz = max(min_size, bid_size * 0.5) if bid_size > 0 else 0.0
        l2_ask_sz = max(min_size, ask_size * 0.5) if ask_size > 0 else 0.0

        if l2_bid_sz > 0:
            cost2 = l2_bid * 0.01 * l2_bid_sz
            if cost2 <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=l2_bid, quantity=l2_bid_sz))
                spendable -= cost2
        if l2_ask_sz > 0:
            cov2 = min(l2_ask_sz, avail_yes)
            unc2 = max(0.0, l2_ask_sz - cov2)
            coll2 = (1.0 - l2_ask * 0.01) * unc2
            if coll2 <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=l2_ask, quantity=l2_ask_sz))
                avail_yes = max(0.0, avail_yes - l2_ask_sz)
                if coll2 > 0:
                    spendable -= coll2

        # --- L3: even wider ---
        l3_bid = max(1, my_bid - 9)
        l3_ask = min(99, my_ask + 9)
        l3_bid_sz = max(min_size, bid_size * 0.3) if bid_size > 0 else 0.0
        l3_ask_sz = max(min_size, ask_size * 0.3) if ask_size > 0 else 0.0

        if l3_bid_sz > 0:
            cost3 = l3_bid * 0.01 * l3_bid_sz
            if cost3 <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=l3_bid, quantity=l3_bid_sz))
                spendable -= cost3
        if l3_ask_sz > 0:
            cov3 = min(l3_ask_sz, avail_yes)
            unc3 = max(0.0, l3_ask_sz - cov3)
            coll3 = (1.0 - l3_ask * 0.01) * unc3
            if coll3 <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=l3_ask, quantity=l3_ask_sz))
                avail_yes = max(0.0, avail_yes - l3_ask_sz)
                if coll3 > 0:
                    spendable -= coll3

        # --- L4: widest ---
        l4_bid = max(1, my_bid - 14)
        l4_ask = min(99, my_ask + 14)
        l4_bid_sz = max(min_size, bid_size * 0.15) if bid_size > 0 else 0.0
        l4_ask_sz = max(min_size, ask_size * 0.15) if ask_size > 0 else 0.0

        if l4_bid_sz > 0:
            cost4 = l4_bid * 0.01 * l4_bid_sz
            if cost4 <= spendable:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=l4_bid, quantity=l4_bid_sz))
                spendable -= cost4
        if l4_ask_sz > 0:
            cov4 = min(l4_ask_sz, avail_yes)
            unc4 = max(0.0, l4_ask_sz - cov4)
            coll4 = (1.0 - l4_ask * 0.01) * unc4
            if coll4 <= spendable + 1e-9:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=l4_ask, quantity=l4_ask_sz))
                avail_yes = max(0.0, avail_yes - l4_ask_sz)
                if coll4 > 0:
                    spendable -= coll4

        # --- Sentinel orders inside competitor spread during calm periods ---
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
