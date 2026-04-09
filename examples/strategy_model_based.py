"""Model-based optimal quoting with jump-diffusion analytics.

Builds on proven round4 design (tight spread, large size, trend+shock controls)
with model-based enhancements:
1. Fill-aware shock detection (large fills reveal arb sweep = prob moved)
2. Convex inventory management with Hermite spline smoothing
3. Trend following and volatility EMA
4. Multi-level quoting for wide spreads during calm periods
"""

from __future__ import annotations

import math

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import Action, CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """Model-based quoting with trend, shock, and multi-level extensions."""

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.trend = 0.0
        self.shock_remaining = 0
        self.shock_sign = 0

    def _convex_inv_shift(self, net_inv: float) -> float:
        """Hermite-spline inventory skew: smooth below soft cap, cubic tail above."""
        if net_inv == 0.0:
            return 0.0
        sig = 1.0 if net_inv > 0 else -1.0
        a = min(abs(net_inv), 12.0)
        skew = 0.028
        soft = 4.6
        edge_mult = 0.016
        if a <= soft:
            t = a / soft
            w = t * t * (3.0 - 2.0 * t)
            return -sig * skew * a * w
        excess = a - soft
        base = skew * soft
        tail = skew * excess * (1.0 + edge_mult * excess * excess)
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
        spread = ask - bid

        # --- Volatility and trend tracking ---
        move = 0.0
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_estimate = 0.935 * self.vol_estimate + 0.065 * abs(move)
            self.trend = 0.654 * self.trend + 0.173 * move

            # Shock detection: large mid moves indicate prob jump
            shock_trigger = max(1.85, 3.05 * max(self.vol_estimate, 0.355))
            if abs(move) >= shock_trigger:
                self.shock_remaining = 4
                self.shock_sign = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        # --- Fill bias tracking ---
        if state.buy_filled_quantity > 0:
            self.fill_bias -= 0.498
        if state.sell_filled_quantity > 0:
            self.fill_bias += 0.498
        self.fill_bias *= 0.590

        # Enhanced shock: large fills indicate arb sweep
        if state.buy_filled_quantity > 8.0 or state.sell_filled_quantity > 8.0:
            if self.shock_remaining < 2:
                self.shock_remaining = 2
                if state.buy_filled_quantity > state.sell_filled_quantity:
                    self.shock_sign = 1
                else:
                    self.shock_sign = -1

        # --- Fair value ---
        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend * 0.997

        # --- Spread calculation ---
        half_spread = 2
        if self.vol_estimate > 1.23:
            half_spread += 1
        if self.shock_remaining > 0:
            half_spread += 2

        my_bid = max(1, int(round(fair - half_spread)))
        my_ask = min(99, int(round(fair + half_spread)))
        if my_bid >= my_ask:
            return actions

        # --- Size calculation ---
        vol_scale = max(0.064, 1.0 - self.vol_estimate * 2.40)
        if self.shock_remaining > 0:
            vol_scale *= 0.15

        base_size = 14.1
        max_inv = 8.67
        min_size = 0.55

        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        # Side damping
        if net_inv > 5.5:
            bid_size *= 0.94
        elif net_inv < -5.5:
            ask_size *= 0.94

        # Uncovered sell penalty
        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            ask_size = max(min_size, avail_yes + uncovered * 0.88)

        # Cash reserve
        vol_ref = max(self.vol_estimate, 0.064)
        reserve = 5.0 * vol_ref
        spendable = max(0.0, state.free_cash - reserve)

        # Shock: suppress the losing side
        if self.shock_remaining > 0:
            if self.shock_sign < 0:
                bid_size = 0.0
            elif self.shock_sign > 0:
                ask_size = 0.0
            self.shock_remaining -= 1

        # --- Cash constraints ---
        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            scale = spendable / max(buy_cost, 1e-12)
            bid_size = bid_size * scale
            if bid_size < min_size:
                bid_size = 0.0
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
                if new_ask < min_size:
                    ask_size = 0.0
                else:
                    ask_size = max(min_size, new_ask)

        # --- Place primary orders ---
        if bid_size > 0.0 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        # --- Multi-level extension for wide spreads ---
        if spread >= 6 and self.shock_remaining <= 0 and vol_scale > 0.3:
            ext_size = base_size * vol_scale * 0.35
            ext_bid = max(1, my_bid - 1)
            ext_ask = min(99, my_ask + 1)

            if ext_bid >= 1:
                ext_bid_size = max(min_size, ext_size * max(0.0, 1.0 - net_inv / max_inv))
                ext_cost = ext_bid * 0.01 * ext_bid_size
                remaining = state.free_cash - buy_cost
                if ext_bid_size >= min_size and ext_cost <= remaining * 0.3:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=ext_bid, quantity=ext_bid_size))

            if ext_ask <= 99:
                ext_ask_size = max(min_size, ext_size * max(0.0, 1.0 + net_inv / max_inv))
                if ext_ask_size >= min_size:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=ext_ask, quantity=ext_ask_size))

        return actions
