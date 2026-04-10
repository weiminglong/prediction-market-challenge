"""Team C: v23 core + two-rung ladder (near ±2, far ±4) with dual-tier size split.

Distinct from single-level hunt_arch_best_v23: same fair/vol/fill-bias/inventory scaling,
but resting size is split across an inner and outer bid/ask to capture different flow
depths without vol-based spread widening (matches v23 docstring).
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _sell_reserved_cash(*, ask_tick: int, ask_size: float, yes_inventory: float) -> float:
    price = ask_tick * 0.01
    covered = min(max(0.0, yes_inventory), ask_size)
    uncovered = max(0.0, ask_size - covered)
    return max(0.0, 1.0 - price) * uncovered


class Strategy(BaseStrategy):
    # hunt_arch_best_v23
    VOL_EWMA = 0.9
    VOL_COEFF = 1.5
    FILL_HIT = 1.0
    FILL_DECAY = 0.4
    INV_SKEW = 0.06
    BASE_SIZE = 10.0
    MAX_INV = 10
    MIN_LEG = 0.2
    NEAR_HALF = 2
    FAR_EXTRA = 2  # far rungs at ±(NEAR_HALF + FAR_EXTRA) from fair
    NEAR_FRAC = 0.62
    FAR_FRAC = 0.38

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0

    def on_step(self, state: StepState):
        actions = [CancelAll()]

        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks
        if bid is None and ask is None:
            return actions
        if bid is None:
            bid = max(1, ask - 6)  # type: ignore[operator]
        if ask is None:
            ask = min(99, bid + 6)  # type: ignore[operator]

        mid = (bid + ask) / 2.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_estimate = self.VOL_EWMA * self.vol_estimate + (1.0 - self.VOL_EWMA) * abs(move)
        self.prev_bid = bid
        self.prev_ask = ask

        if state.buy_filled_quantity > 0:
            self.fill_bias -= self.FILL_HIT
        if state.sell_filled_quantity > 0:
            self.fill_bias += self.FILL_HIT
        self.fill_bias *= self.FILL_DECAY

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * self.INV_SKEW
        fair = mid + self.fill_bias + inv_skew

        near_bid = max(1, int(round(fair - self.NEAR_HALF)))
        near_ask = min(99, int(round(fair + self.NEAR_HALF)))
        total_half = self.NEAR_HALF + self.FAR_EXTRA
        far_bid = max(1, int(round(fair - total_half)))
        far_ask = min(99, int(round(fair + total_half)))

        if far_bid >= near_bid:
            far_bid = max(1, near_bid - 1)
        if near_ask >= far_ask:
            far_ask = min(99, near_ask + 1)
        if near_bid >= near_ask:
            return actions

        vol_scale = max(0.1, 1.0 - self.vol_estimate * self.VOL_COEFF)
        bid_total = max(
            self.MIN_LEG,
            vol_scale * self.BASE_SIZE * max(0.0, 1.0 - net_inv / self.MAX_INV),
        )
        ask_total = max(
            self.MIN_LEG,
            vol_scale * self.BASE_SIZE * max(0.0, 1.0 + net_inv / self.MAX_INV),
        )

        bid_sizes = [
            round(bid_total * self.NEAR_FRAC, 4),
            round(bid_total * self.FAR_FRAC, 4),
        ]
        ask_sizes = [
            round(ask_total * self.NEAR_FRAC, 4),
            round(ask_total * self.FAR_FRAC, 4),
        ]
        bid_prices = [near_bid, far_bid]
        ask_prices = [near_ask, far_ask]

        def merge_levels(prices: list[int], sizes: list[float]) -> tuple[list[int], list[float]]:
            out_p: list[int] = []
            out_s: list[float] = []
            for p, s in zip(prices, sizes):
                if s < self.MIN_LEG:
                    continue
                if out_p and p == out_p[-1]:
                    out_s[-1] = round(out_s[-1] + s, 4)
                else:
                    out_p.append(p)
                    out_s.append(s)
            return out_p, out_s

        bid_prices, bid_sizes = merge_levels(bid_prices, bid_sizes)
        ask_prices, ask_sizes = merge_levels(ask_prices, ask_sizes)

        available = state.free_cash

        for price, size in zip(bid_prices, bid_sizes):
            if size < self.MIN_LEG:
                continue
            cost = price * 0.01 * size
            if cost <= available:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=price, quantity=size))
                available -= cost

        for price, size in zip(ask_prices, ask_sizes):
            if size < self.MIN_LEG:
                continue
            coll = _sell_reserved_cash(
                ask_tick=price,
                ask_size=size,
                yes_inventory=state.yes_inventory,
            )
            if coll <= available:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=price, quantity=size))
                available -= coll

        return actions
