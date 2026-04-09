"""Team B: high-size MM + fade (mean-revert) competitor short-horizon moves.

Differs from hunt_arch_best_v23 by shifting fair opposite to an EWMA of competitor
mid changes, while keeping v23-style size (base 10, vol_coeff 1.5, no vol widen).
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

from strategies.mean_edge_lab.directions.common import estimate_sell_collateral, normalize_competitor_quotes


class Strategy(BaseStrategy):
    BASE_SIZE = 10.0
    MIN_SIZE = 0.2
    MAX_INV_SOFT = 10.0
    MAX_INV_HARD = 38.0
    FILL_HIT = 1.0
    FILL_DECAY = 0.4
    INV_SKEW = 0.06
    VOL_DECAY = 0.9
    VOL_COEFF = 1.5
    VOL_FLOOR = 0.1
    SPREAD_BASE = 2
    MOM_DECAY = 0.62
    FADE_MOM = 0.72
    SLOW_DECAY = 0.985
    FADE_ANCHOR = 0.14

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.momentum = 0.0
        self.slow_mid: float | None = None

    def on_step(self, state: StepState):
        actions = [CancelAll()]
        comp_bid, comp_ask, mid = normalize_competitor_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )

        move = 0.0
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_estimate = self.VOL_DECAY * self.vol_estimate + (1.0 - self.VOL_DECAY) * abs(move)
        self.prev_bid = comp_bid
        self.prev_ask = comp_ask

        self.momentum = self.MOM_DECAY * self.momentum + (1.0 - self.MOM_DECAY) * move
        if self.slow_mid is None:
            self.slow_mid = mid
        else:
            self.slow_mid = self.SLOW_DECAY * self.slow_mid + (1.0 - self.SLOW_DECAY) * mid

        anchor_pull = 0.0 if self.slow_mid is None else self.FADE_ANCHOR * (self.slow_mid - mid)

        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= self.FILL_HIT
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += self.FILL_HIT
        self.fill_bias *= self.FILL_DECAY

        net_inv = state.yes_inventory - state.no_inventory
        fair = (
            mid
            + self.fill_bias
            - net_inv * self.INV_SKEW
            - self.FADE_MOM * self.momentum
            + anchor_pull
        )

        my_bid = max(1, int(round(fair - self.SPREAD_BASE)))
        my_ask = min(99, int(round(fair + self.SPREAD_BASE)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(self.VOL_FLOOR, 1.0 - self.vol_estimate * self.VOL_COEFF)
        base = self.BASE_SIZE * vol_scale
        max_inv = max(self.MAX_INV_SOFT, 1e-9)
        bid_size = max(self.MIN_SIZE, base * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(self.MIN_SIZE, base * max(0.0, 1.0 + net_inv / max_inv))

        if net_inv >= self.MAX_INV_HARD:
            bid_size = 0.0
        if net_inv <= -self.MAX_INV_HARD:
            ask_size = 0.0

        available = state.free_cash
        if bid_size > 0.0:
            cost = my_bid * 0.01 * bid_size
            if cost <= available:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
                available -= cost

        if ask_size > 0.0:
            coll = estimate_sell_collateral(
                ask_tick=my_ask,
                ask_size=ask_size,
                yes_inventory=state.yes_inventory,
            )
            if coll <= available:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
