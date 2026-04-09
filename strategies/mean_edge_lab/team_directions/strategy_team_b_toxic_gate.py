"""Team B: high size + one-sided toxicity gates (buy/sell shutdown after shocks)."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

from strategies.mean_edge_lab.directions.common import estimate_sell_collateral, normalize_competitor_quotes


class Strategy(BaseStrategy):
    """Large quotes with aggressive one-sided pauses after adverse fills / momentum shocks."""

    BASE_SIZE = 10.0
    MIN_SIZE = 0.2
    MAX_INV_SOFT = 8.0
    MAX_INV_HARD = 38.0
    FILL_HIT = 1.0
    FILL_DECAY = 0.5
    INV_SKEW = 0.06
    SPREAD_BASE = 2
    VOL_DECAY = 0.9
    VOL_FLOOR = 0.2
    VOL_COEFF = 0.7
    VOL_WIDEN_TH = 1.0
    VOL_WIDEN_EXTRA = 1
    MOM_DECAY = 0.65
    MOM_WEIGHT = 0.0
    SHOCK_MOVE = 1.55
    ADVERSE_MOVE = 0.55

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.momentum = 0.0
        self.bid_gate = 0
        self.ask_gate = 0

    @staticmethod
    def _gate_duration(move: float, vol: float, filled_qty: float) -> int:
        """1–3 steps: stronger when move and vol are large or size is meaningful."""
        mag = abs(move)
        stress = mag + 0.35 * vol + 0.05 * min(filled_qty, 15.0)
        if stress >= 3.4:
            return 3
        if stress >= 2.2:
            return 2
        return 1

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

        buy_f = state.buy_filled_quantity
        sell_f = state.sell_filled_quantity

        if buy_f > 0.0:
            self.fill_bias -= self.FILL_HIT
            if move <= -self.ADVERSE_MOVE:
                self.bid_gate = max(self.bid_gate, self._gate_duration(move, self.vol_estimate, buy_f))
            elif move <= -0.2:
                self.bid_gate = max(self.bid_gate, 1)
        if sell_f > 0.0:
            self.fill_bias += self.FILL_HIT
            if move >= self.ADVERSE_MOVE:
                self.ask_gate = max(self.ask_gate, self._gate_duration(move, self.vol_estimate, sell_f))
            elif move >= 0.2:
                self.ask_gate = max(self.ask_gate, 1)
        self.fill_bias *= self.FILL_DECAY

        shock = max(self.SHOCK_MOVE, 0.95 * max(0.55, self.vol_estimate))
        if move <= -shock:
            self.bid_gate = max(self.bid_gate, self._gate_duration(move, self.vol_estimate, 0.0))
        if move >= shock:
            self.ask_gate = max(self.ask_gate, self._gate_duration(move, self.vol_estimate, 0.0))

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self.momentum * self.MOM_WEIGHT - net_inv * self.INV_SKEW

        widen = self.VOL_WIDEN_EXTRA if self.vol_estimate > self.VOL_WIDEN_TH else 0
        my_bid = max(1, int(round(fair - self.SPREAD_BASE - widen)))
        my_ask = min(99, int(round(fair + self.SPREAD_BASE + widen)))
        if my_bid >= my_ask:
            self._decay_gates(move)
            return actions

        vol_scale = max(self.VOL_FLOOR, 1.0 - self.vol_estimate * self.VOL_COEFF)
        base = self.BASE_SIZE * vol_scale
        max_inv = max(self.MAX_INV_SOFT, 1e-9)
        bid_size = max(self.MIN_SIZE, base * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(self.MIN_SIZE, base * max(0.0, 1.0 + net_inv / max_inv))

        if self.bid_gate > 0:
            bid_size = 0.0
        if self.ask_gate > 0:
            ask_size = 0.0

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

        self._decay_gates(move)
        return actions

    def _decay_gates(self, move: float) -> None:
        """Count down gates; early release when momentum favors the paused side."""
        if self.bid_gate > 0:
            self.bid_gate -= 1
            if move > 0.35 and self.momentum > 0.22:
                self.bid_gate = max(0, self.bid_gate - 1)
        if self.ask_gate > 0:
            self.ask_gate -= 1
            if move < -0.35 and self.momentum < -0.22:
                self.ask_gate = max(0, self.ask_gate - 1)
