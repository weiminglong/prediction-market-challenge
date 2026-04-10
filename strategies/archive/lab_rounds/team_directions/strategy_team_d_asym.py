"""Team D: v23 core + asymmetric one-sided gating (toxicity + inventory).

Under stress, scale down the risky side toward zero while boosting size and
slightly tightening the quote on the side that reduces exposure.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

from strategies.mean_edge_lab.directions.common import estimate_sell_collateral, normalize_competitor_quotes


class Strategy(BaseStrategy):
    VOL_COEFF = 1.5
    FILL_BIAS_DECAY = 0.4
    FILL_HIT = 1.0
    INV_SKEW = 0.06
    SPREAD_HALF = 2
    BASE_SIZE = 10.0
    MAX_INV = 10.0
    MIN_SIZE = 0.2

    TOX_DECAY = 0.72
    ADVERSE = 0.45
    STRESS_SCALE = 1.15
    SAFE_BOOST_MAX = 0.35
    TIGHTEN_TICKS = 1

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.tox_buy = 0.0
        self.tox_sell = 0.0

    @staticmethod
    def _stress_to_mult(stress: float) -> float:
        """Map combined stress to size multiplier; 1 = normal, 0 = off."""
        if stress <= 0.0:
            return 1.0
        m = 1.0 - stress * Strategy.STRESS_SCALE
        return max(0.0, min(1.0, m))

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
            self.vol_estimate = 0.9 * self.vol_estimate + 0.1 * abs(move)
        self.prev_bid = comp_bid
        self.prev_ask = comp_ask

        buy_f = state.buy_filled_quantity
        sell_f = state.sell_filled_quantity

        if buy_f > 0.0:
            self.fill_bias -= self.FILL_HIT
            if move <= -self.ADVERSE:
                self.tox_buy += min(buy_f, 12.0) * 0.09 + abs(move) * 0.12
            elif move < -0.15:
                self.tox_buy += min(buy_f, 12.0) * 0.04
        if sell_f > 0.0:
            self.fill_bias += self.FILL_HIT
            if move >= self.ADVERSE:
                self.tox_sell += min(sell_f, 12.0) * 0.09 + abs(move) * 0.12
            elif move > 0.15:
                self.tox_sell += min(sell_f, 12.0) * 0.04

        self.fill_bias *= self.FILL_BIAS_DECAY
        self.tox_buy *= self.TOX_DECAY
        self.tox_sell *= self.TOX_DECAY

        shock = 1.35 + 0.5 * self.vol_estimate
        if move <= -shock:
            self.tox_buy += abs(move) * 0.08
        if move >= shock:
            self.tox_sell += abs(move) * 0.08

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * self.INV_SKEW
        fair = mid + self.fill_bias + inv_skew

        max_inv = max(self.MAX_INV, 1e-9)
        inv_buy_press = max(0.0, net_inv / max_inv)
        inv_sell_press = max(0.0, -net_inv / max_inv)

        stress_buy = self.tox_buy + inv_buy_press * 0.95 + self.vol_estimate * 0.08
        stress_sell = self.tox_sell + inv_sell_press * 0.95 + self.vol_estimate * 0.08

        bid_mult = self._stress_to_mult(stress_buy)
        ask_mult = self._stress_to_mult(stress_sell)

        bid_off = self.SPREAD_HALF
        ask_off = self.SPREAD_HALF
        if bid_mult < ask_mult:
            boost = min(self.SAFE_BOOST_MAX, (ask_mult - bid_mult) * 0.55 + (1.0 - bid_mult) * 0.12)
            ask_mult = min(1.0 + self.SAFE_BOOST_MAX, ask_mult + boost)
            ask_off = max(1, self.SPREAD_HALF - self.TIGHTEN_TICKS)
        elif ask_mult < bid_mult:
            boost = min(self.SAFE_BOOST_MAX, (bid_mult - ask_mult) * 0.55 + (1.0 - ask_mult) * 0.12)
            bid_mult = min(1.0 + self.SAFE_BOOST_MAX, bid_mult + boost)
            bid_off = max(1, self.SPREAD_HALF - self.TIGHTEN_TICKS)

        my_bid = max(1, int(round(fair - bid_off)))
        my_ask = min(99, int(round(fair + ask_off)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(0.1, 1.0 - self.vol_estimate * self.VOL_COEFF)
        base = self.BASE_SIZE * vol_scale
        bid_size = max(self.MIN_SIZE, base * max(0.0, 1.0 - net_inv / max_inv) * bid_mult)
        ask_size = max(self.MIN_SIZE, base * max(0.0, 1.0 + net_inv / max_inv) * ask_mult)

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
