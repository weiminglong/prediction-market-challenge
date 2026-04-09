"""Direction H: two-level ladder quoting on both sides."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

from strategies.mean_edge_lab.directions.common import estimate_sell_collateral, normalize_competitor_quotes


class Strategy(BaseStrategy):
    def __init__(self):
        self.prev_mid: float | None = None
        self.momentum = 0.0
        self.vol_estimate = 0.0
        self.fill_bias = 0.0

    def on_step(self, state: StepState):
        actions = [CancelAll()]

        _, _, mid = normalize_competitor_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )
        move = 0.0 if self.prev_mid is None else mid - self.prev_mid
        self.prev_mid = mid
        self.momentum = 0.6 * self.momentum + 0.4 * move
        self.vol_estimate = 0.87 * self.vol_estimate + 0.13 * abs(move)

        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= 0.8
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += 0.8
        self.fill_bias *= 0.6

        net_inventory = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + 0.3 * self.momentum - 0.06 * net_inventory

        near_width = 2
        far_width = 4 if self.vol_estimate < 1.1 else 5
        near_bid = max(1, int(round(fair - near_width)))
        near_ask = min(99, int(round(fair + near_width)))
        far_bid = max(1, int(round(fair - far_width)))
        far_ask = min(99, int(round(fair + far_width)))
        if near_bid >= near_ask or far_bid >= far_ask:
            return actions

        vol_scale = max(0.25, 1.0 - self.vol_estimate * 0.45)
        base_total = 1.35 * vol_scale
        near_frac = 0.65
        far_frac = 0.35
        bid_total = max(0.2, base_total * max(0.0, 1.0 - net_inventory / 28.0))
        ask_total = max(0.2, base_total * max(0.0, 1.0 + net_inventory / 28.0))
        if net_inventory >= 40.0:
            bid_total = 0.0
        if net_inventory <= -40.0:
            ask_total = 0.0

        bid_sizes = [round(bid_total * near_frac, 4), round(bid_total * far_frac, 4)]
        ask_sizes = [round(ask_total * near_frac, 4), round(ask_total * far_frac, 4)]
        bid_prices = [near_bid, far_bid]
        ask_prices = [near_ask, far_ask]

        available_cash = state.free_cash
        for price, size in zip(bid_prices, bid_sizes):
            if size <= 0.0:
                continue
            collateral = price * 0.01 * size
            if collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=price, quantity=size))
                available_cash -= collateral

        for price, size in zip(ask_prices, ask_sizes):
            if size <= 0.0:
                continue
            collateral = estimate_sell_collateral(
                ask_tick=price,
                ask_size=size,
                yes_inventory=state.yes_inventory,
            )
            if collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=price, quantity=size))
                available_cash -= collateral

        return actions
