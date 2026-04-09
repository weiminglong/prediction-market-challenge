"""Direction F: aggressively quote inside competitor spread."""

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
        self.bid_cooldown = 0
        self.ask_cooldown = 0

    def on_step(self, state: StepState):
        actions = [CancelAll()]

        bid, ask, mid = normalize_competitor_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )
        move = 0.0 if self.prev_mid is None else mid - self.prev_mid
        self.prev_mid = mid
        self.momentum = 0.55 * self.momentum + 0.45 * move
        self.vol_estimate = 0.85 * self.vol_estimate + 0.15 * abs(move)

        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= 0.7
            self.bid_cooldown = 1
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += 0.7
            self.ask_cooldown = 1
        self.fill_bias *= 0.65

        net_inventory = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + 0.2 * self.momentum - 0.05 * net_inventory

        target_bid = bid + 1
        target_ask = ask - 1
        if self.bid_cooldown > 0:
            target_bid -= 1
            self.bid_cooldown -= 1
        if self.ask_cooldown > 0:
            target_ask += 1
            self.ask_cooldown -= 1

        # Blend inside-join target with fair-value center.
        fair_bid = int(round(fair - 1))
        fair_ask = int(round(fair + 1))
        my_bid = max(1, min(98, int(round(0.65 * target_bid + 0.35 * fair_bid))))
        my_ask = min(99, max(2, int(round(0.65 * target_ask + 0.35 * fair_ask))))
        if my_bid >= my_ask:
            my_bid = max(1, min(98, int(round(fair - 1))))
            my_ask = min(99, max(2, int(round(fair + 1))))
            if my_bid >= my_ask:
                return actions

        vol_scale = max(0.3, 1.0 - self.vol_estimate * 0.4)
        base_size = 1.35 * vol_scale
        bid_size = max(0.2, base_size * max(0.0, 1.0 - net_inventory / 24.0))
        ask_size = max(0.2, base_size * max(0.0, 1.0 + net_inventory / 24.0))
        if net_inventory >= 32.0:
            bid_size = 0.0
        if net_inventory <= -32.0:
            ask_size = 0.0

        available_cash = state.free_cash
        if bid_size > 0.0:
            bid_collateral = my_bid * 0.01 * bid_size
            if bid_collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
                available_cash -= bid_collateral

        if ask_size > 0.0:
            ask_collateral = estimate_sell_collateral(
                ask_tick=my_ask,
                ask_size=ask_size,
                yes_inventory=state.yes_inventory,
            )
            if ask_collateral <= available_cash:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
