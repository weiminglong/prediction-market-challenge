"""Direction K: pulse quoting cadence to reduce toxic exposure."""

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
        self.pulse_on = True

    def on_step(self, state: StepState):
        actions = [CancelAll()]

        _, _, mid = normalize_competitor_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )
        move = 0.0 if self.prev_mid is None else mid - self.prev_mid
        self.prev_mid = mid
        self.momentum = 0.6 * self.momentum + 0.4 * move
        self.vol_estimate = 0.88 * self.vol_estimate + 0.12 * abs(move)

        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= 0.7
            self.pulse_on = False
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += 0.7
            self.pulse_on = False
        self.fill_bias *= 0.65

        # Pulse cadence: quote every other step in calm, always in high volatility.
        should_quote = self.vol_estimate >= 1.0 or self.pulse_on
        self.pulse_on = not self.pulse_on
        if not should_quote:
            return actions

        net_inventory = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + 0.25 * self.momentum - 0.06 * net_inventory

        spread = 2 + (1 if self.vol_estimate >= 1.2 else 0)
        my_bid = max(1, int(round(fair - spread)))
        my_ask = min(99, int(round(fair + spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.5)
        base_size = 1.2 * vol_scale
        bid_size = max(0.12, base_size * max(0.0, 1.0 - net_inventory / 28.0))
        ask_size = max(0.12, base_size * max(0.0, 1.0 + net_inventory / 28.0))
        if net_inventory >= 38.0:
            bid_size = 0.0
        if net_inventory <= -38.0:
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
