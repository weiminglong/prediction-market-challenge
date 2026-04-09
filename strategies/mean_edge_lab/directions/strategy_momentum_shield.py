"""Direction D: momentum-shielded one-sided quoting."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelOrder, PlaceOrder, Side, StepState

from strategies.mean_edge_lab.directions.common import (
    estimate_buy_cash_release,
    estimate_sell_collateral,
    first_order_for_side,
    normalize_competitor_quotes,
    should_replace,
)


class Strategy(BaseStrategy):
    def __init__(self):
        self.prev_mid: float | None = None
        self.momentum = 0.0
        self.vol_estimate = 0.0
        self.fill_bias = 0.0
        self.bid_shield_steps = 0
        self.ask_shield_steps = 0

    def on_step(self, state: StepState):
        _, _, mid = normalize_competitor_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )
        move = 0.0 if self.prev_mid is None else mid - self.prev_mid
        self.prev_mid = mid

        self.momentum = 0.7 * self.momentum + 0.3 * move
        self.vol_estimate = 0.86 * self.vol_estimate + 0.14 * abs(move)

        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= 0.85
            if move <= 0.0:
                self.bid_shield_steps = max(self.bid_shield_steps, 2)
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += 0.85
            if move >= 0.0:
                self.ask_shield_steps = max(self.ask_shield_steps, 2)
        self.fill_bias *= 0.62

        if self.momentum <= -0.55:
            self.bid_shield_steps = max(self.bid_shield_steps, 1)
        if self.momentum >= 0.55:
            self.ask_shield_steps = max(self.ask_shield_steps, 1)

        net_inventory = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self.momentum * 0.45 - net_inventory * 0.055

        spread = 2 if self.vol_estimate < 1.0 else 3
        my_bid = max(1, int(round(fair - spread)))
        my_ask = min(99, int(round(fair + spread)))
        if my_bid >= my_ask:
            return []

        vol_scale = max(0.18, 1.0 - self.vol_estimate * 0.52)
        base_size = 1.0 * vol_scale
        bid_size = max(0.1, base_size * max(0.0, 1.0 - net_inventory / 30.0))
        ask_size = max(0.1, base_size * max(0.0, 1.0 + net_inventory / 30.0))

        if self.bid_shield_steps > 0 or net_inventory >= 40.0:
            bid_size = 0.0
        if self.ask_shield_steps > 0 or net_inventory <= -40.0:
            ask_size = 0.0

        actions = []
        available_cash = state.free_cash
        bid_order = first_order_for_side(state, Side.BUY)
        ask_order = first_order_for_side(state, Side.SELL)

        if bid_order is not None and should_replace(
            bid_order,
            target_tick=my_bid,
            target_size=bid_size,
            tolerance_ticks=1,
            min_size_ratio=0.4,
            max_size_ratio=2.0,
        ):
            actions.append(CancelOrder(bid_order.order_id))
            available_cash += estimate_buy_cash_release(bid_order)
            bid_order = None

        if ask_order is not None and should_replace(
            ask_order,
            target_tick=my_ask,
            target_size=ask_size,
            tolerance_ticks=1,
            min_size_ratio=0.4,
            max_size_ratio=2.0,
        ):
            actions.append(CancelOrder(ask_order.order_id))
            ask_order = None

        if bid_order is None and bid_size > 0.0:
            bid_collateral = my_bid * 0.01 * bid_size
            if bid_collateral <= available_cash:
                actions.append(
                    PlaceOrder(
                        side=Side.BUY,
                        price_ticks=my_bid,
                        quantity=bid_size,
                        client_order_id="dird-bid",
                    )
                )
                available_cash -= bid_collateral

        if ask_order is None and ask_size > 0.0:
            ask_collateral = estimate_sell_collateral(
                ask_tick=my_ask,
                ask_size=ask_size,
                yes_inventory=state.yes_inventory,
            )
            if ask_collateral <= available_cash:
                actions.append(
                    PlaceOrder(
                        side=Side.SELL,
                        price_ticks=my_ask,
                        quantity=ask_size,
                        client_order_id="dird-ask",
                    )
                )

        if self.bid_shield_steps > 0:
            self.bid_shield_steps -= 1
        if self.ask_shield_steps > 0:
            self.ask_shield_steps -= 1
        return actions
