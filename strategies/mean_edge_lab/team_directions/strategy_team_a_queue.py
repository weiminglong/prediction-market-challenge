"""Team A: queue-preserving variant of high-size no-cooldown / no-momentum family."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelOrder, PlaceOrder, Side, StepState

from strategies.mean_edge_lab.directions.common import (
    estimate_buy_cash_release,
    estimate_sell_collateral,
    should_replace,
)

CLIENT_BID = "teama-bid"
CLIENT_ASK = "teama-ask"

# Same tuned params as grind rank01 (mean edge ~5.357682 baseline family).
PARAMS = {
    "base_size": 10.0,
    "fill_decay": 0.5,
    "fill_hit": 1.0,
    "inventory_skew": 0.06,
    "max_inventory": 8.0,
    "min_size": 0.2,
    "spread_base": 2,
    "vol_coeff": 0.7,
    "vol_decay": 0.9,
    "vol_floor": 0.2,
    "vol_widen_extra": 1,
    "vol_widen_threshold": 1.0,
}


def _order_by_client_id(state: StepState, client_id: str):
    for o in state.own_orders:
        if o.order_id == client_id:
            return o
    return None


def _estimate_sell_reserve_release(order, yes_inventory: float) -> float:
    """Mirror place_order collateral for remaining size (cancel frees this reserved cash)."""
    return estimate_sell_collateral(
        ask_tick=order.price_ticks,
        ask_size=order.remaining_quantity,
        yes_inventory=yes_inventory,
    )


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0

    def on_step(self, state: StepState):
        p = PARAMS
        actions: list = []

        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks
        if bid is None and ask is None:
            for o in state.own_orders:
                actions.append(CancelOrder(o.order_id))
            return actions
        if bid is None:
            bid = max(1, int(ask) - 6)
        if ask is None:
            ask = min(99, int(bid) + 6)

        mid = (bid + ask) / 2.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            vd = float(p["vol_decay"])
            self.vol_estimate = vd * self.vol_estimate + (1.0 - vd) * abs(move)
        self.prev_bid = int(bid)
        self.prev_ask = int(ask)

        if state.buy_filled_quantity > 0:
            self.fill_bias -= float(p["fill_hit"])
        if state.sell_filled_quantity > 0:
            self.fill_bias += float(p["fill_hit"])
        self.fill_bias *= float(p["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias - net_inv * float(p["inventory_skew"])

        vol_widen = int(p["vol_widen_extra"]) if self.vol_estimate > float(p["vol_widen_threshold"]) else 0
        spread_base = int(p["spread_base"])
        my_bid = max(1, int(round(fair - spread_base - vol_widen)))
        my_ask = min(99, int(round(fair + spread_base + vol_widen)))
        if my_bid >= my_ask:
            for o in state.own_orders:
                actions.append(CancelOrder(o.order_id))
            return actions

        vol_scale = max(float(p["vol_floor"]), 1.0 - self.vol_estimate * float(p["vol_coeff"]))
        base_size = float(p["base_size"])
        max_inv = max(float(p["max_inventory"]), 1e-9)
        min_size = float(p["min_size"])
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        avail = float(state.free_cash)
        # Cancel stray participant orders (wrong client id) to avoid duplicate order_id errors.
        for o in state.own_orders:
            if o.side is Side.BUY and o.order_id != CLIENT_BID:
                actions.append(CancelOrder(o.order_id))
                avail += estimate_buy_cash_release(o)
            if o.side is Side.SELL and o.order_id != CLIENT_ASK:
                actions.append(CancelOrder(o.order_id))
                avail += _estimate_sell_reserve_release(o, state.yes_inventory)
        bid_order = _order_by_client_id(state, CLIENT_BID)
        ask_order = _order_by_client_id(state, CLIENT_ASK)

        if bid_order is not None and (
            bid_size <= 0.0
            or should_replace(
                bid_order,
                target_tick=my_bid,
                target_size=bid_size,
                tolerance_ticks=1,
                min_size_ratio=0.78,
                max_size_ratio=1.28,
            )
        ):
            actions.append(CancelOrder(bid_order.order_id))
            avail += estimate_buy_cash_release(bid_order)
            bid_order = None

        if ask_order is not None and (
            ask_size <= 0.0
            or should_replace(
                ask_order,
                target_tick=my_ask,
                target_size=ask_size,
                tolerance_ticks=1,
                min_size_ratio=0.78,
                max_size_ratio=1.28,
            )
        ):
            actions.append(CancelOrder(ask_order.order_id))
            avail += _estimate_sell_reserve_release(ask_order, state.yes_inventory)
            ask_order = None

        if bid_order is None and bid_size > 0.0:
            buy_collateral = my_bid * 0.01 * bid_size
            if buy_collateral <= avail + 1e-9:
                actions.append(
                    PlaceOrder(
                        side=Side.BUY,
                        price_ticks=my_bid,
                        quantity=bid_size,
                        client_order_id=CLIENT_BID,
                    )
                )
                avail -= buy_collateral

        if ask_order is None and ask_size > 0.0:
            ask_collateral = estimate_sell_collateral(
                ask_tick=my_ask,
                ask_size=ask_size,
                yes_inventory=state.yes_inventory,
            )
            if ask_collateral <= avail + 1e-9:
                actions.append(
                    PlaceOrder(
                        side=Side.SELL,
                        price_ticks=my_ask,
                        quantity=ask_size,
                        client_order_id=CLIENT_ASK,
                    )
                )

        return actions
