"""Shared helpers for direction-exploration strategies."""

from __future__ import annotations

from orderbook_pm_challenge.types import OwnOrderView, Side, StepState


def normalize_competitor_quotes(
    best_bid: int | None,
    best_ask: int | None,
) -> tuple[int, int, float]:
    if best_bid is None and best_ask is None:
        bid = 47
        ask = 53
    elif best_bid is None:
        ask = int(best_ask)
        bid = max(1, ask - 6)
        if bid >= ask:
            ask = min(99, bid + 1)
    elif best_ask is None:
        bid = int(best_bid)
        ask = min(99, bid + 6)
        if bid >= ask:
            bid = max(1, ask - 1)
    else:
        bid = int(best_bid)
        ask = int(best_ask)
    return bid, ask, (bid + ask) / 2.0


def first_order_for_side(state: StepState, side: Side) -> OwnOrderView | None:
    for order in state.own_orders:
        if order.side is side:
            return order
    return None


def should_replace(
    order: OwnOrderView,
    *,
    target_tick: int,
    target_size: float,
    tolerance_ticks: int,
    min_size_ratio: float,
    max_size_ratio: float,
) -> bool:
    if abs(order.price_ticks - target_tick) > tolerance_ticks:
        return True
    if target_size <= 0.0:
        return True
    if order.remaining_quantity < target_size * min_size_ratio:
        return True
    if order.remaining_quantity > target_size * max_size_ratio:
        return True
    return False


def estimate_buy_cash_release(order: OwnOrderView) -> float:
    return max(0.0, order.price_ticks * 0.01 * order.remaining_quantity)


def estimate_sell_collateral(
    *,
    ask_tick: int,
    ask_size: float,
    yes_inventory: float,
) -> float:
    ask_price = ask_tick * 0.01
    covered = min(max(0.0, yes_inventory), ask_size)
    uncovered = max(0.0, ask_size - covered)
    return max(0.0, 1.0 - ask_price) * uncovered
