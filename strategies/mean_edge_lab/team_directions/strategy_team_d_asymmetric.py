"""Team D: asymmetric spread — widen toxic side (momentum + fills + inventory), stay tighter on the safer side."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _normalize_quotes(
    best_bid: int | None,
    best_ask: int | None,
) -> tuple[int, int, float] | None:
    if best_bid is None and best_ask is None:
        return None
    if best_bid is None:
        ask = int(best_ask)  # type: ignore[arg-type]
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


def _sell_collateral(ask_tick: int, ask_size: float, yes_inventory: float) -> float:
    ask_price = ask_tick * 0.01
    covered = min(max(0.0, yes_inventory), ask_size)
    uncovered = max(0.0, ask_size - covered)
    return max(0.0, (1.0 - ask_price) * uncovered)


class Strategy(BaseStrategy):
    """High-intensity quoting with asymmetric retreat on the side that looks more toxic."""

    MOMENTUM_DECAY = 0.88
    FILL_HIT = 1.0
    FILL_DECAY = 0.5
    INVENTORY_SKEW = 0.06
    # Keep fair aligned with the strong no-momentum baseline; asymmetry is spread-only.
    MOMENTUM_FAIR_WEIGHT = 0.0
    VOL_DECAY = 0.9
    SPREAD_BASE = 2
    VOL_THRESHOLD = 1.0
    VOL_EXTRA = 1
    VOL_FLOOR = 0.2
    VOL_COEFF = 0.7
    BASE_SIZE = 10.0
    MIN_SIZE = 0.2
    MAX_INVENTORY = 8.0
    MAX_INVENTORY_HARD = 40.0
    # Asymmetry caps (ticks added on top of base + vol); never zero out a side via spread alone.
    MAX_SIDE_EXTRA = 5

    def __init__(self) -> None:
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.comp_momentum = 0.0
        self.vol_estimate = 0.0

    def on_step(self, state: StepState):
        actions: list = [CancelAll()]
        norm = _normalize_quotes(
            state.competitor_best_bid_ticks,
            state.competitor_best_ask_ticks,
        )
        if norm is None:
            return actions
        bid, ask, mid = norm

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.comp_momentum = self.MOMENTUM_DECAY * self.comp_momentum + (
                1.0 - self.MOMENTUM_DECAY
            ) * move
            self.vol_estimate = self.VOL_DECAY * self.vol_estimate + (1.0 - self.VOL_DECAY) * abs(
                move
            )
        self.prev_bid = bid
        self.prev_ask = ask

        if state.buy_filled_quantity > 0.0:
            self.fill_bias -= self.FILL_HIT
        if state.sell_filled_quantity > 0.0:
            self.fill_bias += self.FILL_HIT
        self.fill_bias *= self.FILL_DECAY

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * self.INVENTORY_SKEW
        fair = mid + self.fill_bias + inv_skew + self.comp_momentum * self.MOMENTUM_FAIR_WEIGHT

        vol_extra = self.VOL_EXTRA if self.vol_estimate > self.VOL_THRESHOLD else 0
        bid_spread = self.SPREAD_BASE + vol_extra
        ask_spread = self.SPREAD_BASE + vol_extra

        # --- Asymmetric widening (prefer asymmetry over one-sided shutdown) ---
        bid_extra = 0
        ask_extra = 0
        mom = self.comp_momentum
        # Downward drift: bid more exposed; upward: ask more exposed.
        if mom < -0.18:
            bid_extra += min(self.MAX_SIDE_EXTRA, int(1 + abs(mom) * 6.5))
        elif mom > 0.18:
            ask_extra += min(self.MAX_SIDE_EXTRA, int(1 + mom * 6.5))

        # Fill imbalance: widen the side that just got lifted.
        if self.fill_bias < -0.45:
            bid_extra += 2
        elif self.fill_bias < -0.2:
            bid_extra += 1
        elif self.fill_bias > 0.45:
            ask_extra += 2
        elif self.fill_bias > 0.2:
            ask_extra += 1

        # Inventory skew spreads (mild tightening on offload side).
        max_inv = max(self.MAX_INVENTORY, 1e-9)
        inv_ratio = net_inv / max_inv
        if inv_ratio > 0.45:
            bid_extra += min(2, int(1 + inv_ratio * 2.0))
            ask_extra = max(0, ask_extra - 1)
        elif inv_ratio < -0.45:
            ask_extra += min(2, int(1 + abs(inv_ratio) * 2.0))
            bid_extra = max(0, bid_extra - 1)

        bid_spread += min(bid_extra, self.MAX_SIDE_EXTRA)
        ask_spread += min(ask_extra, self.MAX_SIDE_EXTRA)

        my_bid = max(1, int(round(fair - bid_spread)))
        my_ask = min(99, int(round(fair + ask_spread)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(self.VOL_FLOOR, 1.0 - self.vol_estimate * self.VOL_COEFF)
        base_sz = self.BASE_SIZE * vol_scale
        bid_size = max(self.MIN_SIZE, base_sz * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(self.MIN_SIZE, base_sz * max(0.0, 1.0 + net_inv / max_inv))

        if net_inv >= self.MAX_INVENTORY_HARD:
            bid_size = 0.0
        if net_inv <= -self.MAX_INVENTORY_HARD:
            ask_size = 0.0

        cash = max(0.0, state.free_cash)
        if bid_size > 0.0:
            buy_cost = my_bid * 0.01 * bid_size
            if buy_cost <= cash:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
                cash -= buy_cost

        if ask_size > 0.0:
            coll = _sell_collateral(my_ask, ask_size, state.yes_inventory)
            if coll <= cash:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
