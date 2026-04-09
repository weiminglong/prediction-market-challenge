"""V2: Aggressive inside-competitor quoting with tiny size.

Theory: quote just inside the competitor spread (best bid/ask) to capture
ALL retail flow, but use very small size to limit arb damage. The arb
hits us first but only takes a tiny bite. Retail hits us right after.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def on_step(self, state: StepState):
        actions = [CancelAll()]

        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks

        if bid is None and ask is None:
            return actions
        if bid is None:
            bid = max(1, ask - 6)
        if ask is None:
            ask = min(99, bid + 6)

        # Quote just inside the competitor
        my_bid = bid + 1
        my_ask = ask - 1

        if my_bid >= my_ask:
            # No room inside competitor, skip or go one-sided
            mid = (bid + ask) / 2.0
            net_inv = state.yes_inventory - state.no_inventory
            if net_inv > 2:
                # Prefer to sell
                my_ask = ask - 1
                if 1 <= my_ask <= 99:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=1.5))
            elif net_inv < -2:
                my_bid = bid + 1
                if 1 <= my_bid <= 99:
                    cost = my_bid * 0.01 * 1.5
                    if cost <= state.free_cash:
                        actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=1.5))
            return actions

        # Inventory management
        net_inv = state.yes_inventory - state.no_inventory
        max_inv = 40

        # Very small size to limit arb damage
        bid_size = max(0.5, 1.5 * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.5, 1.5 * max(0.0, 1.0 + net_inv / max_inv))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
