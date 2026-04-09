"""Penny strategy: quote just inside the competitor's best bid/ask.

Instead of computing fair value and adding a spread, we penny the competitor:
- Our bid = competitor_bid + 1 (penny them to get filled first)
- Our ask = competitor_ask - 1 (penny them to get filled first)

This means we always have tighter spread than competitor and get all retail flow.
The arb will still sweep us when they move the market, but we'll capture more
retail by being at the best price.
"""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.bid_cooldown = 0
        self.ask_cooldown = 0
        self.prev_bid = None
        self.prev_ask = None
        self.vol_estimate = 0.0

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

        mid = (bid + ask) / 2.0

        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.vol_estimate = 0.9 * self.vol_estimate + 0.1 * abs(move)
        self.prev_bid = bid
        self.prev_ask = ask

        if state.buy_filled_quantity > 0:
            self.fill_bias -= 1.0
            self.bid_cooldown = 2
        if state.sell_filled_quantity > 0:
            self.fill_bias += 1.0
            self.ask_cooldown = 2
        self.fill_bias *= 0.5

        net_inv = state.yes_inventory - state.no_inventory
        inv_skew = -net_inv * 0.06

        # Penny the competitor: place just inside their quotes
        # But adjust for fill bias and inventory
        adj = self.fill_bias + inv_skew

        if self.bid_cooldown > 0:
            my_bid = max(1, bid - 1 + int(round(adj)))  # Stay behind on cooldown
            self.bid_cooldown -= 1
        else:
            my_bid = max(1, bid + 1 + int(round(adj)))  # Penny ahead normally

        if self.ask_cooldown > 0:
            my_ask = min(99, ask + 1 + int(round(adj)))  # Stay behind on cooldown
            self.ask_cooldown -= 1
        else:
            my_ask = min(99, ask - 1 + int(round(adj)))  # Penny ahead normally

        if my_bid >= my_ask:
            # Can't cross, fall back to mid ± 2
            my_bid = max(1, int(round(mid + adj - 2)))
            my_ask = min(99, int(round(mid + adj + 2)))
            if my_bid >= my_ask:
                return actions

        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)

        max_inv = 10
        bid_size = max(0.5, round(10.0 * vol_scale * max(0.0, 1.0 - net_inv / max_inv), 1))
        ask_size = max(0.5, round(10.0 * vol_scale * max(0.0, 1.0 + net_inv / max_inv), 1))

        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))

        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))

        return actions
