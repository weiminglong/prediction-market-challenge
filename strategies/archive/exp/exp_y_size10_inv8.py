"""Exp Y: Size 10, inv 8 — even tighter cap."""
from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

class Strategy(BaseStrategy):
    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid = None
        self.prev_ask = None
        self.comp_momentum = 0.0
        self.bid_cooldown = 0
        self.ask_cooldown = 0
        self.vol_estimate = 0.0
    def on_step(self, state: StepState):
        actions = [CancelAll()]
        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks
        if bid is None and ask is None: return actions
        if bid is None: bid = max(1, ask - 6)
        if ask is None: ask = min(99, bid + 6)
        mid = (bid + ask) / 2.0
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid
            self.comp_momentum = 0.5 * self.comp_momentum + 0.5 * move
            self.vol_estimate = 0.9 * self.vol_estimate + 0.1 * abs(move)
        self.prev_bid = bid; self.prev_ask = ask
        if state.buy_filled_quantity > 0: self.fill_bias -= 1.0; self.bid_cooldown = 2
        if state.sell_filled_quantity > 0: self.fill_bias += 1.0; self.ask_cooldown = 2
        self.fill_bias *= 0.5
        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + (-net_inv * 0.06) + self.comp_momentum * 0.5
        vol_widen = 1 if self.vol_estimate > 1.0 else 0
        bid_spread = (3 if self.bid_cooldown > 0 else 2) + vol_widen
        ask_spread = (3 if self.ask_cooldown > 0 else 2) + vol_widen
        if self.bid_cooldown > 0: self.bid_cooldown -= 1
        if self.ask_cooldown > 0: self.ask_cooldown -= 1
        my_bid = max(1, int(round(fair - bid_spread)))
        my_ask = min(99, int(round(fair + ask_spread)))
        if my_bid >= my_ask: return actions
        vol_scale = max(0.2, 1.0 - self.vol_estimate * 0.7)
        bid_base = 0.8 if self.bid_cooldown > 0 else 10.0
        ask_base = 0.8 if self.ask_cooldown > 0 else 10.0
        max_inv = 8
        bid_size = max(0.2, bid_base * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(0.2, ask_base * vol_scale * max(0.0, 1.0 + net_inv / max_inv))
        cost = my_bid * 0.01 * bid_size
        if cost <= state.free_cash:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions
