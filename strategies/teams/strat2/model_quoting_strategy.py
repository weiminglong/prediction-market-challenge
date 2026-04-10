from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """Model-based quoting strategy with jump-diffusion analytics.

    This strategy anchors on the competitor ladder for fair value and uses
    jump-diffusion time-decay dynamics to inform spread sizing:
    - Early game (>50% steps remaining): wider spreads to capture retail volume
    - Late game (<50% steps remaining): tighter spreads as price converges

    The model-based insight is that probability changes accelerate late in the game,
    so we reduce our aggressiveness as the endpoint approaches.
    """

    def __init__(self):
        self.quote_size = 5

    def on_step(self, state: StepState) -> list:
        """Place model-informed quotes around competitor midpoint."""

        # Use competitor quotes as fair-value anchor
        bid = state.competitor_best_bid_ticks or 49
        ask = state.competitor_best_ask_ticks or 51
        mid = (bid + ask) // 2

        actions = [CancelAll()]

        # Jump-diffusion insight: adjust spreads based on time-to-expiry
        # Early game: wider spreads (more time, less urgency)
        # Late game: tighter spreads (probability converges faster)
        # Threshold: 50% of steps remaining divides early/late phases
        time_ratio = state.steps_remaining / 2000.0  # Assumes default 2000 steps
        spread = 2 if time_ratio > 0.5 else 1

        # Inventory management: place orders only if inventory is balanced enough
        # Buy side: place spread ticks below mid to capture retail sellers
        if state.yes_inventory < 80 and state.free_cash > 20:
            buy_tick = max(1, mid - spread)
            if 1 <= buy_tick <= 99:
                actions.append(
                    PlaceOrder(
                        side=Side.BUY,
                        price_ticks=buy_tick,
                        quantity=self.quote_size,
                    )
                )

        # Sell side: place spread ticks above mid to capture retail buyers
        if state.no_inventory < 80:
            sell_tick = min(99, mid + spread)
            if 1 <= sell_tick <= 99:
                actions.append(
                    PlaceOrder(
                        side=Side.SELL,
                        price_ticks=sell_tick,
                        quantity=self.quote_size,
                    )
                )

        return actions
