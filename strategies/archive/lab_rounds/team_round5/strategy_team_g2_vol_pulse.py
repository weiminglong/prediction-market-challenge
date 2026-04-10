"""Round 5 G2: micro-volatility pulse widens spread and skews sides with exponential decay."""

from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _tick_to_price(price_ticks: int) -> float:
    return price_ticks * 0.01


class Strategy(BaseStrategy):
    PARAMS = {
        "base_size": 14.113052,
        "fill_decay": 0.590229,
        "fill_hit": 0.497805,
        "inv_skew": 0.027657,
        "inv_soft": 4.6,
        "inv_edge_mult": 0.0155,
        "max_inventory": 8.674922,
        "min_size": 0.550072,
        "side_damp_soft": 5.5,
        "damp_bid_when_long": 0.94,
        "damp_ask_when_short": 0.94,
        "uncovered_penalty": 0.12,
        "spread_base": 2,
        "spread_vol_extra": 1,
        "spread_vol_threshold": 1.227116,
        "trend_alpha": 0.173101,
        "trend_decay": 0.654128,
        "trend_weight": 0.996645,
        "vol_coeff": 2.402131,
        "vol_decay": 0.935142,
        "vol_floor": 0.064398,
        "reserve_cash_base": 0.0,
        "reserve_cash_vol": 5.0,
        # Pulse layer: fast micro-vol vs slow vol, continuous decay, asymmetric quotes
        "micro_decay": 0.48,
        "pulse_ratio_trigger": 2.05,
        "pulse_move_ticks": 1.65,
        "pulse_increment": 0.42,
        "pulse_cap": 1.0,
        "pulse_decay": 0.87,
        "pulse_spread_ticks": 3.25,
        "pulse_asym_ticks": 2.4,
        "pulse_size_floor": 0.78,
    }

    def __init__(self):
        self.fill_bias = 0.0
        self.prev_bid: int | None = None
        self.prev_ask: int | None = None
        self.vol_estimate = 0.0
        self.micro_vol = 0.0
        self.trend = 0.0
        self.pulse = 0.0
        self.pulse_dir = 0

    def _convex_inv_shift(self, net_inv: float) -> float:
        p = self.PARAMS
        if net_inv == 0.0:
            return 0.0
        sig = 1.0 if net_inv > 0 else -1.0
        max_inv = max(float(p["max_inventory"]), 1e-6)
        a = min(abs(net_inv), max_inv * 1.4)
        skew = float(p["inv_skew"])
        soft = max(float(p["inv_soft"]), 1e-6)
        em = float(p["inv_edge_mult"])
        if a <= soft:
            t = a / soft
            w = t * t * (3.0 - 2.0 * t)
            return -sig * skew * a * w
        excess = a - soft
        base = skew * soft
        tail = skew * excess * (1.0 + em * excess * excess)
        return -sig * (base + tail)

    def on_step(self, state: StepState):
        p = self.PARAMS
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
        move = 0.0
        if self.prev_bid is not None and self.prev_ask is not None:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            move = mid - prev_mid

            vol_decay = float(p["vol_decay"])
            self.vol_estimate = vol_decay * self.vol_estimate + (1.0 - vol_decay) * abs(move)

            md = float(p["micro_decay"])
            self.micro_vol = md * self.micro_vol + (1.0 - md) * abs(move)

            trend_decay = float(p["trend_decay"])
            trend_alpha = float(p["trend_alpha"])
            self.trend = trend_decay * self.trend + trend_alpha * move

            slow = max(self.vol_estimate, float(p["vol_floor"]))
            ratio = self.micro_vol / max(slow, 1e-9)
            burst = ratio >= float(p["pulse_ratio_trigger"]) or abs(move) >= float(p["pulse_move_ticks"])
            if burst and abs(move) > 1e-12:
                cap = float(p["pulse_cap"])
                self.pulse = min(cap, self.pulse + float(p["pulse_increment"]))
                self.pulse_dir = 1 if move > 0 else -1

        self.prev_bid = bid
        self.prev_ask = ask

        self.pulse *= float(p["pulse_decay"])
        if self.pulse < 1e-4:
            self.pulse = 0.0
            self.pulse_dir = 0

        fill_hit = float(p["fill_hit"])
        if state.buy_filled_quantity > 0:
            self.fill_bias -= fill_hit
        if state.sell_filled_quantity > 0:
            self.fill_bias += fill_hit
        self.fill_bias *= float(p["fill_decay"])

        net_inv = state.yes_inventory - state.no_inventory
        fair = mid + self.fill_bias + self._convex_inv_shift(net_inv) + self.trend * float(p["trend_weight"])

        half_spread = int(p["spread_base"])
        if self.vol_estimate > float(p["spread_vol_threshold"]):
            half_spread += int(p["spread_vol_extra"])

        pulse_w = float(self.pulse)
        extra_half = int(round(pulse_w * float(p["pulse_spread_ticks"])))
        half_spread += extra_half

        asym = int(round(pulse_w * float(p["pulse_asym_ticks"])))
        pd = self.pulse_dir
        if pd > 0:
            bid_half = max(1, half_spread - asym)
            ask_half = half_spread + asym
        elif pd < 0:
            bid_half = half_spread + asym
            ask_half = max(1, half_spread - asym)
        else:
            bid_half = ask_half = half_spread

        my_bid = max(1, int(round(fair - bid_half)))
        my_ask = min(99, int(round(fair + ask_half)))
        if my_bid >= my_ask:
            return actions

        vol_scale = max(float(p["vol_floor"]), 1.0 - self.vol_estimate * float(p["vol_coeff"]))
        if pulse_w > 0.0:
            floor = float(p["pulse_size_floor"])
            vol_scale *= floor + (1.0 - floor) * (1.0 - pulse_w)

        base_size = float(p["base_size"])
        max_inv = max(float(p["max_inventory"]), 1e-9)
        min_size = float(p["min_size"])
        bid_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 - net_inv / max_inv))
        ask_size = max(min_size, base_size * vol_scale * max(0.0, 1.0 + net_inv / max_inv))

        s_soft = float(p["side_damp_soft"])
        if net_inv > s_soft:
            bid_size *= float(p["damp_bid_when_long"])
        elif net_inv < -s_soft:
            ask_size *= float(p["damp_ask_when_short"])

        ask_px = _tick_to_price(my_ask)
        avail_yes = max(0.0, state.yes_inventory)
        if ask_size > avail_yes:
            uncovered = ask_size - avail_yes
            pen = float(p["uncovered_penalty"])
            ask_size = max(min_size, avail_yes + max(0.0, uncovered * (1.0 - pen)))

        vol_ref = max(self.vol_estimate, float(p["vol_floor"]))
        reserve_need = float(p["reserve_cash_base"]) + float(p["reserve_cash_vol"]) * vol_ref
        spendable = max(0.0, state.free_cash - reserve_need)

        buy_cost = my_bid * 0.01 * bid_size
        if bid_size > 0.0 and buy_cost > spendable:
            scale = spendable / max(buy_cost, 1e-12)
            bid_size = bid_size * scale
            if bid_size < min_size:
                bid_size = 0.0
            else:
                bid_size = max(min_size, bid_size)
            buy_cost = my_bid * 0.01 * bid_size

        free_after_bid = state.free_cash - buy_cost
        one_m_ask = max(1e-9, 1.0 - ask_px)
        if ask_size > 0.0:
            cov = min(ask_size, avail_yes)
            unc = max(0.0, ask_size - cov)
            coll = one_m_ask * unc
            if coll > free_after_bid + 1e-9:
                max_unc = max(0.0, free_after_bid / one_m_ask)
                new_ask = cov + max_unc
                if new_ask < min_size - 1e-9:
                    ask_size = 0.0
                else:
                    ask_size = max(min_size, new_ask)

        if bid_size > 0.0 and buy_cost <= state.free_cash and buy_cost <= spendable + 1e-6:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=bid_size))
        if ask_size > 0.0:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=ask_size))
        return actions
