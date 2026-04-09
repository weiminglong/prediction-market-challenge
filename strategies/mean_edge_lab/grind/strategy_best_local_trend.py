from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 14.113052, "fill_decay": 0.590229, "fill_hit": 0.497805, "inv_skew": 0.027657, "max_inventory": 8.674922, "min_size": 0.550072, "shock_duration": 4, "shock_size_mult": 0.15, "shock_trigger_min": 1.846626, "shock_trigger_vol_mult": 3.044949, "shock_vol_floor": 0.354703, "spread_base": 2, "spread_shock_extra": 2, "spread_vol_extra": 1, "spread_vol_threshold": 1.227116, "trend_alpha": 0.173101, "trend_decay": 0.654128, "trend_weight": 0.996645, "vol_coeff": 2.402131, "vol_decay": 0.935142, "vol_floor": 0.064398}
