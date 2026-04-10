from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 8.640958, "fill_decay": 0.282002, "fill_hit": 0.591273, "inv_skew": 0.066879, "max_inventory": 6.529583, "min_size": 0.416024, "shock_duration": 4, "shock_size_mult": 0.765048, "shock_trigger_min": 3.081568, "shock_trigger_vol_mult": 5.124923, "shock_vol_floor": 0.306781, "spread_base": 2, "spread_shock_extra": 1, "spread_vol_extra": 1, "spread_vol_threshold": 1.117368, "trend_alpha": 0.08026, "trend_decay": 0.692618, "trend_weight": 0.586967, "vol_coeff": 2.178562, "vol_decay": 0.876428, "vol_floor": 0.206294}
