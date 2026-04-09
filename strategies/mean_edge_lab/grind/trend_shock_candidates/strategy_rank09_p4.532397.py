from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 13.770245, "fill_decay": 0.474108, "fill_hit": 1.656037, "inv_skew": 0.052558, "max_inventory": 10.663162, "min_size": 0.329093, "shock_duration": 3, "shock_size_mult": 0.60703, "shock_trigger_min": 3.06376, "shock_trigger_vol_mult": 2.99949, "shock_vol_floor": 0.367324, "spread_base": 2, "spread_shock_extra": 2, "spread_vol_extra": 0, "spread_vol_threshold": 0.807832, "trend_alpha": 0.174076, "trend_decay": 0.897573, "trend_weight": 0.546875, "vol_coeff": 1.332968, "vol_decay": 0.943809, "vol_floor": 0.168934}
