from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 14.620702, "fill_decay": 0.52, "fill_hit": 0.55, "inv_skew": 0.048, "max_inventory": 10.678467, "min_size": 0.22, "shock_duration": 8, "shock_size_mult": 0.44, "shock_trigger_min": 2.2, "shock_trigger_vol_mult": 4.2, "shock_vol_floor": 0.337442, "spread_base": 2, "spread_shock_extra": 4, "spread_vol_extra": 0, "spread_vol_threshold": 1.102918, "trend_alpha": 0.12, "trend_decay": 0.7, "trend_weight": 0.67, "vol_coeff": 1.95, "vol_decay": 0.965, "vol_floor": 0.06}
