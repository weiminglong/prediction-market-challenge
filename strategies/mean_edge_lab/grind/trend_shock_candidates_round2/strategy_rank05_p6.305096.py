from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 10.0, "fill_decay": 0.4, "fill_hit": 1.0, "inv_skew": 0.06, "max_inventory": 10.0, "min_size": 0.2, "shock_duration": 4, "shock_size_mult": 0.4, "shock_trigger_min": 2.2, "shock_trigger_vol_mult": 3.8, "shock_vol_floor": 0.35, "spread_base": 2, "spread_shock_extra": 2, "spread_vol_extra": 1, "spread_vol_threshold": 1.0, "trend_alpha": 0.18, "trend_decay": 0.82, "trend_weight": 0.62, "vol_coeff": 1.5, "vol_decay": 0.9, "vol_floor": 0.1}
