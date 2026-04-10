from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 9.531654, "fill_decay": 0.382235, "fill_hit": 0.534191, "inv_skew": 0.054126, "max_inventory": 6.598441, "min_size": 0.322626, "shock_duration": 3, "shock_size_mult": 0.756981, "shock_trigger_min": 2.816062, "shock_trigger_vol_mult": 2.923261, "shock_vol_floor": 0.515599, "spread_base": 2, "spread_shock_extra": 2, "spread_vol_extra": 0, "spread_vol_threshold": 1.22044, "trend_alpha": 0.129387, "trend_decay": 0.773171, "trend_weight": 1.012201, "vol_coeff": 2.179037, "vol_decay": 0.84511, "vol_floor": 0.207867}
