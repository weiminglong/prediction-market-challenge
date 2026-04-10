from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 12.43366, "fill_decay": 0.365615, "fill_hit": 1.494976, "inv_skew": 0.039263, "max_inventory": 10.572665, "min_size": 0.241537, "shock_duration": 5, "shock_size_mult": 0.570777, "shock_trigger_min": 2.991949, "shock_trigger_vol_mult": 3.581672, "shock_vol_floor": 0.316723, "spread_base": 2, "spread_shock_extra": 3, "spread_vol_extra": 0, "spread_vol_threshold": 1.237506, "trend_alpha": 0.321282, "trend_decay": 0.79823, "trend_weight": 0.504992, "vol_coeff": 2.045554, "vol_decay": 0.862339, "vol_floor": 0.078881}
