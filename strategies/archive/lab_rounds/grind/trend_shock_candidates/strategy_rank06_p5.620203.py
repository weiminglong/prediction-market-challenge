from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 12.895038, "fill_decay": 0.284649, "fill_hit": 1.614732, "inv_skew": 0.035874, "max_inventory": 8.126949, "min_size": 0.381611, "shock_duration": 4, "shock_size_mult": 0.439912, "shock_trigger_min": 2.852341, "shock_trigger_vol_mult": 5.445417, "shock_vol_floor": 0.572684, "spread_base": 2, "spread_shock_extra": 2, "spread_vol_extra": 0, "spread_vol_threshold": 1.002906, "trend_alpha": 0.196613, "trend_decay": 0.866782, "trend_weight": 0.27499, "vol_coeff": 1.899808, "vol_decay": 0.890412, "vol_floor": 0.340119}
