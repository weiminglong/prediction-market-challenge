from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 9.617229, "fill_decay": 0.485189, "fill_hit": 0.527365, "inv_skew": 0.076573, "max_inventory": 6.827987, "min_size": 0.072087, "shock_duration": 1, "shock_size_mult": 0.774827, "shock_trigger_min": 2.928408, "shock_trigger_vol_mult": 4.233706, "shock_vol_floor": 0.46478, "spread_base": 2, "spread_shock_extra": 2, "spread_vol_extra": 1, "spread_vol_threshold": 1.612906, "trend_alpha": 0.072124, "trend_decay": 0.754879, "trend_weight": 0.717126, "vol_coeff": 1.48332, "vol_decay": 0.964583, "vol_floor": 0.240692}
