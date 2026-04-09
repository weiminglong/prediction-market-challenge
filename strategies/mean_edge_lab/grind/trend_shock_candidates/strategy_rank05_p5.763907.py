from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 11.048154, "fill_decay": 0.43515, "fill_hit": 1.634682, "inv_skew": 0.033561, "max_inventory": 9.005528, "min_size": 0.424802, "shock_duration": 5, "shock_size_mult": 0.760878, "shock_trigger_min": 3.147015, "shock_trigger_vol_mult": 2.66039, "shock_vol_floor": 0.22248, "spread_base": 2, "spread_shock_extra": 3, "spread_vol_extra": 1, "spread_vol_threshold": 1.07358, "trend_alpha": 0.144233, "trend_decay": 0.865783, "trend_weight": 0.25042, "vol_coeff": 1.144661, "vol_decay": 0.88112, "vol_floor": 0.052361}
