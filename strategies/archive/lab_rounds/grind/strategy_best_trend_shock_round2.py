from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 12.524638, "fill_decay": 0.411074, "fill_hit": 0.968015, "inv_skew": 0.031775, "max_inventory": 7.094808, "min_size": 0.596682, "shock_duration": 4, "shock_size_mult": 0.40536, "shock_trigger_min": 2.367231, "shock_trigger_vol_mult": 4.892454, "shock_vol_floor": 0.587077, "spread_base": 2, "spread_shock_extra": 3, "spread_vol_extra": 2, "spread_vol_threshold": 0.816761, "trend_alpha": 0.152747, "trend_decay": 0.769552, "trend_weight": 0.3656, "vol_coeff": 1.456347, "vol_decay": 0.930025, "vol_floor": 0.228262}
