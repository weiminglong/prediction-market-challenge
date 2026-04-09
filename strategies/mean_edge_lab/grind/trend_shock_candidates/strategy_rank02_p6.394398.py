from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy

class Strategy(ParametricTrendShockStrategy):
    PARAMS = {"base_size": 9.692471, "fill_decay": 0.52759, "fill_hit": 1.439822, "inv_skew": 0.054844, "max_inventory": 6.522518, "min_size": 0.114653, "shock_duration": 1, "shock_size_mult": 0.425627, "shock_trigger_min": 1.627008, "shock_trigger_vol_mult": 5.257224, "shock_vol_floor": 0.295436, "spread_base": 2, "spread_shock_extra": 2, "spread_vol_extra": 1, "spread_vol_threshold": 1.426036, "trend_alpha": 0.117414, "trend_decay": 0.705066, "trend_weight": 0.207578, "vol_coeff": 1.739061, "vol_decay": 0.947294, "vol_floor": 0.081173}
