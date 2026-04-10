from strategies.mean_edge_lab.grind.high_size_no_mom_template import ParametricNoCooldownNoMomentumStrategy

class Strategy(ParametricNoCooldownNoMomentumStrategy):
    PARAMS = {"base_size": 9.0, "fill_decay": 0.5, "fill_hit": 1.0, "inventory_skew": 0.06, "max_inventory": 8.0, "min_size": 0.2, "spread_base": 2, "vol_coeff": 0.7, "vol_decay": 0.9, "vol_floor": 0.2, "vol_widen_extra": 1, "vol_widen_threshold": 1.0}
