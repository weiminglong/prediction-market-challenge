from strategies.mean_edge_lab.grind.high_size_no_mom_template import ParametricNoCooldownNoMomentumStrategy

class Strategy(ParametricNoCooldownNoMomentumStrategy):
    PARAMS = {"base_size": 14.833638, "fill_decay": 0.396821, "fill_hit": 0.849218, "inventory_skew": 0.044553, "max_inventory": 15.085728, "min_size": 0.144282, "spread_base": 2, "vol_coeff": 0.910453, "vol_decay": 0.850835, "vol_floor": 0.156635, "vol_widen_extra": 1, "vol_widen_threshold": 1.316386}
