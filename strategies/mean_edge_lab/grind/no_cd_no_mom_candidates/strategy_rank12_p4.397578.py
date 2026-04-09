from strategies.mean_edge_lab.grind.high_size_no_mom_template import ParametricNoCooldownNoMomentumStrategy

class Strategy(ParametricNoCooldownNoMomentumStrategy):
    PARAMS = {"base_size": 14.032644, "fill_decay": 0.390806, "fill_hit": 1.133519, "inventory_skew": 0.034611, "max_inventory": 9.262033, "min_size": 0.743754, "spread_base": 2, "vol_coeff": 0.827933, "vol_decay": 0.859372, "vol_floor": 0.090239, "vol_widen_extra": 1, "vol_widen_threshold": 1.322325}
