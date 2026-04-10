from strategies.mean_edge_lab.grind.high_size_no_mom_template import ParametricNoCooldownNoMomentumStrategy

class Strategy(ParametricNoCooldownNoMomentumStrategy):
    PARAMS = {"base_size": 13.739228, "fill_decay": 0.45139, "fill_hit": 0.937655, "inventory_skew": 0.059422, "max_inventory": 13.637967, "min_size": 0.279033, "spread_base": 2, "vol_coeff": 0.968689, "vol_decay": 0.968098, "vol_floor": 0.348104, "vol_widen_extra": 2, "vol_widen_threshold": 1.079033}
