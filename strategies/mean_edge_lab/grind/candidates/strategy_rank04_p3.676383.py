from strategies.mean_edge_lab.grind.high_size_template import ParametricHighSizeStrategy

class Strategy(ParametricHighSizeStrategy):
    PARAMS = {"base_size": 10.0, "cooldown_extra": 0, "cooldown_steps": 0, "fill_decay": 0.5, "fill_hit": 0.0, "inventory_skew": 0.06, "max_inventory": 10.0, "max_inventory_hard": 16.0, "min_size": 0.2, "momentum_decay": 0.5, "momentum_weight": 0.0, "spread_base": 2, "vol_coeff": 0.7, "vol_decay": 0.9, "vol_floor": 0.2, "vol_widen_extra": 1, "vol_widen_threshold": 1.0}
