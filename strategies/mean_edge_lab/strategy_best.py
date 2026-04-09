"""Best strategy found by mean-edge random search.

This file is updated by `strategies/mean_edge_lab/optimize.py`.
"""

from strategies.mean_edge_lab.strategy_template import ParametricStrategy


class Strategy(ParametricStrategy):
    PARAMS = dict(ParametricStrategy.PARAMS)
