"""Coordinate tune around the best trend-shock parameters."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from orderbook_pm_challenge.runner import run_batch

BASE_PARAMS: dict[str, float | int] = {
    "fill_hit": 0.649709,
    "fill_decay": 0.459286,
    "inv_skew": 0.05423,
    "vol_decay": 0.944588,
    "trend_decay": 0.765523,
    "trend_alpha": 0.155986,
    "trend_weight": 0.671633,
    "shock_vol_floor": 0.337442,
    "shock_trigger_min": 2.699927,
    "shock_trigger_vol_mult": 4.990285,
    "shock_duration": 6,
    "spread_base": 2,
    "spread_vol_threshold": 1.102918,
    "spread_vol_extra": 0,
    "spread_shock_extra": 3,
    "vol_floor": 0.085735,
    "vol_coeff": 1.623611,
    "shock_size_mult": 0.443468,
    "base_size": 14.620702,
    "min_size": 0.134722,
    "max_inventory": 10.678467,
}

INT_KEYS = {
    "shock_duration",
    "spread_base",
    "spread_vol_extra",
    "spread_shock_extra",
}

CHOICES: dict[str, list[float | int]] = {
    "fill_hit": [0.55, 0.65, 0.75],
    "fill_decay": [0.40, 0.46, 0.52],
    "inv_skew": [0.048, 0.054, 0.060],
    "vol_decay": [0.92, 0.945, 0.965],
    "trend_decay": [0.70, 0.77, 0.84],
    "trend_alpha": [0.12, 0.16, 0.20],
    "trend_weight": [0.55, 0.67, 0.80],
    "shock_vol_floor": [0.25, 0.34, 0.45],
    "shock_trigger_min": [2.2, 2.7, 3.2],
    "shock_trigger_vol_mult": [4.2, 5.0, 5.8],
    "shock_duration": [4, 6, 8],
    "spread_vol_threshold": [0.9, 1.1, 1.3],
    "spread_shock_extra": [2, 3, 4],
    "vol_floor": [0.06, 0.09, 0.14],
    "vol_coeff": [1.30, 1.62, 1.95],
    "shock_size_mult": [0.30, 0.44, 0.60],
    "base_size": [12.0, 14.6, 17.0],
    "min_size": [0.08, 0.14, 0.22],
    "max_inventory": [9.0, 10.7, 12.5],
}


@dataclass(frozen=True)
class Metric:
    mean_edge: float
    mean_retail_edge: float
    mean_arb_edge: float
    mean_final_wealth: float
    failure_count: int


def _round_params(params: dict[str, float | int]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, value in params.items():
        if key in INT_KEYS:
            out[key] = int(value)
        else:
            out[key] = round(float(value), 6)
    out["min_size"] = min(float(out["min_size"]), float(out["base_size"]))
    return out


def _strategy_source(params: dict[str, float | int]) -> str:
    return (
        "from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy\n\n"
        "class Strategy(ParametricTrendShockStrategy):\n"
        f"    PARAMS = {json.dumps(params, sort_keys=True)}\n"
    )


def _evaluate(
    params: dict[str, float | int],
    *,
    simulations: int,
    workers: int,
    seed_start: int,
    temp_dir: Path,
    tag: str,
) -> Metric:
    path = temp_dir / f"{tag}.py"
    path.write_text(_strategy_source(params), encoding="utf-8")
    batch = run_batch(
        strategy_path=str(path),
        n_simulations=simulations,
        workers=workers,
        seed_start=seed_start,
    )
    return Metric(
        mean_edge=batch.mean_edge,
        mean_retail_edge=batch.mean_retail_edge,
        mean_arb_edge=batch.mean_arb_edge,
        mean_final_wealth=batch.mean_final_wealth,
        failure_count=batch.failure_count,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2, help="Coordinate descent passes")
    parser.add_argument("--simulations", type=int, default=200, help="Simulation count per trial")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--seed-start", type=int, default=0, help="Simulation seed start")
    parser.add_argument(
        "--output-strategy",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/strategy_tuned_trend_shock.py"),
        help="Path for tuned strategy",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/tune_trend_shock_results.json"),
        help="Path for tune artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    current = _round_params(dict(BASE_PARAMS))
    history: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="tune-trend-shock-") as tmp:
        temp_dir = Path(tmp)
        metric = _evaluate(
            current,
            simulations=args.simulations,
            workers=args.workers,
            seed_start=args.seed_start,
            temp_dir=temp_dir,
            tag="baseline",
        )
        print(
            f"[baseline] edge={metric.mean_edge:+.6f} "
            f"retail={metric.mean_retail_edge:+.6f} "
            f"arb={metric.mean_arb_edge:+.6f}"
        )

        for it in range(1, args.iterations + 1):
            improved = False
            print(f"\n[iteration {it}]")
            for param, values in CHOICES.items():
                best_local_params = current
                best_local_metric = metric
                for value in values:
                    trial = _round_params({**current, param: value})
                    trial_metric = _evaluate(
                        trial,
                        simulations=args.simulations,
                        workers=args.workers,
                        seed_start=args.seed_start,
                        temp_dir=temp_dir,
                        tag=f"it{it}_{param}_{str(value).replace('.', '_')}",
                    )
                    history.append(
                        {
                            "iteration": it,
                            "param": param,
                            "value": value,
                            "params": trial,
                            "mean_edge": trial_metric.mean_edge,
                            "mean_retail_edge": trial_metric.mean_retail_edge,
                            "mean_arb_edge": trial_metric.mean_arb_edge,
                            "mean_final_wealth": trial_metric.mean_final_wealth,
                            "failure_count": trial_metric.failure_count,
                        }
                    )
                    print(
                        f"  {param}={value} -> edge={trial_metric.mean_edge:+.6f} "
                        f"retail={trial_metric.mean_retail_edge:+.6f} "
                        f"arb={trial_metric.mean_arb_edge:+.6f}"
                    )
                    if trial_metric.mean_edge > best_local_metric.mean_edge:
                        best_local_metric = trial_metric
                        best_local_params = trial

                if best_local_metric.mean_edge > metric.mean_edge:
                    print(
                        f"    improve {param}: "
                        f"{metric.mean_edge:+.6f} -> {best_local_metric.mean_edge:+.6f}"
                    )
                    current = best_local_params
                    metric = best_local_metric
                    improved = True

            if not improved:
                print("  no improvements this pass; stopping early")
                break

    args.output_strategy.parent.mkdir(parents=True, exist_ok=True)
    args.output_strategy.write_text(_strategy_source(current), encoding="utf-8")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact = {
        "generated_at": generated_at,
        "simulations": args.simulations,
        "workers": args.workers,
        "seed_start": args.seed_start,
        "final_params": current,
        "final_metric": {
            "mean_edge": metric.mean_edge,
            "mean_retail_edge": metric.mean_retail_edge,
            "mean_arb_edge": metric.mean_arb_edge,
            "mean_final_wealth": metric.mean_final_wealth,
            "failure_count": metric.failure_count,
        },
        "history": history,
        "output_strategy_path": str(args.output_strategy),
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\nFinal tuned metric:")
    print(
        f"  edge={metric.mean_edge:+.6f} "
        f"retail={metric.mean_retail_edge:+.6f} "
        f"arb={metric.mean_arb_edge:+.6f}"
    )
    print(f"  strategy written to {args.output_strategy}")
    print(f"  artifact written to {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
