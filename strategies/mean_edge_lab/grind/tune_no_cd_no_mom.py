"""Coordinate tuning for the top no-cooldown/no-momentum strategy."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from orderbook_pm_challenge.runner import run_batch

BASE_PARAMS: dict[str, float | int] = {
    "fill_hit": 1.0,
    "fill_decay": 0.5,
    "inventory_skew": 0.06,
    "vol_decay": 0.9,
    "spread_base": 2,
    "vol_widen_threshold": 1.0,
    "vol_widen_extra": 1,
    "vol_floor": 0.2,
    "vol_coeff": 0.7,
    "base_size": 10.0,
    "min_size": 0.2,
    "max_inventory": 8.0,
}

INT_KEYS = {"spread_base", "vol_widen_extra"}

SEARCH_VALUES: dict[str, list[float | int]] = {
    "fill_hit": [0.85, 1.0, 1.15],
    "fill_decay": [0.45, 0.5, 0.55],
    "inventory_skew": [0.055, 0.06, 0.065],
    "vol_decay": [0.87, 0.9, 0.93],
    "spread_base": [1, 2, 3],
    "vol_widen_threshold": [0.9, 1.0, 1.1],
    "vol_widen_extra": [0, 1, 2],
    "vol_floor": [0.16, 0.2, 0.24],
    "vol_coeff": [0.64, 0.7, 0.76],
    "base_size": [9.0, 10.0, 11.0],
    "min_size": [0.15, 0.2, 0.25],
    "max_inventory": [7.0, 8.0, 9.0],
}


@dataclass(frozen=True)
class Metric:
    mean_edge: float
    mean_retail_edge: float
    mean_arb_edge: float
    mean_final_wealth: float
    failure_count: int


def _round_params(params: dict[str, float | int]) -> dict[str, float | int]:
    rounded: dict[str, float | int] = {}
    for key, value in params.items():
        if key in INT_KEYS:
            rounded[key] = int(value)
        else:
            rounded[key] = round(float(value), 6)
    rounded["min_size"] = min(float(rounded["min_size"]), float(rounded["base_size"]))
    return rounded


def _strategy_source(params: dict[str, float | int]) -> str:
    return (
        "from strategies.mean_edge_lab.grind.high_size_no_mom_template import ParametricNoCooldownNoMomentumStrategy\n\n"
        "class Strategy(ParametricNoCooldownNoMomentumStrategy):\n"
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
    parser.add_argument("--iterations", type=int, default=2, help="Coordinate-descent passes")
    parser.add_argument("--simulations", type=int, default=200, help="Evaluation simulations per candidate")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--seed-start", type=int, default=0, help="Simulation seed start")
    parser.add_argument(
        "--output-strategy",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/strategy_tuned_no_cd_no_mom.py"),
        help="Path to write tuned strategy",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/tune_no_cd_no_mom_results.json"),
        help="Path to write tuning artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    current = _round_params(dict(BASE_PARAMS))
    history: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="tune-no-cd-no-mom-") as tmp:
        temp_dir = Path(tmp)
        metric = _evaluate(
            current,
            simulations=args.simulations,
            workers=args.workers,
            seed_start=args.seed_start,
            temp_dir=temp_dir,
            tag="baseline",
        )
        best_edge = metric.mean_edge
        print(f"[baseline] edge={metric.mean_edge:+.6f} retail={metric.mean_retail_edge:+.6f} arb={metric.mean_arb_edge:+.6f}")

        for iteration in range(1, args.iterations + 1):
            improved = False
            print(f"\n[iteration {iteration}]")
            for param, choices in SEARCH_VALUES.items():
                best_local_params = current
                best_local_metric = metric
                for choice in choices:
                    trial = dict(current)
                    trial[param] = choice
                    trial = _round_params(trial)
                    trial_metric = _evaluate(
                        trial,
                        simulations=args.simulations,
                        workers=args.workers,
                        seed_start=args.seed_start,
                        temp_dir=temp_dir,
                        tag=f"it{iteration}_{param}_{str(choice).replace('.', '_')}",
                    )
                    history.append(
                        {
                            "iteration": iteration,
                            "param": param,
                            "value": choice,
                            "params": trial,
                            "mean_edge": trial_metric.mean_edge,
                            "mean_retail_edge": trial_metric.mean_retail_edge,
                            "mean_arb_edge": trial_metric.mean_arb_edge,
                            "mean_final_wealth": trial_metric.mean_final_wealth,
                            "failure_count": trial_metric.failure_count,
                        }
                    )
                    print(
                        f"  {param}={choice} -> edge={trial_metric.mean_edge:+.6f} "
                        f"retail={trial_metric.mean_retail_edge:+.6f} "
                        f"arb={trial_metric.mean_arb_edge:+.6f}"
                    )
                    if trial_metric.mean_edge > best_local_metric.mean_edge:
                        best_local_metric = trial_metric
                        best_local_params = trial

                if best_local_metric.mean_edge > metric.mean_edge:
                    print(
                        f"    improve {param}: {metric.mean_edge:+.6f} -> {best_local_metric.mean_edge:+.6f}"
                    )
                    current = best_local_params
                    metric = best_local_metric
                    improved = True
                    if metric.mean_edge > best_edge:
                        best_edge = metric.mean_edge

            if not improved:
                print("  no improvement in this pass; stopping early")
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

    print("\nFinal tuned result:")
    print(
        f"  edge={metric.mean_edge:+.6f} retail={metric.mean_retail_edge:+.6f} "
        f"arb={metric.mean_arb_edge:+.6f}"
    )
    print(f"  strategy written to {args.output_strategy}")
    print(f"  artifact written to {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
