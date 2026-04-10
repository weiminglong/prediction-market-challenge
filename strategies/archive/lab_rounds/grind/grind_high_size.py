"""Aggressive multi-stage grinder for high-size strategy architecture."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from orderbook_pm_challenge.runner import run_batch
from strategies.mean_edge_lab.grind.high_size_template import DEFAULT_PARAMS

INT_KEYS = {
    "cooldown_steps",
    "spread_base",
    "cooldown_extra",
    "vol_widen_extra",
}


@dataclass(frozen=True)
class EvalResult:
    params: dict[str, float | int]
    mean_edge: float
    mean_retail_edge: float
    mean_arb_edge: float
    mean_final_wealth: float
    failure_count: int


def _rounded_params(params: dict[str, float | int]) -> dict[str, float | int]:
    rounded: dict[str, float | int] = {}
    for key, value in params.items():
        if key in INT_KEYS:
            rounded[key] = int(value)
        else:
            rounded[key] = round(float(value), 6)
    return rounded


def _candidate_source(params: dict[str, float | int]) -> str:
    return (
        "from strategies.mean_edge_lab.grind.high_size_template import ParametricHighSizeStrategy\n\n"
        "class Strategy(ParametricHighSizeStrategy):\n"
        f"    PARAMS = {json.dumps(params, sort_keys=True)}\n"
    )


def _sample_params(rng: random.Random) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "fill_hit": rng.uniform(0.0, 2.4),
        "fill_decay": rng.uniform(0.3, 0.85),
        "inventory_skew": rng.uniform(0.02, 0.12),
        "momentum_weight": rng.uniform(0.0, 0.45),
        "momentum_decay": rng.uniform(0.25, 0.9),
        "vol_decay": rng.uniform(0.75, 0.98),
        "cooldown_steps": rng.randint(0, 3),
        "spread_base": rng.randint(1, 3),
        "cooldown_extra": rng.randint(0, 2),
        "vol_widen_threshold": rng.uniform(0.45, 2.8),
        "vol_widen_extra": rng.randint(0, 2),
        "vol_floor": rng.uniform(0.05, 0.55),
        "vol_coeff": rng.uniform(0.2, 1.1),
        "base_size": rng.uniform(4.0, 22.0),
        "min_size": rng.uniform(0.05, 1.2),
        "max_inventory": rng.uniform(5.0, 28.0),
        "max_inventory_hard": rng.uniform(7.0, 42.0),
    }
    params["min_size"] = min(float(params["min_size"]), float(params["base_size"]))
    params["max_inventory_hard"] = max(float(params["max_inventory_hard"]), float(params["max_inventory"]) + 1.0)
    return _rounded_params(params)


def _seed_candidates() -> list[dict[str, float | int]]:
    seeds: list[dict[str, float | int]] = []
    seeds.append(_rounded_params(dict(DEFAULT_PARAMS)))

    # Prior best architecture from broad scan.
    seeds.append(
        _rounded_params(
            {
                "fill_hit": 1.0,
                "fill_decay": 0.5,
                "inventory_skew": 0.06,
                "momentum_weight": 0.0,
                "momentum_decay": 0.5,
                "vol_decay": 0.9,
                "cooldown_steps": 0,
                "spread_base": 2,
                "cooldown_extra": 0,
                "vol_widen_threshold": 1.0,
                "vol_widen_extra": 1,
                "vol_floor": 0.2,
                "vol_coeff": 0.7,
                "base_size": 10.0,
                "min_size": 0.2,
                "max_inventory": 10.0,
                "max_inventory_hard": 16.0,
            }
        )
    )

    # No-cooldown high-size with momentum enabled (runner-up family).
    seeds.append(
        _rounded_params(
            {
                "fill_hit": 1.0,
                "fill_decay": 0.5,
                "inventory_skew": 0.06,
                "momentum_weight": 0.5,
                "momentum_decay": 0.5,
                "vol_decay": 0.9,
                "cooldown_steps": 0,
                "spread_base": 2,
                "cooldown_extra": 0,
                "vol_widen_threshold": 1.0,
                "vol_widen_extra": 1,
                "vol_floor": 0.2,
                "vol_coeff": 0.7,
                "base_size": 10.0,
                "min_size": 0.2,
                "max_inventory": 10.0,
                "max_inventory_hard": 16.0,
            }
        )
    )

    # Minimal no-fill-bias variant.
    seeds.append(
        _rounded_params(
            {
                "fill_hit": 0.0,
                "fill_decay": 0.5,
                "inventory_skew": 0.06,
                "momentum_weight": 0.0,
                "momentum_decay": 0.5,
                "vol_decay": 0.9,
                "cooldown_steps": 0,
                "spread_base": 2,
                "cooldown_extra": 0,
                "vol_widen_threshold": 1.0,
                "vol_widen_extra": 1,
                "vol_floor": 0.2,
                "vol_coeff": 0.7,
                "base_size": 10.0,
                "min_size": 0.2,
                "max_inventory": 10.0,
                "max_inventory_hard": 16.0,
            }
        )
    )
    return seeds


def _evaluate_params(
    params: dict[str, float | int],
    *,
    simulations: int,
    workers: int,
    seed_start: int,
    temp_dir: Path,
    idx: int,
) -> EvalResult:
    candidate_path = temp_dir / f"candidate_{idx:05d}.py"
    candidate_path.write_text(_candidate_source(params), encoding="utf-8")
    batch = run_batch(
        strategy_path=str(candidate_path),
        n_simulations=simulations,
        workers=workers,
        seed_start=seed_start,
    )
    return EvalResult(
        params=params,
        mean_edge=batch.mean_edge,
        mean_retail_edge=batch.mean_retail_edge,
        mean_arb_edge=batch.mean_arb_edge,
        mean_final_wealth=batch.mean_final_wealth,
        failure_count=batch.failure_count,
    )


def _to_dict(result: EvalResult) -> dict:
    return {
        "params": result.params,
        "mean_edge": result.mean_edge,
        "mean_retail_edge": result.mean_retail_edge,
        "mean_arb_edge": result.mean_arb_edge,
        "mean_final_wealth": result.mean_final_wealth,
        "failure_count": result.failure_count,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=220, help="Total sampled candidates including seeded baselines")
    parser.add_argument("--stage1-simulations", type=int, default=20, help="Stage 1 simulation count")
    parser.add_argument("--stage2-top", type=int, default=70, help="Number of candidates promoted from stage 1")
    parser.add_argument("--stage2-simulations", type=int, default=80, help="Stage 2 simulation count")
    parser.add_argument("--stage3-top", type=int, default=16, help="Number of candidates promoted from stage 2")
    parser.add_argument("--stage3-simulations", type=int, default=200, help="Final stage simulation count")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for batch simulation")
    parser.add_argument("--seed", type=int, default=20260409, help="Random seed for sampling")
    parser.add_argument("--seed-start", type=int, default=0, help="Simulation seed start")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/candidates"),
        help="Directory to write top candidate strategy files",
    )
    parser.add_argument(
        "--best-strategy",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/strategy_best.py"),
        help="Path to write best strategy file",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/grind_results.json"),
        help="JSON artifact path",
    )
    parser.add_argument("--top-export-count", type=int, default=12, help="How many top finalists to export as files")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    seeded = _seed_candidates()
    sampled: list[dict[str, float | int]] = []
    seen = set()
    for params in seeded:
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        sampled.append(params)

    target_total = max(len(sampled), args.candidates)
    while len(sampled) < target_total:
        params = _sample_params(rng)
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        sampled.append(params)

    with tempfile.TemporaryDirectory(prefix="grind-high-size-") as tmp:
        temp_dir = Path(tmp)

        stage1: list[EvalResult] = []
        for idx, params in enumerate(sampled, start=1):
            result = _evaluate_params(
                params,
                simulations=args.stage1_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=temp_dir,
                idx=idx,
            )
            stage1.append(result)
            print(
                f"[stage1 {idx:03d}/{len(sampled):03d}] "
                f"edge={result.mean_edge:+.6f} "
                f"retail={result.mean_retail_edge:+.6f} "
                f"arb={result.mean_arb_edge:+.6f}"
            )

        stage1.sort(key=lambda item: item.mean_edge, reverse=True)
        stage2_candidates = stage1[: max(1, min(args.stage2_top, len(stage1)))]

        stage2: list[EvalResult] = []
        for idx, seed_result in enumerate(stage2_candidates, start=1):
            result = _evaluate_params(
                seed_result.params,
                simulations=args.stage2_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=temp_dir,
                idx=100000 + idx,
            )
            stage2.append(result)
            print(
                f"[stage2 {idx:03d}/{len(stage2_candidates):03d}] "
                f"edge={result.mean_edge:+.6f} "
                f"retail={result.mean_retail_edge:+.6f} "
                f"arb={result.mean_arb_edge:+.6f}"
            )

        stage2.sort(key=lambda item: item.mean_edge, reverse=True)
        stage3_candidates = stage2[: max(1, min(args.stage3_top, len(stage2)))]

        stage3: list[EvalResult] = []
        for idx, seed_result in enumerate(stage3_candidates, start=1):
            result = _evaluate_params(
                seed_result.params,
                simulations=args.stage3_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=temp_dir,
                idx=200000 + idx,
            )
            stage3.append(result)
            print(
                f"[stage3 {idx:03d}/{len(stage3_candidates):03d}] "
                f"edge={result.mean_edge:+.6f} "
                f"retail={result.mean_retail_edge:+.6f} "
                f"arb={result.mean_arb_edge:+.6f}"
            )

    stage3.sort(key=lambda item: item.mean_edge, reverse=True)
    best = stage3[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_export = stage3[: max(1, min(args.top_export_count, len(stage3)))]
    for idx, result in enumerate(top_export, start=1):
        safe_edge = f"{result.mean_edge:+.6f}".replace("+", "p").replace("-", "m")
        out_path = args.output_dir / f"strategy_rank{idx:02d}_{safe_edge}.py"
        out_path.write_text(_candidate_source(result.params), encoding="utf-8")

    args.best_strategy.parent.mkdir(parents=True, exist_ok=True)
    args.best_strategy.write_text(_candidate_source(best.params), encoding="utf-8")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact = {
        "generated_at": generated_at,
        "seed": args.seed,
        "seed_start": args.seed_start,
        "workers": args.workers,
        "candidates": len(sampled),
        "stage1_simulations": args.stage1_simulations,
        "stage2_simulations": args.stage2_simulations,
        "stage3_simulations": args.stage3_simulations,
        "stage1_results": [_to_dict(item) for item in stage1],
        "stage2_results": [_to_dict(item) for item in stage2],
        "stage3_results": [_to_dict(item) for item in stage3],
        "best": _to_dict(best),
        "best_strategy_path": str(args.best_strategy),
        "top_candidate_paths": sorted(str(path) for path in args.output_dir.glob("strategy_rank*.py")),
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\nTop finalists:")
    for idx, result in enumerate(stage3[:10], start=1):
        print(
            f"  {idx:02d}. edge={result.mean_edge:+.6f} "
            f"retail={result.mean_retail_edge:+.6f} "
            f"arb={result.mean_arb_edge:+.6f}"
        )
    print(f"\nBest strategy written to {args.best_strategy}")
    print(f"Search artifact written to {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
