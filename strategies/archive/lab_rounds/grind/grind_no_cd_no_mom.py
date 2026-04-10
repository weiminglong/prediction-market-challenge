"""Aggressive grinder for the no-cooldown/no-momentum high-size family."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from orderbook_pm_challenge.runner import run_batch
from strategies.mean_edge_lab.grind.high_size_no_mom_template import DEFAULT_PARAMS

INT_KEYS = {"spread_base", "vol_widen_extra"}


@dataclass(frozen=True)
class EvalResult:
    params: dict[str, float | int]
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


def _candidate_source(params: dict[str, float | int]) -> str:
    return (
        "from strategies.mean_edge_lab.grind.high_size_no_mom_template import ParametricNoCooldownNoMomentumStrategy\n\n"
        "class Strategy(ParametricNoCooldownNoMomentumStrategy):\n"
        f"    PARAMS = {json.dumps(params, sort_keys=True)}\n"
    )


def _sample_params(rng: random.Random) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "fill_hit": rng.uniform(0.55, 1.95),
        "fill_decay": rng.uniform(0.34, 0.78),
        "inventory_skew": rng.uniform(0.03, 0.09),
        "vol_decay": rng.uniform(0.82, 0.97),
        "spread_base": rng.randint(1, 3),
        "vol_widen_threshold": rng.uniform(0.65, 1.75),
        "vol_widen_extra": rng.randint(0, 2),
        "vol_floor": rng.uniform(0.08, 0.38),
        "vol_coeff": rng.uniform(0.45, 1.0),
        "base_size": rng.uniform(7.0, 15.0),
        "min_size": rng.uniform(0.05, 0.9),
        "max_inventory": rng.uniform(7.0, 16.0),
    }
    return _round_params(params)


def _seed_candidates() -> list[dict[str, float | int]]:
    seeds = []
    seeds.append(_round_params(dict(DEFAULT_PARAMS)))

    for base_size in (8.0, 9.0, 10.0, 11.0, 12.0):
        for max_inventory in (8.0, 9.0, 10.0, 11.0, 12.0):
            seeds.append(
                _round_params(
                    {
                        "fill_hit": 1.0,
                        "fill_decay": 0.5,
                        "inventory_skew": 0.06,
                        "vol_decay": 0.9,
                        "spread_base": 2,
                        "vol_widen_threshold": 1.0,
                        "vol_widen_extra": 1,
                        "vol_floor": 0.2,
                        "vol_coeff": 0.7,
                        "base_size": base_size,
                        "min_size": 0.2,
                        "max_inventory": max_inventory,
                    }
                )
            )
    for fill_hit in (0.8, 1.0, 1.2, 1.4, 1.6):
        seeds.append(
            _round_params(
                {
                    "fill_hit": fill_hit,
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
                    "max_inventory": 10.0,
                }
            )
        )
    for vol_coeff in (0.55, 0.65, 0.75, 0.85):
        seeds.append(
            _round_params(
                {
                    "fill_hit": 1.0,
                    "fill_decay": 0.5,
                    "inventory_skew": 0.06,
                    "vol_decay": 0.9,
                    "spread_base": 2,
                    "vol_widen_threshold": 1.0,
                    "vol_widen_extra": 1,
                    "vol_floor": 0.2,
                    "vol_coeff": vol_coeff,
                    "base_size": 10.0,
                    "min_size": 0.2,
                    "max_inventory": 10.0,
                }
            )
        )
    return seeds


def _evaluate(
    params: dict[str, float | int],
    *,
    simulations: int,
    workers: int,
    seed_start: int,
    temp_dir: Path,
    idx: int,
) -> EvalResult:
    path = temp_dir / f"candidate_{idx:05d}.py"
    path.write_text(_candidate_source(params), encoding="utf-8")
    batch = run_batch(
        strategy_path=str(path),
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


def _as_dict(result: EvalResult) -> dict:
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
    parser.add_argument("--candidates", type=int, default=140, help="Total candidate count including seeded candidates")
    parser.add_argument("--stage1-simulations", type=int, default=18, help="Stage 1 simulation count")
    parser.add_argument("--stage2-top", type=int, default=36, help="Candidates promoted to stage 2")
    parser.add_argument("--stage2-simulations", type=int, default=70, help="Stage 2 simulation count")
    parser.add_argument("--stage3-top", type=int, default=14, help="Candidates promoted to stage 3")
    parser.add_argument("--stage3-simulations", type=int, default=200, help="Stage 3 simulation count")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=20260410, help="Sampling RNG seed")
    parser.add_argument("--seed-start", type=int, default=0, help="Simulation seed start")
    parser.add_argument(
        "--best-strategy",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/strategy_best_no_cd_no_mom.py"),
        help="Path to write best strategy",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/grind_no_cd_no_mom_results.json"),
        help="Path to write search artifact",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/no_cd_no_mom_candidates"),
        help="Directory for top candidate files",
    )
    parser.add_argument("--top-export-count", type=int, default=12, help="How many top strategies to export")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    seeded = _seed_candidates()
    rng = random.Random(args.seed)
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

    with tempfile.TemporaryDirectory(prefix="grind-no-cd-no-mom-") as tmp:
        tmp_dir = Path(tmp)

        stage1: list[EvalResult] = []
        for idx, params in enumerate(sampled, start=1):
            result = _evaluate(
                params,
                simulations=args.stage1_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=tmp_dir,
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
        stage2_input = stage1[: max(1, min(args.stage2_top, len(stage1)))]

        stage2: list[EvalResult] = []
        for idx, prev in enumerate(stage2_input, start=1):
            result = _evaluate(
                prev.params,
                simulations=args.stage2_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=tmp_dir,
                idx=100000 + idx,
            )
            stage2.append(result)
            print(
                f"[stage2 {idx:03d}/{len(stage2_input):03d}] "
                f"edge={result.mean_edge:+.6f} "
                f"retail={result.mean_retail_edge:+.6f} "
                f"arb={result.mean_arb_edge:+.6f}"
            )

        stage2.sort(key=lambda item: item.mean_edge, reverse=True)
        stage3_input = stage2[: max(1, min(args.stage3_top, len(stage2)))]

        stage3: list[EvalResult] = []
        for idx, prev in enumerate(stage3_input, start=1):
            result = _evaluate(
                prev.params,
                simulations=args.stage3_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=tmp_dir,
                idx=200000 + idx,
            )
            stage3.append(result)
            print(
                f"[stage3 {idx:03d}/{len(stage3_input):03d}] "
                f"edge={result.mean_edge:+.6f} "
                f"retail={result.mean_retail_edge:+.6f} "
                f"arb={result.mean_arb_edge:+.6f}"
            )

    stage3.sort(key=lambda item: item.mean_edge, reverse=True)
    best = stage3[0]

    args.best_strategy.parent.mkdir(parents=True, exist_ok=True)
    args.best_strategy.write_text(_candidate_source(best.params), encoding="utf-8")

    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(stage3[: max(1, min(args.top_export_count, len(stage3)))], start=1):
        safe_edge = f"{item.mean_edge:+.6f}".replace("+", "p").replace("-", "m")
        out = args.candidate_dir / f"strategy_rank{idx:02d}_{safe_edge}.py"
        out.write_text(_candidate_source(item.params), encoding="utf-8")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact = {
        "generated_at": generated_at,
        "seed": args.seed,
        "seed_start": args.seed_start,
        "workers": args.workers,
        "candidate_count": len(sampled),
        "stage1_simulations": args.stage1_simulations,
        "stage2_simulations": args.stage2_simulations,
        "stage3_simulations": args.stage3_simulations,
        "stage1_results": [_as_dict(item) for item in stage1],
        "stage2_results": [_as_dict(item) for item in stage2],
        "stage3_results": [_as_dict(item) for item in stage3],
        "best": _as_dict(best),
        "best_strategy_path": str(args.best_strategy),
        "candidate_dir": str(args.candidate_dir),
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\nTop finalists:")
    for idx, item in enumerate(stage3[:10], start=1):
        print(
            f"  {idx:02d}. edge={item.mean_edge:+.6f} "
            f"retail={item.mean_retail_edge:+.6f} "
            f"arb={item.mean_arb_edge:+.6f}"
        )
    print(f"\nBest strategy written to {args.best_strategy}")
    print(f"Search artifact written to {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
