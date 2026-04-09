"""Grind the trend+shock architecture for higher mean edge."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from orderbook_pm_challenge.runner import run_batch
from strategies.mean_edge_lab.grind.trend_shock_template import DEFAULT_PARAMS

INT_KEYS = {
    "shock_duration",
    "spread_base",
    "spread_vol_extra",
    "spread_shock_extra",
}


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


def _sample_params(rng: random.Random) -> dict[str, float | int]:
    p: dict[str, float | int] = {
        "fill_hit": rng.uniform(0.5, 1.8),
        "fill_decay": rng.uniform(0.25, 0.7),
        "inv_skew": rng.uniform(0.03, 0.1),
        "vol_decay": rng.uniform(0.82, 0.97),
        "trend_decay": rng.uniform(0.68, 0.94),
        "trend_alpha": rng.uniform(0.06, 0.35),
        "trend_weight": rng.uniform(0.2, 1.05),
        "shock_vol_floor": rng.uniform(0.2, 0.65),
        "shock_trigger_min": rng.uniform(1.4, 3.4),
        "shock_trigger_vol_mult": rng.uniform(2.4, 5.5),
        "shock_duration": rng.randint(1, 6),
        "spread_base": rng.randint(1, 3),
        "spread_vol_threshold": rng.uniform(0.7, 1.7),
        "spread_vol_extra": rng.randint(0, 2),
        "spread_shock_extra": rng.randint(1, 3),
        "vol_floor": rng.uniform(0.05, 0.35),
        "vol_coeff": rng.uniform(0.8, 2.2),
        "shock_size_mult": rng.uniform(0.2, 0.85),
        "base_size": rng.uniform(7.0, 15.0),
        "min_size": rng.uniform(0.05, 0.6),
        "max_inventory": rng.uniform(6.0, 14.0),
    }
    return _round_params(p)


def _candidate_source(params: dict[str, float | int]) -> str:
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
    idx: int,
) -> EvalResult:
    path = temp_dir / f"cand_{idx:05d}.py"
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


def _to_dict(x: EvalResult) -> dict:
    return {
        "params": x.params,
        "mean_edge": x.mean_edge,
        "mean_retail_edge": x.mean_retail_edge,
        "mean_arb_edge": x.mean_arb_edge,
        "mean_final_wealth": x.mean_final_wealth,
        "failure_count": x.failure_count,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=120, help="Total candidates including seeds")
    parser.add_argument("--stage1-simulations", type=int, default=18, help="Stage 1 sims")
    parser.add_argument("--stage2-top", type=int, default=30, help="Promoted candidates")
    parser.add_argument("--stage2-simulations", type=int, default=70, help="Stage 2 sims")
    parser.add_argument("--stage3-top", type=int, default=12, help="Finalists")
    parser.add_argument("--stage3-simulations", type=int, default=200, help="Stage 3 sims")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=20260411, help="Sampling RNG seed")
    parser.add_argument("--seed-start", type=int, default=0, help="Simulation seed start")
    parser.add_argument(
        "--best-strategy",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/strategy_best_trend_shock.py"),
        help="Output path for best strategy file",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/trend_shock_candidates"),
        help="Directory for top candidate files",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/grind_trend_shock_results.json"),
        help="Output path for artifact JSON",
    )
    parser.add_argument("--top-export-count", type=int, default=10, help="Exported finalist count")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    seeded = [_round_params(dict(DEFAULT_PARAMS))]
    sampled: list[dict[str, float | int]] = []
    seen = set()
    for p in seeded:
        k = json.dumps(p, sort_keys=True)
        if k not in seen:
            seen.add(k)
            sampled.append(p)

    while len(sampled) < args.candidates:
        p = _sample_params(rng)
        k = json.dumps(p, sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        sampled.append(p)

    with tempfile.TemporaryDirectory(prefix="grind-trend-shock-") as tmp:
        temp_dir = Path(tmp)

        stage1: list[EvalResult] = []
        for idx, params in enumerate(sampled, start=1):
            res = _evaluate(
                params,
                simulations=args.stage1_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=temp_dir,
                idx=idx,
            )
            stage1.append(res)
            print(
                f"[stage1 {idx:03d}/{len(sampled):03d}] "
                f"edge={res.mean_edge:+.6f} "
                f"retail={res.mean_retail_edge:+.6f} "
                f"arb={res.mean_arb_edge:+.6f}"
            )

        stage1.sort(key=lambda x: x.mean_edge, reverse=True)
        stage2_input = stage1[: max(1, min(args.stage2_top, len(stage1)))]

        stage2: list[EvalResult] = []
        for idx, prev in enumerate(stage2_input, start=1):
            res = _evaluate(
                prev.params,
                simulations=args.stage2_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=temp_dir,
                idx=100000 + idx,
            )
            stage2.append(res)
            print(
                f"[stage2 {idx:03d}/{len(stage2_input):03d}] "
                f"edge={res.mean_edge:+.6f} "
                f"retail={res.mean_retail_edge:+.6f} "
                f"arb={res.mean_arb_edge:+.6f}"
            )

        stage2.sort(key=lambda x: x.mean_edge, reverse=True)
        stage3_input = stage2[: max(1, min(args.stage3_top, len(stage2)))]

        stage3: list[EvalResult] = []
        for idx, prev in enumerate(stage3_input, start=1):
            res = _evaluate(
                prev.params,
                simulations=args.stage3_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                temp_dir=temp_dir,
                idx=200000 + idx,
            )
            stage3.append(res)
            print(
                f"[stage3 {idx:03d}/{len(stage3_input):03d}] "
                f"edge={res.mean_edge:+.6f} "
                f"retail={res.mean_retail_edge:+.6f} "
                f"arb={res.mean_arb_edge:+.6f}"
            )

    stage3.sort(key=lambda x: x.mean_edge, reverse=True)
    best = stage3[0]

    args.best_strategy.parent.mkdir(parents=True, exist_ok=True)
    args.best_strategy.write_text(_candidate_source(best.params), encoding="utf-8")

    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    for idx, x in enumerate(stage3[: max(1, min(args.top_export_count, len(stage3)))], start=1):
        safe = f"{x.mean_edge:+.6f}".replace("+", "p").replace("-", "m")
        out = args.candidate_dir / f"strategy_rank{idx:02d}_{safe}.py"
        out.write_text(_candidate_source(x.params), encoding="utf-8")

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
        "stage1_results": [_to_dict(x) for x in stage1],
        "stage2_results": [_to_dict(x) for x in stage2],
        "stage3_results": [_to_dict(x) for x in stage3],
        "best": _to_dict(best),
        "best_strategy_path": str(args.best_strategy),
        "candidate_dir": str(args.candidate_dir),
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\nTop finalists:")
    for idx, x in enumerate(stage3[:10], start=1):
        print(
            f"  {idx:02d}. edge={x.mean_edge:+.6f} "
            f"retail={x.mean_retail_edge:+.6f} "
            f"arb={x.mean_arb_edge:+.6f}"
        )
    print(f"\nBest strategy written to {args.best_strategy}")
    print(f"Search artifact written to {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
