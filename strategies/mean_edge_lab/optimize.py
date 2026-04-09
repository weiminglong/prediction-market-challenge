"""Random-search optimizer for maximizing strategy mean edge."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from orderbook_pm_challenge.runner import run_batch
from strategies.mean_edge_lab.strategy_template import DEFAULT_PARAMS

INT_KEYS = {
    "cooldown_steps",
    "spread_base",
    "cooldown_extra",
    "vol_widen_extra",
}


def _rounded_params(params: dict[str, float | int]) -> dict[str, float | int]:
    rounded: dict[str, float | int] = {}
    for key, value in params.items():
        if key in INT_KEYS:
            rounded[key] = int(value)
        else:
            rounded[key] = round(float(value), 4)
    return rounded


def _sample_params(rng: random.Random) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "fill_hit": rng.uniform(0.5, 1.75),
        "fill_decay": rng.uniform(0.35, 0.85),
        "inventory_skew": rng.uniform(0.02, 0.12),
        "momentum_weight": rng.uniform(0.0, 0.95),
        "momentum_decay": rng.uniform(0.2, 0.9),
        "vol_decay": rng.uniform(0.72, 0.98),
        "cooldown_steps": rng.randint(1, 4),
        "spread_base": rng.randint(1, 3),
        "cooldown_extra": rng.randint(0, 2),
        "vol_widen_threshold": rng.uniform(0.45, 2.7),
        "vol_widen_extra": rng.randint(0, 2),
        "vol_size_floor": rng.uniform(0.05, 0.55),
        "vol_size_coeff": rng.uniform(0.15, 1.0),
        "base_size": rng.uniform(0.45, 2.0),
        "min_size": rng.uniform(0.05, 0.45),
        "max_inventory": rng.uniform(14.0, 55.0),
        "max_inventory_hard": rng.uniform(18.0, 75.0),
        "fill_vol_spike": rng.uniform(0.0, 2.5),
        "cash_buffer": rng.uniform(0.0, 25.0),
    }

    params["max_inventory_hard"] = max(float(params["max_inventory_hard"]), float(params["max_inventory"]) + 1.0)
    params["min_size"] = min(float(params["min_size"]), float(params["base_size"]))
    return _rounded_params(params)


def _candidate_source(params: dict[str, float | int]) -> str:
    params_json = json.dumps(params, sort_keys=True)
    return (
        "from strategies.mean_edge_lab.strategy_template import ParametricStrategy\n\n"
        "class Strategy(ParametricStrategy):\n"
        f"    PARAMS = {params_json}\n"
    )


def _final_source(
    params: dict[str, float | int],
    *,
    screen_mean_edge: float,
    final_mean_edge: float,
    generated_at: str,
) -> str:
    params_json = json.dumps(params, sort_keys=True)
    return (
        '"""Best strategy found by mean-edge random search.\n\n'
        f"Generated: {generated_at}\n"
        f"Screen Mean Edge: {screen_mean_edge:.6f}\n"
        f"Final Mean Edge: {final_mean_edge:.6f}\n"
        '"""\n\n'
        "from strategies.mean_edge_lab.strategy_template import ParametricStrategy\n\n"
        "class Strategy(ParametricStrategy):\n"
        f"    PARAMS = {params_json}\n"
    )


def _evaluate_candidate(
    candidate_path: Path,
    *,
    simulations: int,
    workers: int,
    seed_start: int,
) -> dict[str, float]:
    batch = run_batch(
        strategy_path=str(candidate_path),
        n_simulations=simulations,
        workers=workers,
        seed_start=seed_start,
    )
    return {
        "mean_edge": batch.mean_edge,
        "mean_retail_edge": batch.mean_retail_edge,
        "mean_arb_edge": batch.mean_arb_edge,
        "mean_final_wealth": batch.mean_final_wealth,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=90, help="Total candidates to screen (includes baseline)")
    parser.add_argument("--screen-simulations", type=int, default=50, help="Simulations per candidate in screen stage")
    parser.add_argument("--finalists", type=int, default=12, help="Top screen candidates to rerun with final simulations")
    parser.add_argument("--final-simulations", type=int, default=200, help="Simulations per finalist in final stage")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers passed into run_batch")
    parser.add_argument("--seed", type=int, default=20260409, help="Random seed for sampling parameter sets")
    parser.add_argument("--seed-start", type=int, default=0, help="Simulation seed start for both stages")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("strategies/mean_edge_lab/strategy_best.py"),
        help="Output strategy file path",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/last_search_results.json"),
        help="JSON artifact with screen/final results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    total_candidates = max(1, args.candidates)
    finalists = max(1, min(args.finalists, total_candidates))
    rng = random.Random(args.seed)

    sampled: list[dict[str, float | int]] = [dict(DEFAULT_PARAMS)]
    seen = {json.dumps(_rounded_params(sampled[0]), sort_keys=True)}
    while len(sampled) < total_candidates:
        params = _sample_params(rng)
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        sampled.append(params)

    screen_results = []
    final_results = []

    with tempfile.TemporaryDirectory(prefix="mean-edge-lab-") as tmp_dir:
        tmp_root = Path(tmp_dir)

        for idx, params in enumerate(sampled, start=1):
            candidate_path = tmp_root / f"candidate_{idx:03d}.py"
            candidate_path.write_text(_candidate_source(params), encoding="utf-8")
            metrics = _evaluate_candidate(
                candidate_path,
                simulations=args.screen_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
            )
            result = {
                "rank_hint": idx,
                "params": params,
                **metrics,
            }
            screen_results.append(result)
            print(
                f"[screen {idx:03d}/{total_candidates:03d}] "
                f"mean_edge={metrics['mean_edge']:.6f} "
                f"retail={metrics['mean_retail_edge']:.6f} "
                f"arb={metrics['mean_arb_edge']:.6f}"
            )

        screen_results.sort(key=lambda item: item["mean_edge"], reverse=True)
        top_screen = screen_results[:finalists]

        print("\nTop screen candidates:")
        for idx, result in enumerate(top_screen, start=1):
            print(
                f"  {idx:02d}. edge={result['mean_edge']:.6f} "
                f"retail={result['mean_retail_edge']:.6f} arb={result['mean_arb_edge']:.6f}"
            )

        for idx, result in enumerate(top_screen, start=1):
            candidate_path = tmp_root / f"finalist_{idx:03d}.py"
            candidate_path.write_text(_candidate_source(result["params"]), encoding="utf-8")
            metrics = _evaluate_candidate(
                candidate_path,
                simulations=args.final_simulations,
                workers=args.workers,
                seed_start=args.seed_start,
            )
            final_result = {
                "screen_rank": idx,
                "screen_mean_edge": result["mean_edge"],
                "params": result["params"],
                **metrics,
            }
            final_results.append(final_result)
            print(
                f"[final {idx:02d}/{len(top_screen):02d}] "
                f"mean_edge={metrics['mean_edge']:.6f} "
                f"retail={metrics['mean_retail_edge']:.6f} "
                f"arb={metrics['mean_arb_edge']:.6f}"
            )

    final_results.sort(key=lambda item: item["mean_edge"], reverse=True)
    best = final_results[0]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _final_source(
            best["params"],
            screen_mean_edge=best["screen_mean_edge"],
            final_mean_edge=best["mean_edge"],
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )

    artifact = {
        "generated_at": generated_at,
        "seed": args.seed,
        "seed_start": args.seed_start,
        "workers": args.workers,
        "screen_simulations": args.screen_simulations,
        "final_simulations": args.final_simulations,
        "screen_results": screen_results,
        "final_results": final_results,
        "best": best,
        "output_strategy_path": str(args.output),
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\nBest finalist:")
    print(
        f"  mean_edge={best['mean_edge']:.6f} "
        f"retail={best['mean_retail_edge']:.6f} arb={best['mean_arb_edge']:.6f}"
    )
    print(f"  wrote strategy to {args.output}")
    print(f"  wrote artifact to {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
