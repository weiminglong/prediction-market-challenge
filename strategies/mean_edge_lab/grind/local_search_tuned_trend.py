"""Local perturbation search around tuned trend-shock parameters."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from orderbook_pm_challenge.runner import run_batch

BASE_PARAMS = {
    "base_size": 14.620702,
    "fill_decay": 0.52,
    "fill_hit": 0.55,
    "inv_skew": 0.048,
    "max_inventory": 10.678467,
    "min_size": 0.22,
    "shock_duration": 8,
    "shock_size_mult": 0.44,
    "shock_trigger_min": 2.2,
    "shock_trigger_vol_mult": 4.2,
    "shock_vol_floor": 0.337442,
    "spread_base": 2,
    "spread_shock_extra": 4,
    "spread_vol_extra": 0,
    "spread_vol_threshold": 1.102918,
    "trend_alpha": 0.12,
    "trend_decay": 0.7,
    "trend_weight": 0.67,
    "vol_coeff": 1.95,
    "vol_decay": 0.965,
    "vol_floor": 0.06,
}

INT_KEYS = {"shock_duration", "spread_base", "spread_shock_extra", "spread_vol_extra"}


@dataclass(frozen=True)
class Score:
    params: dict[str, float | int]
    mean_edge: float
    mean_retail_edge: float
    mean_arb_edge: float
    mean_final_wealth: float


def _round_params(params: dict[str, float | int]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, value in params.items():
        if key in INT_KEYS:
            out[key] = int(value)
        else:
            out[key] = round(float(value), 6)
    out["min_size"] = min(float(out["min_size"]), float(out["base_size"]))
    return out


def _candidate_source(params: dict[str, float | int]) -> str:
    return (
        "from strategies.mean_edge_lab.grind.trend_shock_template import ParametricTrendShockStrategy\n\n"
        "class Strategy(ParametricTrendShockStrategy):\n"
        f"    PARAMS = {json.dumps(params, sort_keys=True)}\n"
    )


def _jitter(base: dict[str, float | int], rng: random.Random) -> dict[str, float | int]:
    p = dict(base)
    p["fill_hit"] = max(0.2, min(1.5, float(base["fill_hit"]) + rng.uniform(-0.20, 0.25)))
    p["fill_decay"] = max(0.25, min(0.8, float(base["fill_decay"]) + rng.uniform(-0.08, 0.08)))
    p["inv_skew"] = max(0.02, min(0.1, float(base["inv_skew"]) + rng.uniform(-0.01, 0.01)))
    p["vol_decay"] = max(0.85, min(0.99, float(base["vol_decay"]) + rng.uniform(-0.03, 0.02)))
    p["trend_decay"] = max(0.55, min(0.95, float(base["trend_decay"]) + rng.uniform(-0.08, 0.08)))
    p["trend_alpha"] = max(0.04, min(0.35, float(base["trend_alpha"]) + rng.uniform(-0.05, 0.06)))
    p["trend_weight"] = max(0.2, min(1.2, float(base["trend_weight"]) + rng.uniform(-0.15, 0.18)))
    p["shock_vol_floor"] = max(0.15, min(0.8, float(base["shock_vol_floor"]) + rng.uniform(-0.08, 0.08)))
    p["shock_trigger_min"] = max(1.2, min(4.0, float(base["shock_trigger_min"]) + rng.uniform(-0.35, 0.45)))
    p["shock_trigger_vol_mult"] = max(2.0, min(6.8, float(base["shock_trigger_vol_mult"]) + rng.uniform(-0.8, 0.9)))
    p["shock_duration"] = max(2, min(10, int(round(float(base["shock_duration"]) + rng.uniform(-2, 2)))))
    p["spread_base"] = max(1, min(4, int(round(float(base["spread_base"]) + rng.uniform(-1, 1)))))
    p["spread_shock_extra"] = max(1, min(5, int(round(float(base["spread_shock_extra"]) + rng.uniform(-1, 1)))))
    p["spread_vol_threshold"] = max(0.5, min(2.0, float(base["spread_vol_threshold"]) + rng.uniform(-0.25, 0.25)))
    p["spread_vol_extra"] = max(0, min(3, int(round(float(base["spread_vol_extra"]) + rng.uniform(-1, 1)))))
    p["vol_floor"] = max(0.02, min(0.3, float(base["vol_floor"]) + rng.uniform(-0.03, 0.06)))
    p["vol_coeff"] = max(0.8, min(2.6, float(base["vol_coeff"]) + rng.uniform(-0.35, 0.35)))
    p["shock_size_mult"] = max(0.15, min(0.95, float(base["shock_size_mult"]) + rng.uniform(-0.16, 0.16)))
    p["base_size"] = max(9.0, min(22.0, float(base["base_size"]) + rng.uniform(-2.8, 3.8)))
    p["min_size"] = max(0.05, min(0.8, float(base["min_size"]) + rng.uniform(-0.08, 0.15)))
    p["max_inventory"] = max(6.0, min(18.0, float(base["max_inventory"]) + rng.uniform(-2.2, 2.6)))
    return _round_params(p)


def _evaluate(
    params: dict[str, float | int],
    *,
    simulations: int,
    workers: int,
    seed_start: int,
    tmp: Path,
    idx: int,
) -> Score:
    path = tmp / f"local_{idx:05d}.py"
    path.write_text(_candidate_source(params), encoding="utf-8")
    batch = run_batch(
        strategy_path=str(path),
        n_simulations=simulations,
        workers=workers,
        seed_start=seed_start,
    )
    return Score(
        params=params,
        mean_edge=batch.mean_edge,
        mean_retail_edge=batch.mean_retail_edge,
        mean_arb_edge=batch.mean_arb_edge,
        mean_final_wealth=batch.mean_final_wealth,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=80, help="Number of local perturbation trials")
    parser.add_argument("--simulations", type=int, default=200, help="Simulation count per trial")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--seed-start", type=int, default=0, help="Simulation seed start")
    parser.add_argument("--seed", type=int, default=20260413, help="RNG seed for local perturbations")
    parser.add_argument(
        "--best-strategy",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/strategy_best_local_trend.py"),
        help="Where to write the best import-based strategy",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/grind/local_search_tuned_trend_results.json"),
        help="Where to write results artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    scores: list[Score] = []
    base = _round_params(dict(BASE_PARAMS))

    with tempfile.TemporaryDirectory(prefix="local-trend-search-") as temp_dir:
        tmp = Path(temp_dir)
        baseline = _evaluate(
            base,
            simulations=args.simulations,
            workers=args.workers,
            seed_start=args.seed_start,
            tmp=tmp,
            idx=0,
        )
        scores.append(baseline)
        best = baseline
        print(
            f"[trial 000/{args.trials:03d}] edge={baseline.mean_edge:+.6f} "
            f"retail={baseline.mean_retail_edge:+.6f} "
            f"arb={baseline.mean_arb_edge:+.6f} (baseline)"
        )

        for i in range(1, args.trials + 1):
            trial = _jitter(best.params, rng)
            score = _evaluate(
                trial,
                simulations=args.simulations,
                workers=args.workers,
                seed_start=args.seed_start,
                tmp=tmp,
                idx=i,
            )
            scores.append(score)
            tag = ""
            if score.mean_edge > best.mean_edge:
                best = score
                tag = "  <-- new best"
            print(
                f"[trial {i:03d}/{args.trials:03d}] edge={score.mean_edge:+.6f} "
                f"retail={score.mean_retail_edge:+.6f} "
                f"arb={score.mean_arb_edge:+.6f}{tag}"
            )

    scores.sort(key=lambda x: x.mean_edge, reverse=True)
    best = scores[0]
    args.best_strategy.parent.mkdir(parents=True, exist_ok=True)
    args.best_strategy.write_text(_candidate_source(best.params), encoding="utf-8")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact = {
        "generated_at": generated_at,
        "trials": args.trials,
        "simulations": args.simulations,
        "workers": args.workers,
        "seed_start": args.seed_start,
        "seed": args.seed,
        "best": {
            "params": best.params,
            "mean_edge": best.mean_edge,
            "mean_retail_edge": best.mean_retail_edge,
            "mean_arb_edge": best.mean_arb_edge,
            "mean_final_wealth": best.mean_final_wealth,
        },
        "top_results": [
            {
                "params": s.params,
                "mean_edge": s.mean_edge,
                "mean_retail_edge": s.mean_retail_edge,
                "mean_arb_edge": s.mean_arb_edge,
                "mean_final_wealth": s.mean_final_wealth,
            }
            for s in scores[:20]
        ],
        "best_strategy_path": str(args.best_strategy),
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\nTop 10 local-search results:")
    for idx, s in enumerate(scores[:10], start=1):
        print(
            f"  {idx:02d}. edge={s.mean_edge:+.6f} "
            f"retail={s.mean_retail_edge:+.6f} "
            f"arb={s.mean_arb_edge:+.6f}"
        )
    print(f"\nBest strategy written to {args.best_strategy}")
    print(f"Artifact written to {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
