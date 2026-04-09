"""Explore distinct strategy directions before parameter fine tuning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

from orderbook_pm_challenge.runner import run_batch


@dataclass(frozen=True)
class Direction:
    name: str
    path: str
    thesis: str


DIRECTIONS = (
    Direction(
        name="v28_baseline",
        path="strategies/iterations/strategy_v28.py",
        thesis="Known strong cancel-all baseline from iterations",
    ),
    Direction(
        name="cancelall_clone",
        path="strategies/mean_edge_lab/directions/strategy_baseline_cancelall.py",
        thesis="Control: local baseline clone to keep comparison stable",
    ),
    Direction(
        name="queue_reprice",
        path="strategies/mean_edge_lab/directions/strategy_queue_reprice.py",
        thesis="Preserve queue priority and replace only when stale",
    ),
    Direction(
        name="regime_onesided",
        path="strategies/mean_edge_lab/directions/strategy_regime_onesided.py",
        thesis="Regime-switch and disable toxic side in stressed states",
    ),
    Direction(
        name="momentum_shield",
        path="strategies/mean_edge_lab/directions/strategy_momentum_shield.py",
        thesis="Use momentum to shield adverse side temporarily",
    ),
    Direction(
        name="retail_capture",
        path="strategies/mean_edge_lab/directions/strategy_retail_capture.py",
        thesis="Tighter spread and larger size for retail flow capture",
    ),
    Direction(
        name="inside_join",
        path="strategies/mean_edge_lab/directions/strategy_inside_join.py",
        thesis="Aggressively join inside spread for stronger retail capture",
    ),
    Direction(
        name="outside_defensive",
        path="strategies/mean_edge_lab/directions/strategy_outside_defensive.py",
        thesis="Quote outside competitor to cut adverse selection",
    ),
    Direction(
        name="ladder_two_level",
        path="strategies/mean_edge_lab/directions/strategy_ladder_two_level.py",
        thesis="Two-level bid/ask ladder for mixed flow capture",
    ),
    Direction(
        name="fill_reactive",
        path="strategies/mean_edge_lab/directions/strategy_fill_reactive.py",
        thesis="Strong post-fill retreat on hit side",
    ),
    Direction(
        name="mean_reversion",
        path="strategies/mean_edge_lab/directions/strategy_mean_reversion.py",
        thesis="Fade short-term competitor momentum",
    ),
    Direction(
        name="pulse_quote",
        path="strategies/mean_edge_lab/directions/strategy_pulse_quote.py",
        thesis="Intermittent quoting cadence to avoid toxic flow",
    ),
    Direction(
        name="parametric_template_default",
        path="strategies/mean_edge_lab/strategy_template.py",
        thesis="Current parametric template with default settings",
    ),
)


def _evaluate(path: str, *, simulations: int, workers: int, seed_start: int) -> dict[str, float]:
    batch = run_batch(
        strategy_path=path,
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


def _parse_seed_starts(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("seed-starts must contain at least one integer")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-simulations", type=int, default=120, help="Simulations per direction in screen stage")
    parser.add_argument("--robust-simulations", type=int, default=200, help="Simulations per robust rerun")
    parser.add_argument("--robust-seed-starts", type=str, default="0,300,600", help="Comma-separated seed starts")
    parser.add_argument("--top-k", type=int, default=3, help="Top directions from screen to rerun robustly")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("strategies/mean_edge_lab/direction_exploration.json"),
        help="Output JSON artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    robust_seed_starts = _parse_seed_starts(args.robust_seed_starts)

    screen_rows: list[dict] = []
    for direction in DIRECTIONS:
        path = Path(direction.path)
        if not path.exists():
            raise FileNotFoundError(f"Direction strategy not found: {direction.path}")
        metrics = _evaluate(
            direction.path,
            simulations=args.screen_simulations,
            workers=args.workers,
            seed_start=robust_seed_starts[0],
        )
        row = {
            "name": direction.name,
            "path": direction.path,
            "thesis": direction.thesis,
            **metrics,
        }
        screen_rows.append(row)
        print(
            f"[screen] {direction.name:<28} "
            f"edge={metrics['mean_edge']:+.6f} "
            f"retail={metrics['mean_retail_edge']:+.6f} "
            f"arb={metrics['mean_arb_edge']:+.6f}"
        )

    screen_rows.sort(key=lambda row: row["mean_edge"], reverse=True)
    top_k = max(1, min(args.top_k, len(screen_rows)))
    finalists = screen_rows[:top_k]

    robust_rows: list[dict] = []
    print("\nRobust reruns for top directions:")
    for finalist in finalists:
        reruns = []
        for seed_start in robust_seed_starts:
            metrics = _evaluate(
                finalist["path"],
                simulations=args.robust_simulations,
                workers=args.workers,
                seed_start=seed_start,
            )
            reruns.append({"seed_start": seed_start, **metrics})
            print(
                f"  {finalist['name']:<24} seed_start={seed_start:<4d} "
                f"edge={metrics['mean_edge']:+.6f}"
            )

        edge_values = [run["mean_edge"] for run in reruns]
        robust_rows.append(
            {
                "name": finalist["name"],
                "path": finalist["path"],
                "thesis": finalist["thesis"],
                "screen_mean_edge": finalist["mean_edge"],
                "robust_mean_edge": mean(edge_values),
                "robust_edge_stddev": pstdev(edge_values) if len(edge_values) > 1 else 0.0,
                "robust_runs": reruns,
            }
        )

    robust_rows.sort(key=lambda row: row["robust_mean_edge"], reverse=True)
    best = robust_rows[0]

    print("\nDirection ranking (robust mean edge):")
    for idx, row in enumerate(robust_rows, start=1):
        print(
            f"  {idx:02d}. {row['name']:<24} "
            f"mean={row['robust_mean_edge']:+.6f} std={row['robust_edge_stddev']:.6f}"
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact = {
        "generated_at": generated_at,
        "screen_simulations": args.screen_simulations,
        "robust_simulations": args.robust_simulations,
        "robust_seed_starts": robust_seed_starts,
        "workers": args.workers,
        "directions": [asdict(direction) for direction in DIRECTIONS],
        "screen_results": screen_rows,
        "robust_results": robust_rows,
        "recommended_direction": best,
    }

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nWrote exploration artifact: {args.results_json}")
    print(f"Recommended direction: {best['name']} ({best['path']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
