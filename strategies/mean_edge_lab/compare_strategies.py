"""Compare strategy files across multiple seed ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from orderbook_pm_challenge.runner import run_batch


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


def _load_paths(args: argparse.Namespace) -> list[str]:
    paths = list(args.strategy_paths)
    if args.strategy_list_file is not None:
        for line in args.strategy_list_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    deduped = []
    seen = set()
    for path in paths:
        normalized = str(Path(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    if not deduped:
        raise ValueError("No strategy paths provided")
    return deduped


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy_paths", nargs="*", help="Strategy file paths to compare")
    parser.add_argument(
        "--strategy-list-file",
        type=Path,
        default=None,
        help="Optional file with one strategy path per line",
    )
    parser.add_argument("--simulations", type=int, default=200, help="Simulations per seed start")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--seed-starts", type=str, default="0,300,600", help="Comma-separated seed starts")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox mode")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    strategy_paths = _load_paths(args)
    seed_starts = _parse_seed_starts(args.seed_starts)

    rows = []
    for strategy_path in strategy_paths:
        path = Path(strategy_path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy path not found: {strategy_path}")

        runs = []
        edges = []
        for seed_start in seed_starts:
            batch = run_batch(
                strategy_path=str(path),
                n_simulations=args.simulations,
                workers=args.workers,
                seed_start=seed_start,
                sandbox=args.sandbox,
            )
            run = {
                "seed_start": seed_start,
                "mean_edge": batch.mean_edge,
                "mean_retail_edge": batch.mean_retail_edge,
                "mean_arb_edge": batch.mean_arb_edge,
                "mean_final_wealth": batch.mean_final_wealth,
                "failure_count": batch.failure_count,
            }
            runs.append(run)
            edges.append(batch.mean_edge)
            print(
                f"{path.name} seed_start={seed_start} "
                f"edge={batch.mean_edge:+.6f} retail={batch.mean_retail_edge:+.6f} "
                f"arb={batch.mean_arb_edge:+.6f}"
            )

        rows.append(
            {
                "path": str(path),
                "seed0_mean_edge": runs[0]["mean_edge"],
                "robust_mean_edge": mean(edges),
                "robust_edge_stddev": pstdev(edges) if len(edges) > 1 else 0.0,
                "runs": runs,
            }
        )

    by_seed0 = sorted(rows, key=lambda row: row["seed0_mean_edge"], reverse=True)
    by_robust = sorted(rows, key=lambda row: row["robust_mean_edge"], reverse=True)

    print("\nRank by seed_start[0]:")
    for idx, row in enumerate(by_seed0, start=1):
        print(
            f"  {idx:02d}. {Path(row['path']).name:<44} "
            f"seed0={row['seed0_mean_edge']:+.6f} robust={row['robust_mean_edge']:+.6f}"
        )

    print("\nRank by robust mean:")
    for idx, row in enumerate(by_robust, start=1):
        print(
            f"  {idx:02d}. {Path(row['path']).name:<44} "
            f"robust={row['robust_mean_edge']:+.6f} std={row['robust_edge_stddev']:.6f}"
        )

    if args.output_json is not None:
        payload = {
            "simulations": args.simulations,
            "workers": args.workers,
            "seed_starts": seed_starts,
            "sandbox": args.sandbox,
            "rank_by_seed0": by_seed0,
            "rank_by_robust": by_robust,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
