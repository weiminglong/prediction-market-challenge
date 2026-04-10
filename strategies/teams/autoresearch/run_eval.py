"""Run strategy simulations and score against eval criteria."""
import json
import subprocess
import sys

STRATEGY_PATH = "strategies/strategy_final.py"
SIMS_PER_RUN = 500
WORKERS = 4
RUNS = 3
# Different seed ranges per run so results are independent
SEED_STARTS = [0, 10000, 20000]

EVALS = [
    ("Profitable", lambda m: m["mean_edge"] > 0),
    ("Strong edge", lambda m: m["mean_edge"] > 0.8),
    ("Very strong edge", lambda m: m["mean_edge"] > 1.2),
    ("Arb controlled", lambda m: m["arb_edge"] > -1.2),
    ("Strong retail", lambda m: m["retail_edge"] > 2.0),
    ("No failures", lambda m: m["failures"] == 0),
]


def run_once(seed_start):
    result = subprocess.run(
        ["uv", "run", "orderbook-pm", "run", STRATEGY_PATH,
         "--simulations", str(SIMS_PER_RUN), "--workers", str(WORKERS),
         "--seed-start", str(seed_start)],
        capture_output=True, text=True, timeout=600,
    )
    output = result.stdout + result.stderr
    metrics = {}
    for line in output.splitlines():
        if "Mean Edge:" in line and "Retail" not in line and "Arb" not in line:
            metrics["mean_edge"] = float(line.split(":")[1].strip())
        elif "Mean Retail Edge:" in line:
            metrics["retail_edge"] = float(line.split(":")[1].strip())
        elif "Mean Arb Edge:" in line:
            metrics["arb_edge"] = float(line.split(":")[1].strip())
        elif "Mean Final Wealth:" in line:
            metrics["final_wealth"] = float(line.split(":")[1].strip())
        elif "Failures:" in line:
            metrics["failures"] = int(line.split(":")[1].strip())
        elif "Successes:" in line:
            metrics["successes"] = int(line.split(":")[1].strip())
    return metrics


def score_run(metrics):
    results = []
    for name, check in EVALS:
        try:
            passed = check(metrics)
        except (KeyError, TypeError):
            passed = False
        results.append((name, passed))
    return results


def main():
    all_results = []
    total_score = 0
    max_score = len(EVALS) * RUNS

    for i in range(RUNS):
        seed = SEED_STARTS[i]
        print(f"Run {i+1}/{RUNS} (seed_start={seed})...", flush=True)
        metrics = run_once(seed)
        print(f"  edge={metrics.get('mean_edge', '?'):.4f}  "
              f"retail={metrics.get('retail_edge', '?'):.4f}  "
              f"arb={metrics.get('arb_edge', '?'):.4f}", flush=True)
        scored = score_run(metrics)
        run_score = sum(1 for _, p in scored if p)
        total_score += run_score
        all_results.append({"metrics": metrics, "evals": scored, "score": run_score})

    pass_rate = (total_score / max_score) * 100 if max_score > 0 else 0

    # Per-eval breakdown
    eval_breakdown = []
    for j, (name, _) in enumerate(EVALS):
        passes = sum(1 for r in all_results if r["evals"][j][1])
        eval_breakdown.append({"name": name, "pass_count": passes, "total": RUNS})

    output = {
        "score": total_score,
        "max_score": max_score,
        "pass_rate": round(pass_rate, 1),
        "runs": all_results,
        "eval_breakdown": eval_breakdown,
    }

    print(f"\nSCORE: {total_score}/{max_score} ({pass_rate:.1f}%)")
    print("EVAL_JSON:" + json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
