import argparse
import json
import os
from pathlib import Path

RESULTS_DIR = Path("results")
REPORT_PATH = RESULTS_DIR / "analysis_report.json"

# OpenAI pricing (per 1K tokens, approximate as of mid-2024)
PRICE_PER_1K = {
    "gpt-4o-mini": {"prompt": 0.000150, "completion": 0.000600},
    "gpt-4o":      {"prompt": 0.005000, "completion": 0.015000},
}

METRIC_COLS = [
    "recall_at_k", "precision_at_k",
    "json_schema", "response_length", "prohibited_keywords", "citation_present",
    "groundedness", "tone_safety", "tool_selection",
]


def load_variant(variant_key: str) -> dict | None:
    for path in RESULTS_DIR.glob(f"experiment_{variant_key}_*.json"):
        with path.open() as f:
            return json.load(f)
    return None


def estimate_cost(summary: dict, model: str) -> float:
    prices = PRICE_PER_1K.get(model, {"prompt": 0.001, "completion": 0.002})
    prompt_cost = summary.get("total_prompt_tokens", 0) / 1000 * prices["prompt"]
    completion_cost = summary.get("total_completion_tokens", 0) / 1000 * prices["completion"]
    return round(prompt_cost + completion_cost, 6)


def diff_tag(val_new: float, val_base: float, threshold: float = 0.02) -> str:
    delta = val_new - val_base
    if delta > threshold:
        return f"+{delta:+.3f} ▲ IMPROVED"
    elif delta < -threshold:
        return f"{delta:+.3f} ▼ REGRESSION"
    else:
        return f"{delta:+.3f}   (same)"


def print_table(headers: list, rows: list):
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


def run_analysis(baseline_key: str = "A"):
    variants = {}
    for key in ["A", "B", "C"]:
        data = load_variant(key)
        if data:
            variants[key] = data
        else:
            print(f"Warning: results for Variant {key} not found in {RESULTS_DIR}/")

    if not variants:
        print("No experiment results found. Run: python experiments.py")
        return

    print("\n" + "=" * 70)
    print("  RAG EXPERIMENT — COMPARATIVE ANALYSIS")
    print("=" * 70)

    # ── Metric comparison table ────────────────────────────────────────────────
    print("\n--- Metric Comparison (mean over all questions) ---\n")
    keys_present = sorted(variants.keys())
    headers = ["Metric"] + [f"Variant {k} ({variants[k]['config']['name']})" for k in keys_present]
    rows = []
    for metric in METRIC_COLS:
        row = [metric]
        for k in keys_present:
            val = variants[k]["summary"].get(metric, "N/A")
            row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        rows.append(row)
    print_table(headers, rows)

    # ── Cost & Latency ─────────────────────────────────────────────────────────
    print("\n--- Cost & Latency ---\n")
    cost_rows = []
    for k in keys_present:
        s = variants[k]["summary"]
        model = variants[k]["config"]["model"]
        cost = estimate_cost(s, model)
        cost_rows.append([
            f"Variant {k}",
            model,
            f"{s.get('avg_latency_s', 0):.2f}s",
            str(s.get("avg_total_tokens", "?")),
            f"${cost:.4f}",
        ])
    print_table(["Variant", "Model", "Avg Latency", "Avg Tokens", "Est. Total Cost"], cost_rows)

    # ── Regressions & Improvements vs baseline ────────────────────────────────
    if baseline_key in variants:
        base_summary = variants[baseline_key]["summary"]
        print(f"\n--- Changes vs Variant {baseline_key} (baseline) ---\n")

        regressions = []
        improvements = []
        diff_rows = []

        for k in keys_present:
            if k == baseline_key:
                continue
            for metric in METRIC_COLS:
                base_val = base_summary.get(metric, 0)
                new_val  = variants[k]["summary"].get(metric, 0)
                if not isinstance(base_val, (int, float)):
                    continue
                tag = diff_tag(new_val, base_val)
                diff_rows.append([f"Var {k}", metric, f"{base_val:.3f}", f"{new_val:.3f}", tag])
                if "REGRESSION" in tag:
                    regressions.append({"variant": k, "metric": metric,
                                        "baseline": base_val, "new": new_val})
                elif "IMPROVED" in tag:
                    improvements.append({"variant": k, "metric": metric,
                                         "baseline": base_val, "new": new_val})

        print_table(["Variant", "Metric", "Baseline", "New", "Change"], diff_rows)

        if regressions:
            print(f"\n  Regressions detected ({len(regressions)}):")
            for r in regressions:
                print(f"    Var {r['variant']}  {r['metric']}: {r['baseline']:.3f} → {r['new']:.3f}")
        else:
            print("\n  No regressions detected.")

        if improvements:
            print(f"\n  Improvements detected ({len(improvements)}):")
            for r in improvements:
                print(f"    Var {r['variant']}  {r['metric']}: {r['baseline']:.3f} → {r['new']:.3f}")

    # ── Winner recommendation ──────────────────────────────────────────────────
    print("\n--- Winner Recommendation ---\n")

    def composite_score(summary: dict) -> float:
        """Weighted composite: quality metrics at 70%, cost/speed penalty at 30%."""
        quality = sum(summary.get(m, 0) for m in METRIC_COLS) / len(METRIC_COLS)
        # Normalize latency penalty: lower is better, scale 0-1
        lat = summary.get("avg_latency_s", 1)
        lat_norm = max(0, 1 - lat / 10)  # 10s = worst
        return round(0.7 * quality + 0.3 * lat_norm, 4)

    scores = {k: composite_score(variants[k]["summary"]) for k in keys_present}
    winner = max(scores, key=scores.get)
    print(f"  Composite scores (70% quality + 30% latency):")
    for k, s in sorted(scores.items(), key=lambda x: -x[1]):
        cfg = variants[k]["config"]
        marker = "  ← WINNER" if k == winner else ""
        print(f"    Variant {k} ({cfg['name']}, {cfg['model']}): {s:.4f}{marker}")

    winner_cfg = variants[winner]["config"]
    print(f"\n  Recommended configuration: Variant {winner}")
    print(f"    Model  : {winner_cfg['model']}")
    print(f"    Strategy: {winner_cfg['name']}")
    print(f"    Reason : Best balance of answer quality and response speed.\n")

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "baseline": baseline_key,
        "variants_analyzed": keys_present,
        "metric_summary": {k: variants[k]["summary"] for k in keys_present},
        "regressions":  regressions if baseline_key in variants else [],
        "improvements": improvements if baseline_key in variants else [],
        "composite_scores": scores,
        "winner": winner,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Full analysis report saved → {REPORT_PATH}")
    return report


def main():
    p = argparse.ArgumentParser(description="Analyze experiment variant results.")
    p.add_argument("--baseline", default="A", choices=["A", "B", "C"],
                   help="Variant to use as baseline for diff (default: A)")
    args = p.parse_args()
    run_analysis(baseline_key=args.baseline)


if __name__ == "__main__":
    main()
