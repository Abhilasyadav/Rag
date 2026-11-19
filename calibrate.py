import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from judges import judge_groundedness, judge_tone_safety, judge_tool_selection

CALIBRATION_PATH = Path("data/calibration_examples.json")
RESULTS_PATH = Path("results/calibration_report.json")

TARGET_PRECISION = 0.85


def compute_metrics(predictions: list[int], labels: list[int]) -> dict:
    """Compute TP, FP, FN, TN, precision, recall, accuracy."""
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))
    total = len(predictions)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "accuracy":  round(accuracy, 4),
        "n": total,
    }


def run_calibration():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")

    client = OpenAI()

    with CALIBRATION_PATH.open() as f:
        examples = json.load(f)

    print(f"Running calibration on {len(examples)} human-labeled examples...\n")

    # Collect per-judge predictions
    judge_names = ["groundedness", "tone_safety", "tool_selection"]
    preds  = {j: [] for j in judge_names}
    labels = {j: [] for j in judge_names}
    per_example = []

    for ex in examples:
        g_result = judge_groundedness(client, ex["context"], ex["answer"])
        t_result = judge_tone_safety(client, ex["question"], ex["answer"])
        s_result = judge_tool_selection(client, ex["answer"], ex["expected_doctor"])

        row = {
            "id": ex["id"],
            "question": ex["question"][:60],
            "groundedness_pred":   int(g_result["score"]),
            "groundedness_label":  ex["human_labels"]["groundedness"],
            "tone_safety_pred":    int(t_result["score"]),
            "tone_safety_label":   ex["human_labels"]["tone_safety"],
            "tool_selection_pred": int(s_result["score"]),
            "tool_selection_label":ex["human_labels"]["tool_selection"],
        }
        per_example.append(row)

        for jname, result, human_key in [
            ("groundedness",   g_result, "groundedness"),
            ("tone_safety",    t_result, "tone_safety"),
            ("tool_selection", s_result, "tool_selection"),
        ]:
            preds[jname].append(int(result["score"]))
            labels[jname].append(ex["human_labels"][human_key])

        print(
            f"[{ex['id']:2d}] "
            f"GND={int(g_result['score'])}/h{ex['human_labels']['groundedness']}  "
            f"TON={int(t_result['score'])}/h{ex['human_labels']['tone_safety']}  "
            f"SEL={int(s_result['score'])}/h{ex['human_labels']['tool_selection']}"
        )

    # Compute metrics per judge
    print("\n=== Calibration Report ===\n")
    summary = {}
    all_pass = True
    for jname in judge_names:
        m = compute_metrics(preds[jname], labels[jname])
        summary[jname] = m
        meets = "✓ PASS" if m["precision"] >= TARGET_PRECISION else "✗ FAIL"
        print(
            f"  {jname:<20} "
            f"precision={m['precision']:.2%}  "
            f"recall={m['recall']:.2%}  "
            f"accuracy={m['accuracy']:.2%}  "
            f"[{meets}]"
        )
        if m["precision"] < TARGET_PRECISION:
            all_pass = False

    print()
    if all_pass:
        print("All judges meet the ≥85% precision target.")
    else:
        print("WARNING: One or more judges are below the 85% precision target.")
        print("Consider revising the judge prompts in judges.py.")

    # Save report
    output = {
        "target_precision": TARGET_PRECISION,
        "all_pass": all_pass,
        "summary": summary,
        "per_example": per_example,
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nReport saved to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    run_calibration()
