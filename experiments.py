import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from rag import retrieve, load_index, embed, INDEX_PATH
from eval import compute_recall_at_k, compute_precision_at_k
from judges import judge_groundedness, judge_tone_safety, judge_tool_selection
from verifiers import (
    check_json_schema, check_response_length,
    check_prohibited_keywords, check_citation_present,
)

QUESTIONS_PATH = Path("data/test_questions.json")
RESULTS_DIR = Path("results")
K = 4

# ── System prompts for each variant ───────────────────────────────────────────

SYSTEM_BASELINE = (
    "You answer questions using only the provided context passages. "
    "Cite passages inline like [1], [2]. If the context does not contain the answer, "
    "say you don't know rather than guessing."
)

SYSTEM_COT = (
    "You answer medical questions using only the provided context passages. "
    "Follow these steps before writing your answer:\n"
    "  Step 1: Identify the patient's primary symptoms.\n"
    "  Step 2: Find the best-matching specialist in the context.\n"
    "  Step 3: Write a concise, empathetic recommendation citing sources like [1].\n"
    "If the context does not contain the answer, say you don't know. "
    "Show your reasoning briefly before the final recommendation."
)

FEW_SHOT_EXAMPLES = """
Example 1:
Patient: I have chest pain and shortness of breath.
Context: [1] Dr. Arthur Pendelton — Cardiology. Specializes in chest pain and heart disease.
Answer: Based on your symptoms, I recommend seeing Dr. Arthur Pendelton [1], a Cardiologist who specializes in chest pain and heart disease. Please seek care promptly.

Example 2:
Patient: I have a rash on my arms and dry, flaky skin.
Context: [1] Dr. Sophia Chen — Dermatology. Specializes in skin rashes, eczema, and psoriasis.
Answer: Your symptoms suggest a skin condition. I recommend Dr. Sophia Chen [1], a Dermatologist who can evaluate and treat your rash.

Example 3:
Patient: I feel very anxious and cannot sleep.
Context: [1] Dr. Rachel Green — Psychiatry. Specializes in anxiety, depression, and insomnia.
Answer: I understand how difficult this must feel. Dr. Rachel Green [1], a Psychiatrist, can help with anxiety and sleep difficulties.
"""

SYSTEM_FEW_SHOT = (
    "You answer medical questions using only the provided context passages. "
    "Always cite passages inline like [1], [2]. Be empathetic and professional. "
    "Here are examples of ideal answers:\n"
    + FEW_SHOT_EXAMPLES
    + "\nNow answer the patient's question following the same style."
)

VARIANTS = {
    "A": {"name": "baseline",   "model": "gpt-4o-mini", "system": SYSTEM_BASELINE},
    "B": {"name": "chain_of_thought", "model": "gpt-4o-mini", "system": SYSTEM_COT},
    "C": {"name": "few_shot",   "model": "gpt-4o",      "system": SYSTEM_FEW_SHOT},
}


def answer_variant(client: OpenAI, index: dict, question: str, system: str, model: str):
    """Run the RAG answer pipeline for a given variant. Returns (answer_text, usage, latency)."""
    hits = retrieve(client, index, question, k=K)
    context = "\n\n".join(
        f"[{i}] (source: {src})\n{text}"
        for i, (src, text, _) in enumerate(hits, 1)
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    latency = time.time() - t0

    answer_text = resp.choices[0].message.content.strip()
    usage = {
        "prompt_tokens":     resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens":      resp.usage.total_tokens,
    }
    return answer_text, context, hits, usage, latency


def evaluate_one(client, index, item, system, model):
    question     = item["question"]
    gold_passage = item["gold_passage"]

    answer_text, context, hits, usage, latency = answer_variant(
        client, index, question, system, model
    )

    output = {
        "answer":       answer_text,
        "context":      context,
        "question":     question,
        "gold_passage": gold_passage,
    }

    # Deterministic verifiers
    v_schema    = check_json_schema(output)
    v_length    = check_response_length(output)
    v_prohibit  = check_prohibited_keywords(output)
    v_citation  = check_citation_present(output)

    # Retrieval metrics
    recall    = compute_recall_at_k(hits, gold_passage)
    precision = compute_precision_at_k(hits, gold_passage)

    # LLM judges
    j_ground = judge_groundedness(client, context, answer_text)
    j_tone   = judge_tone_safety(client, question, answer_text)
    j_tool   = judge_tool_selection(client, answer_text, gold_passage)

    return {
        "id":           item["id"],
        "question":     question,
        "answer":       answer_text,
        # retrieval
        "recall_at_k":    recall,
        "precision_at_k": precision,
        # verifiers
        "json_schema":           v_schema["score"],
        "response_length":       v_length["score"],
        "prohibited_keywords":   v_prohibit["score"],
        "citation_present":      v_citation["score"],
        # judges
        "groundedness":    j_ground["score"],
        "tone_safety":     j_tone["score"],
        "tool_selection":  j_tool["score"],
        # cost / latency
        "latency_s":          round(latency, 3),
        "prompt_tokens":      usage["prompt_tokens"],
        "completion_tokens":  usage["completion_tokens"],
        "total_tokens":       usage["total_tokens"],
    }


def run_variant(variant_key: str):
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")

    cfg = VARIANTS[variant_key]
    print(f"\n{'='*60}")
    print(f"Variant {variant_key}: {cfg['name'].upper()}  (model={cfg['model']})")
    print(f"{'='*60}\n")

    client = OpenAI()
    index  = load_index()

    with QUESTIONS_PATH.open() as f:
        questions = json.load(f)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"  [{i:2d}/{len(questions)}] {item['question'][:55]}...")
        r = evaluate_one(client, index, item, cfg["system"], cfg["model"])
        results.append(r)
        print(
            f"         recall={r['recall_at_k']:.2f}  prec={r['precision_at_k']:.2f}  "
            f"gnd={int(r['groundedness'])}  tone={int(r['tone_safety'])}  "
            f"sel={int(r['tool_selection'])}  lat={r['latency_s']:.1f}s"
        )

    # Summary
    metric_cols = [
        "recall_at_k", "precision_at_k",
        "json_schema", "response_length", "prohibited_keywords", "citation_present",
        "groundedness", "tone_safety", "tool_selection",
    ]
    summary = {col: round(sum(r[col] for r in results) / len(results), 4) for col in metric_cols}
    summary["avg_latency_s"]          = round(sum(r["latency_s"] for r in results) / len(results), 3)
    summary["avg_total_tokens"]       = round(sum(r["total_tokens"] for r in results) / len(results), 1)
    summary["total_prompt_tokens"]    = sum(r["prompt_tokens"] for r in results)
    summary["total_completion_tokens"]= sum(r["completion_tokens"] for r in results)

    print(f"\n--- Variant {variant_key} Summary ---")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    output = {
        "variant": variant_key,
        "config": {
            "name":   cfg["name"],
            "model":  cfg["model"],
            "k":      K,
            "system": cfg["system"][:120] + "...",
        },
        "summary": summary,
        "results": results,
    }

    out_path = RESULTS_DIR / f"experiment_{variant_key}_{cfg['name']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    return output


def main():
    p = argparse.ArgumentParser(description="Run experiment variants A, B, C.")
    p.add_argument(
        "--variant",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which variant to run (default: all)",
    )
    args = p.parse_args()

    keys = ["A", "B", "C"] if args.variant == "all" else [args.variant]
    for k in keys:
        run_variant(k)

    print("\nAll requested variants complete.")


if __name__ == "__main__":
    main()
