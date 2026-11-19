import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from rag import retrieve, answer, load_index, CHAT_MODEL

QUESTIONS_PATH = Path("data/test_questions.json")
RESULTS_DIR = Path("results")
DEFAULT_OUT = RESULTS_DIR / "eval_report.json"


def compute_recall_at_k(hits: list, gold_passage: str) -> float:
    """Return 1.0 if gold_passage appears in any retrieved chunk, else 0.0."""
    return 1.0 if any(gold_passage.lower() in text.lower() for _, text, _ in hits) else 0.0


def compute_precision_at_k(hits: list, gold_passage: str) -> float:
    """Return fraction of retrieved chunks that contain the gold passage."""
    if not hits:
        return 0.0
    relevant = sum(gold_passage.lower() in text.lower() for _, text, _ in hits)
    return relevant / len(hits)


def judge_faithfulness(client: OpenAI, context: str, generated_answer: str) -> bool:
    """Ask the LLM whether the answer is fully grounded in the provided context."""
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{generated_answer}\n\n"
        "Is the answer fully grounded in the context above? Reply YES or NO only."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict faithfulness judge. Determine whether a given answer "
                    "is fully supported by the provided context passages. An answer is faithful "
                    "if every claim in it can be directly verified from the context. "
                    "Reply with exactly one word: YES or NO."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")


def judge_answer_relevancy(client: OpenAI, question: str, generated_answer: str) -> bool:
    """Ask the LLM whether the answer actually addresses the question asked."""
    prompt = (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {generated_answer}\n\n"
        "Does the answer address the question? Reply YES or NO only."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevancy judge. Determine whether an answer actually addresses "
                    "the question asked. An answer is relevant if it directly responds to what "
                    "was asked, even if the answer is 'I don't know.' "
                    "Reply with exactly one word: YES or NO."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")


def evaluate_question(client: OpenAI, index: dict, item: dict, k: int = 4) -> dict:
    """Run all four metrics for a single Q&A pair.

    Args:
        client: OpenAI client instance.
        index: Loaded RAG index dict from load_index().
        item: One entry from test_questions.json.
        k: Number of chunks to retrieve.

    Returns:
        Dict with id, question, recall_at_k, precision_at_k,
        faithfulness, answer_relevancy, generated_answer.
    """
    question = item["question"]
    gold_passage = item["gold_passage"]

    hits = retrieve(client, index, question, k=k)

    # Build context string the same way answer() does, for the faithfulness judge.
    context = "\n\n".join(
        f"[{i}] (source: {src})\n{text}"
        for i, (src, text, _) in enumerate(hits, 1)
    )

    # Get the full response then strip the "Sources:" footer.
    full_response = answer(client, index, question)
    answer_text = full_response.split("\n\nSources:")[0]

    recall = compute_recall_at_k(hits, gold_passage)
    precision = compute_precision_at_k(hits, gold_passage)
    faithful = judge_faithfulness(client, context, answer_text)
    relevant = judge_answer_relevancy(client, question, answer_text)

    return {
        "id": item["id"],
        "question": question,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "faithfulness": float(faithful),
        "answer_relevancy": float(relevant),
        "generated_answer": answer_text,
    }


def run_evaluation(
    questions_path: Path = QUESTIONS_PATH,
    out_path: Path = DEFAULT_OUT,
    k: int = 2,
) -> dict:
    """Run evaluation across all questions and save a JSON report."""
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first (or add it to .env).")

    client = OpenAI()
    index = load_index()

    if not questions_path.exists():
        sys.exit(f"Questions file not found: {questions_path}")
    with questions_path.open() as f:
        questions = json.load(f)

    print(f"Running evaluation on {len(questions)} questions (k={k})...\n")
    results = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Q{item['id']}: {item['question'][:60]}...")
        result = evaluate_question(client, index, item, k=k)
        results.append(result)
        print(
            f"  recall@{k}={result['recall_at_k']:.2f}  "
            f"precision@{k}={result['precision_at_k']:.2f}  "
            f"faithful={int(result['faithfulness'])}  "
            f"relevant={int(result['answer_relevancy'])}"
        )

    df = pd.DataFrame(results)
    metric_cols = ["recall_at_k", "precision_at_k", "faithfulness", "answer_relevancy"]
    summary = {col: float(df[col].mean()) for col in metric_cols}

    output = {
        "config": {
            "k": k,
            "embed_model": index["embed_model"],
            "chat_model": CHAT_MODEL,
            "num_questions": len(questions),
        },
        "summary": summary,
        "results": df.to_dict(orient="records"),
    }

    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSummary:")
    for metric, val in summary.items():
        print(f"  {metric:<20} {val:.2f}")
    print(f"\nReport saved to {out_path}")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RAG pipeline.")
    p.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve (default: 4)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output path for JSON report")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH,
                   help="Path to test_questions.json")
    args = p.parse_args()
    run_evaluation(questions_path=args.questions, out_path=args.out, k=args.k)


if __name__ == "__main__":
    main()
