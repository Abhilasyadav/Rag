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
from braintrust import Eval

sys.path.insert(0, str(Path(__file__).parent))
from rag import retrieve, answer, load_index, CHAT_MODEL
from eval import (
    compute_recall_at_k,
    compute_precision_at_k,
    judge_faithfulness,
    judge_answer_relevancy,
)
from verifiers import ALL_VERIFIERS
from judges import score_groundedness, score_tone_safety, score_tool_selection

# --- Setup ---
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY in your .env file.")
if not os.getenv("BRAINTRUST_API_KEY"):
    sys.exit("Set BRAINTRUST_API_KEY in your .env file.")

client = OpenAI()
index = load_index()
K = 4

# --- Load test questions as Braintrust dataset ---
with open("data/test_questions.json") as f:
    questions = json.load(f)

data = [
    {"input": item, "expected": item["gold_passage"]}
    for item in questions
]

# --- Task: run the RAG pipeline for one question ---
def task(input):
    item = input
    question = item["question"]
    gold_passage = item["gold_passage"]

    hits = retrieve(client, index, question, k=K)
    context = "\n\n".join(
        f"[{i}] (source: {src})\n{text}"
        for i, (src, text, _) in enumerate(hits, 1)
    )
    full_response = answer(client, index, question)
    answer_text = full_response.split("\n\nSources:")[0]

    return {
        "answer": answer_text,
        "context": context,
        "hits": hits,
        "gold_passage": gold_passage,
        "question": question,
    }

# --- Scorers: one function per metric ---
def score_recall(input, output, expected):
    return {"name": "recall_at_k", "score": compute_recall_at_k(output["hits"], output["gold_passage"])}

def score_precision(input, output, expected):
    return {"name": "precision_at_k", "score": compute_precision_at_k(output["hits"], output["gold_passage"])}

def score_faithfulness(input, output, expected):
    result = judge_faithfulness(client, output["context"], output["answer"])
    return {"name": "faithfulness", "score": float(result)}

def score_relevancy(input, output, expected):
    result = judge_answer_relevancy(client, output["question"], output["answer"])
    return {"name": "answer_relevancy", "score": float(result)}

# --- Run the eval ---
Eval(
    "RAG Eval",   # project name on Braintrust dashboard
    data=data,
    task=task,
    scores=[
        # Original eval.py metrics
        score_recall, score_precision, score_faithfulness, score_relevancy,
        # Task 01 — deterministic verifiers
        *ALL_VERIFIERS,
        # Task 02 — LLM judges
        score_groundedness, score_tone_safety, score_tool_selection,
    ],
    metadata={
        "k": K,
        "chat_model": CHAT_MODEL,
        "embed_model": index["embed_model"],
    },
)
