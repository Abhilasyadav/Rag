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
from datasets import Dataset
from ragas import evaluate
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

sys.path.insert(0, str(Path(__file__).parent))
from rag import retrieve, answer, load_index, CHAT_MODEL

# --- Setup ---
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY in your .env file.")

client = OpenAI()
index = load_index()
K = 4

# --- Load test questions ---
with open("data/test_questions.json") as f:
    questions = json.load(f)

# --- Run RAG pipeline for each question ---
print(f"Running RAG pipeline for {len(questions)} questions...\n")

data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for i, item in enumerate(questions, 1):
    question = item["question"]
    gold_passage = item["gold_passage"]

    print(f"[{i}/{len(questions)}] {question[:60]}...")

    # Retrieve chunks
    hits = retrieve(client, index, question, k=K)
    contexts = [text for _, text, _ in hits]

    # Generate answer
    full_response = answer(client, index, question)
    answer_text = full_response.split("\n\nSources:")[0]

    data["question"].append(question)
    data["answer"].append(answer_text)
    data["contexts"].append(contexts)
    data["ground_truth"].append(gold_passage)

# --- Convert to RAGAS Dataset ---
dataset = Dataset.from_dict(data)

# --- Run RAGAS Evaluation ---
print("\nRunning RAGAS evaluation...\n")
result = evaluate(
    dataset=dataset,
    metrics=metrics,
)

# --- Print Results ---
print("\n=== RAGAS Results ===")
print(result)

# --- Save to JSON ---
output_path = Path("results/ragas_report.json")
output_path.parent.mkdir(exist_ok=True)

df = result.to_pandas()
result_dict = {
    "faithfulness": float(df["faithfulness"].mean()),
    "answer_relevancy": float(df["answer_relevancy"].mean()),
    "context_recall": float(df["context_recall"].mean()),
    "context_precision": float(df["context_precision"].mean()),
}
with output_path.open("w") as f:
    json.dump(result_dict, f, indent=2)

print(f"\nReport saved to {output_path}")
