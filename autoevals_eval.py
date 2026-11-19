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
from autoevals import Factuality, ClosedQA

sys.path.insert(0, str(Path(__file__).parent))
from rag import retrieve, answer, load_index, CHAT_MODEL

# --- Setup ---
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY in your .env file.")
if not os.getenv("BRAINTRUST_API_KEY"):
    sys.exit("Set BRAINTRUST_API_KEY in your .env file.")

client = OpenAI()
index = load_index()
K = 4

# --- Load test questions ---
with open("data/test_questions.json") as f:
    questions = json.load(f)

data = [
    {"input": item, "expected": item["gold_passage"]}
    for item in questions
]

# --- Task: run RAG pipeline ---
def task(input):
    question = input["question"]
    gold_passage = input["gold_passage"]

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
        "question": question,
        "gold_passage": gold_passage,
    }

# --- autoevals built-in scorers ---
# Factuality: checks if the answer is factually consistent with the expected (gold_passage)
# ClosedQA: checks if the answer correctly answers the question using only the context

openai_api_key = os.getenv("OPENAI_API_KEY")
# Force autoevals to call OpenAI directly, not through Braintrust gateway
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"

def score_factuality(input, output, expected):
    result = Factuality(api_key=openai_api_key)(
        input=output["question"],
        output=output["answer"],
        expected=expected,
    )
    return {"name": "factuality", "score": result.score}

def score_closedqa(input, output, expected):
    result = ClosedQA(api_key=openai_api_key)(
        input=output["question"],
        output=output["answer"],
        context=output["context"],
    )
    return {"name": "closed_qa", "score": result.score}

# --- Run the eval ---
Eval(
    "RAG Eval",
    data=data,
    task=task,
    scores=[score_factuality, score_closedqa],
    metadata={
        "k": K,
        "chat_model": CHAT_MODEL,
        "embed_model": index["embed_model"],
        "scorer": "autoevals",
    },
)
