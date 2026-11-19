import re

# ── Configuration ──────────────────────────────────────────────────────────────

REQUIRED_OUTPUT_KEYS = {"answer", "context", "question"}

MIN_WORDS = 5
MAX_WORDS = 300

PROHIBITED_KEYWORDS = [
    r"\bi don'?t know\b",          # generic "I don't know" (allow "I don't know based on context")
    r"\bas an ai\b",
    r"\bi cannot help\b",
    r"\bcontact a (doctor|physician|specialist)\b",  # RAG should recommend, not deflect
    r"\bconsult a professional\b",
    r"\bi'?m not (a )?(doctor|medical|licensed)\b",
]

CITATION_PATTERN = re.compile(r"\[\d+\]")

# ── Verifier functions ─────────────────────────────────────────────────────────


def check_json_schema(output: dict) -> dict:
    """Verify that the task output dict has all required keys."""
    missing = REQUIRED_OUTPUT_KEYS - set(output.keys())
    if missing:
        return {
            "name": "json_schema",
            "score": 0,
            "reason": f"Missing required keys: {sorted(missing)}",
        }
    return {"name": "json_schema", "score": 1, "reason": "All required keys present"}


def check_response_length(output: dict) -> dict:
    """Verify the answer text is within the allowed word-count range."""
    answer_text = output.get("answer", "")
    word_count = len(answer_text.split())
    if word_count < MIN_WORDS:
        return {
            "name": "response_length",
            "score": 0,
            "reason": f"Answer too short: {word_count} words (min {MIN_WORDS})",
        }
    if word_count > MAX_WORDS:
        return {
            "name": "response_length",
            "score": 0,
            "reason": f"Answer too long: {word_count} words (max {MAX_WORDS})",
        }
    return {
        "name": "response_length",
        "score": 1,
        "reason": f"Answer length OK: {word_count} words",
    }


def check_prohibited_keywords(output: dict) -> dict:
    """Verify the answer contains none of the prohibited phrases."""
    answer_text = output.get("answer", "").lower()
    for pattern in PROHIBITED_KEYWORDS:
        m = re.search(pattern, answer_text)
        if m:
            return {
                "name": "prohibited_keywords",
                "score": 0,
                "reason": f"Found prohibited phrase matching: '{pattern}' → '{m.group()}'",
            }
    return {
        "name": "prohibited_keywords",
        "score": 1,
        "reason": "No prohibited keywords found",
    }


def check_citation_present(output: dict) -> dict:
    """Verify the answer contains at least one inline citation like [1]."""
    answer_text = output.get("answer", "")
    if CITATION_PATTERN.search(answer_text):
        return {
            "name": "citation_present",
            "score": 1,
            "reason": "At least one citation found",
        }
    return {
        "name": "citation_present",
        "score": 0,
        "reason": "No inline citations [n] found in answer",
    }


# ── Braintrust-compatible scorer wrappers ─────────────────────────────────────
# Each accepts (input, output, expected) to match Braintrust scorer signature.


def score_json_schema(input, output, expected):
    r = check_json_schema(output)
    return {"name": r["name"], "score": r["score"]}


def score_response_length(input, output, expected):
    r = check_response_length(output)
    return {"name": r["name"], "score": r["score"]}


def score_prohibited_keywords(input, output, expected):
    r = check_prohibited_keywords(output)
    return {"name": r["name"], "score": r["score"]}


def score_citation_present(input, output, expected):
    r = check_citation_present(output)
    return {"name": r["name"], "score": r["score"]}


ALL_VERIFIERS = [
    score_json_schema,
    score_response_length,
    score_prohibited_keywords,
    score_citation_present,
]


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    good = {
        "answer": "Based on your symptoms you should see Dr. Arthur Pendelton [1], a cardiologist.",
        "context": "[1] (source: doctors.pdf)\nDr. Arthur Pendelton — Cardiology",
        "question": "I have chest pain.",
        "gold_passage": "Dr. Arthur Pendelton",
    }

    bad_missing_key = {"answer": "See Dr. Smith.", "context": "ctx"}
    bad_too_short = {"answer": "Yes.", "context": "ctx", "question": "?"}
    bad_prohibited = {
        "answer": "As an AI I cannot help with that.",
        "context": "ctx",
        "question": "?",
    }
    bad_no_citation = {
        "answer": "You should see Dr. Arthur Pendelton for your chest pain.",
        "context": "ctx",
        "question": "?",
    }

    for label, sample in [
        ("GOOD output", good),
        ("Missing key", bad_missing_key),
        ("Too short", bad_too_short),
        ("Prohibited kw", bad_prohibited),
        ("No citation", bad_no_citation),
    ]:
        print(f"\n--- {label} ---")
        for fn in [check_json_schema, check_response_length,
                   check_prohibited_keywords, check_citation_present]:
            r = fn(sample)
            status = "PASS" if r["score"] else "FAIL"
            print(f"  [{status}] {r['name']}: {r['reason']}")
