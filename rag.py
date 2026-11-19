import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

INDEX_PATH = Path("index.pkl")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
CHUNK_WORDS = 100
OVERLAP_WORDS = 20
TOP_K = 4
EMBED_BATCH = 96

SYSTEM = (
    "You answer questions using only the provided context passages. "
    "Cite passages inline like [1], [2]. If the context does not contain the answer, "
    "say you don't know rather than guessing."
)


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str) -> list[str]:
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), step)]


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + EMBED_BATCH])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.array(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def ingest(client: OpenAI, folder: str) -> None:
    chunks, sources = [], []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            for c in chunk(read_file(path)):
                chunks.append(c)
                sources.append(path.name)
    if not chunks:
        sys.exit(f"No .txt/.md/.pdf content found in {folder}")
    with INDEX_PATH.open("wb") as f:
        pickle.dump({"vectors": embed(client, chunks), "chunks": chunks, "sources": sources,
                     "embed_model": EMBED_MODEL}, f)
    print(f"Indexed {len(chunks)} chunks from {len(set(sources))} files -> {INDEX_PATH}")


def retrieve(client: OpenAI, index: dict, query: str, k: int = TOP_K):
    q = embed(client, [query])[0]
    scores = index["vectors"] @ q
    top = scores.argsort()[::-1][:k]
    return [(index["sources"][i], index["chunks"][i], float(scores[i])) for i in top]


def answer(client: OpenAI, index: dict, question: str) -> str:
    hits = retrieve(client, index, question)
    context = "\n\n".join(f"[{i}] (source: {src})\n{text}" for i, (src, text, _) in enumerate(hits, 1))
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    sources = "\n".join(f"  [{i}] {src} (score {s:.2f})" for i, (src, _, s) in enumerate(hits, 1))
    return f"{resp.choices[0].message.content}\n\nSources:\n{sources}"


def load_index() -> dict:
    if not INDEX_PATH.exists():
        sys.exit("No index found. Run: python rag.py ingest <folder>")
    with INDEX_PATH.open("rb") as f:
        index = pickle.load(f)
    if index.get("embed_model") != EMBED_MODEL:
        sys.exit("Index was built with a different embedding model; re-run ingest.")
    return index


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("folder")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("chat")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    client = OpenAI()

    if args.cmd == "ingest":
        ingest(client, args.folder)
        return
    index = load_index()
    if args.cmd == "ask":
        print(answer(client, index, args.question))
    else:
        print("Ask questions about your documents (empty line to quit).")
        while (q := input("\n> ").strip()):
            print(answer(client, index, q))


if __name__ == "__main__":
    main()
