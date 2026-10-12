"""Assemble a citation-ready RAG prompt for a question.

  python3 answer.py "deine Frage"

Retrieves the top-k chunks and prints a prompt you can paste into any LLM
(Gemini Flash, etc.). The LLM is told to answer only from the chunks and cite [id]s.
"""
import sys
import numpy as np
import json
from embedder import encode, CHUNKS, VECS, TOP_K

INSTRUCTION = (
    "Beantworte die Frage ausschließlich mit den folgenden Quellen; betone die Quellen-Basis jedoch nicht am Anfang der Antwort."
    "Zitiere jede Aussage mit der Chunk-Nummer in eckigen Klammern, z. B. [25]. "
    "Wenn die Quellen die Frage nicht bestimmt beantworten, sage das.\n"
)


def retrieve(question):
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    vecs = np.load(VECS)
    q = encode([f"query: {question}"])[0]
    top = np.argsort(-(vecs @ q))[:TOP_K]
    return [chunks[i] for i in top]


def build_prompt(question):
    sources = "\n\n".join(
        f"[{c['id']}] ({c['path']})\n{c['text']}" for c in retrieve(question)
    )
    return f"{INSTRUCTION}\nQuellen:\n{sources}\n\nFrage: {question}\n\nAntwort:"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python answer.py \"deine Frage\"")
    print(build_prompt(" ".join(sys.argv[1:])))
