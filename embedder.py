"""Build an embedding index over chunks.json, or query it.

  python3 embedder.py              # embed all chunks -> embeddings.npy
  python3 embedder.py "deine Frage"  # retrieve top-k chunks for a query
"""
import sys, json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"   # CPU-friendly, German + English
CHUNKS, VECS = "chunks.json", "embeddings.npy"
TOP_K = 5

_model = None
def model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model


def encode(texts):
    # e5 is trained for cosine similarity -> L2-normalise so dot product == cosine.
    return model().encode(texts, normalize_embeddings=True, show_progress_bar=True)


def build():
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    vecs = encode([c["embed_text"] for c in chunks]).astype(np.float32)
    np.save(VECS, vecs)
    print(f"{VECS}: {vecs.shape[0]} vectors x {vecs.shape[1]} dims")


def query(text):
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    vecs = np.load(VECS)
    q = encode([f"query: {text}"])[0]          # e5 wants the "query:" prefix
    scores = vecs @ q                          # cosine, since both are normalised
    for i in np.argsort(-scores)[:TOP_K]:
        c = chunks[i]
        print(f"\n[{c['id']}] {scores[i]:.3f}  {c['path']}")
        print(c["text"][:300].strip())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
    else:
        build()
