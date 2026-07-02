"""
Build FAISS and BM25 indexes from assessments.json.

Usage:
    python -m scraper.build_index

Reads:  data/processed/assessments.json
Writes: data/indexes/faiss.index, data/indexes/bm25.pkl
"""
import json
import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def build_document_text(assessment: dict) -> str:
    """Build a richer document string for both semantic and keyword retrieval."""
    parts = [
        assessment.get("name", ""),
        assessment.get("description", ""),
        assessment.get("test_type", ""),
        " ".join(assessment.get("keys", []) or []),
        " ".join(assessment.get("job_levels", []) or []),
        " ".join(assessment.get("languages", []) or []),
        assessment.get("duration", ""),
        assessment.get("remote_testing", ""),
        assessment.get("adaptive_testing", ""),
    ]
    return " ".join(part for part in parts if part)


def build_indexes(data_dir: str = None, model_name: str = None):
    """
    Build FAISS (semantic) and BM25 (keyword) indexes.

    Steps:
        1. Load assessments.json
        2. Encode texts with sentence-transformers → FAISS index
        3. Tokenize texts → BM25 index
        4. Save both to disk
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if model_name is None:
        model_name = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")

    # 1. Load assessments
    assessments_path = os.path.join(data_dir, "processed", "assessments.json")
    with open(assessments_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"Loaded {len(assessments)} assessments")

    # Get text for embedding
    texts = [build_document_text(a) for a in assessments]

    # 2. Build FAISS index
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding documents...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors)
    faiss_index.add(embeddings)
    print(f"FAISS index built: {faiss_index.ntotal} vectors, dim={dimension}")

    # 3. Build BM25 index
    tokenized = [text.lower().split() for text in texts]
    bm25_index = BM25Okapi(tokenized)
    print("BM25 index built")

    # 4. Save indexes
    index_dir = os.path.join(data_dir, "indexes")
    os.makedirs(index_dir, exist_ok=True)

    faiss_path = os.path.join(index_dir, "faiss.index")
    faiss.write_index(faiss_index, faiss_path)
    print(f"Saved FAISS index to {faiss_path}")

    bm25_path = os.path.join(index_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"Saved BM25 index to {bm25_path}")

    print("Done! Indexes are ready.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    build_indexes()
