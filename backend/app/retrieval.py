"""
Hybrid Retriever: BM25 (keyword) + FAISS (semantic) + Reciprocal Rank Fusion.

How it works:
1. BM25 finds documents matching exact keywords.
2. FAISS finds documents that are semantically similar.
3. RRF merges both ranked lists into one final ranking.
"""
import os
import json
import pickle
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)?", re.IGNORECASE)

GENERIC_QUERY_TERMS = {
    "a", "an", "and", "api", "assessment", "assessments", "battery", "can",
    "candidate", "candidates", "core", "developer", "developers", "engineer",
    "engineers", "for", "full", "hiring", "in", "is", "level", "need",
    "of", "our", "recommend", "required", "role", "senior", "skills",
    "stack", "strong", "team", "test", "tests", "the", "to", "we", "with",
    "years",
}


QUERY_EXPANSIONS = [
    (re.compile(r"\bmid[-\s]?level\b", re.IGNORECASE), "mid professional"),
    (re.compile(r"\bjunior\b|\bentry[-\s]?level\b|\bintern\b|\bgraduate\b", re.IGNORECASE), "entry level graduate"),
    (re.compile(r"\bsenior\b|\bexecutive\b|\bdirector\b|\blead\b", re.IGNORECASE), "senior executive director lead"),
    (re.compile(r"\bpersonality\b|\bbehavior\b", re.IGNORECASE), "personality behavior"),
    (re.compile(r"\bcognitive\b|\bability\b|\baptitude\b|\breasoning\b", re.IGNORECASE), "ability aptitude cognitive reasoning"),
    (re.compile(r"\btechnical\b|\bskills\b|\bknowledge\b|\bcoding\b|\bdeveloper\b|\bjava\b|\bpython\b|\bnet\b", re.IGNORECASE), "knowledge skills technical coding"),
    (re.compile(r"\bstakeholder\b|\bcommunication\b|\bcollaborat", re.IGNORECASE), "stakeholder communication collaboration interpersonal"),
    (re.compile(r"\bremote\b|\badaptive\b", re.IGNORECASE), "remote adaptive"),
]


class HybridRetriever:
    """Combines BM25 + FAISS search with Reciprocal Rank Fusion."""

    def __init__(self, assessments: list[dict], faiss_index, bm25_index,
                 embed_model: SentenceTransformer):
        self.assessments = assessments
        self.faiss_index = faiss_index
        self.bm25 = bm25_index
        self.embed_model = embed_model
        self.search_texts = [self._build_search_text(a) for a in assessments]

    def _build_search_text(self, assessment: dict) -> str:
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

    def _normalize_query(self, query: str) -> str:
        normalized = query.strip().lower()
        for pattern, replacement in QUERY_EXPANSIONS:
            if pattern.search(normalized):
                normalized = f"{normalized} {replacement}"
        return normalized

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def _extract_priority_terms(self, query: str) -> set[str]:
        return {
            token for token in self._tokenize(query)
            if len(token) > 2 and token not in GENERIC_QUERY_TERMS
        }

    def _metadata_bonus(self, assessment: dict, query_tokens: set[str]) -> float:
        bonus = 0.0

        weighted_fields = [
            (assessment.get("name", ""), 0.06),
            (assessment.get("test_type", ""), 0.04),
            (" ".join(assessment.get("keys", []) or []), 0.04),
            (" ".join(assessment.get("job_levels", []) or []), 0.03),
            (" ".join(assessment.get("languages", []) or []), 0.02),
            (assessment.get("duration", ""), 0.01),
            (assessment.get("description", ""), 0.015),
        ]

        for field_text, weight in weighted_fields:
            field_tokens = set(self._tokenize(field_text))
            if not field_tokens:
                continue
            overlap = query_tokens & field_tokens
            if overlap:
                bonus += min(weight * len(overlap), weight * 3)

        query_text = " ".join(query_tokens)
        name_text = assessment.get("name", "").lower()
        if name_text and name_text in query_text:
            bonus += 0.15

        if assessment.get("remote_testing", "").lower().startswith("y") and "remote" in query_tokens:
            bonus += 0.02
        if assessment.get("adaptive_testing", "").lower().startswith("y") and "adaptive" in query_tokens:
            bonus += 0.02

        return bonus

    def _skill_match_bonus(self, assessment: dict, query_tokens: set[str]) -> float:
        """Favor assessments that explicitly match technical skills in the query."""
        priority_terms = {
            token for token in query_tokens
            if len(token) > 2 and token not in GENERIC_QUERY_TERMS
        }
        if not priority_terms:
            return 0.0

        name_tokens = set(self._tokenize(assessment.get("name", "")))
        description_tokens = set(self._tokenize(assessment.get("description", "")))

        name_overlap = priority_terms & name_tokens
        description_overlap = priority_terms & description_tokens

        bonus = 0.0
        if name_overlap:
            bonus += min(0.09 * len(name_overlap), 0.24)
        if description_overlap:
            bonus += min(0.03 * len(description_overlap), 0.12)

        return bonus

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Run hybrid search and return top_k assessments.

        Steps:
            1. BM25 keyword search → get ranked doc indices
            2. FAISS semantic search → get ranked doc indices
            3. Reciprocal Rank Fusion to merge rankings
            4. Return top_k results with full metadata
        """
        n_docs = len(self.assessments)
        fetch_k = min(top_k * 3, n_docs)  # fetch more, then fuse
        normalized_query = self._normalize_query(query)
        query_tokens = set(self._tokenize(normalized_query))

        # --- BM25 Search ---
        bm25_scores = self.bm25.get_scores(list(query_tokens))
        bm25_top = np.argsort(bm25_scores)[::-1][:fetch_k].tolist()

        # --- FAISS Search ---
        query_embedding = self.embed_model.encode(
            [normalized_query], normalize_embeddings=True
        )
        faiss_scores, faiss_indices = self.faiss_index.search(
            query_embedding.astype("float32"), fetch_k
        )
        faiss_top = faiss_indices[0].tolist()
        # Filter out -1 (FAISS returns -1 if fewer results than k)
        faiss_top = [i for i in faiss_top if i >= 0]

        # --- Reciprocal Rank Fusion ---
        fused = self._reciprocal_rank_fusion([bm25_top, faiss_top])

        reranked = []
        for doc_id, score in fused:
            if 0 <= doc_id < n_docs:
                bonus = self._metadata_bonus(self.assessments[doc_id], query_tokens)
                bonus += self._skill_match_bonus(self.assessments[doc_id], query_tokens)
                reranked.append((doc_id, score + bonus))
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results with full metadata
        results = []
        for doc_id, score in reranked[:top_k]:
            if 0 <= doc_id < n_docs:
                result = self.assessments[doc_id].copy()
                result["rrf_score"] = round(score, 4)
                results.append(result)

        return results

    def _reciprocal_rank_fusion(
        self, rankings: list[list[int]], k: int = 60
    ) -> list[tuple[int, float]]:
        """
        Merge multiple ranked lists using RRF.

        Formula: score(doc) = sum( 1 / (k + rank) ) across all rankers
        k=60 is the standard smoothing constant.
        """
        scores = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank + 1)

        # Sort by score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def load_retriever(data_dir: str = None) -> HybridRetriever:
    """
    Load pre-built indexes and return a ready-to-use retriever.

    Expected files in data_dir:
        - processed/assessments.json
        - indexes/faiss.index
        - indexes/bm25.pkl
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # Load assessments
    assessments_path = os.path.join(data_dir, "processed", "assessments.json")
    with open(assessments_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    # Load FAISS index
    faiss_path = os.path.join(data_dir, "indexes", "faiss.index")
    faiss_index = faiss.read_index(faiss_path)

    # Load BM25 index
    bm25_path = os.path.join(data_dir, "indexes", "bm25.pkl")
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

    # Load embedding model (for encoding queries at runtime)
    model_name = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")
    embed_model = SentenceTransformer(model_name)

    return HybridRetriever(assessments, faiss_index, bm25_index, embed_model)
