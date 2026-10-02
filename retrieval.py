"""Readable BM25 / TF-IDF / hybrid retrieval with optional local embeddings."""
from collections import Counter
from functools import lru_cache
import math
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from config import EMBEDDING_MODEL
from models import Chunk, SearchHit

STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "explain", "describe", "document", "documents", "according", "tell", "does",
    "did", "using", "used", "say", "says", "notes", "note", "please", "summarise",
    "summarize", "summary", "compare", "comparison", "research", "assistant",
}


def tokens(text):
    return [word for word in re.findall(r"\b\w+\b", text.lower())
            if word not in STOP_WORDS]


@lru_cache(maxsize=1)
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ValueError("Install requirements-semantic.txt to enable embedding search.") from error
    # CPU is predictable on an 8 GB Mac; no CUDA setup is needed.
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu", trust_remote_code=False)


class DocumentIndex:
    def __init__(self, chunks: list[Chunk], use_embeddings=False):
        if not chunks:
            raise ValueError("No readable passages to index.")
        # Identical documents should not appear repeatedly in search results.
        self.chunks = list({chunk.chunk_id: chunk for chunk in chunks}.values())
        self.counts = [Counter(tokens(chunk.text)) for chunk in self.chunks]
        self.lengths = np.array([sum(count.values()) for count in self.counts], dtype=float)
        self.average_length = float(self.lengths.mean()) or 1.0
        self.document_frequency = Counter()
        for count in self.counts:
            self.document_frequency.update(count.keys())
        self.vectorizer = TfidfVectorizer(tokenizer=tokens, token_pattern=None,
                                         lowercase=False, ngram_range=(1, 2))
        try:
            self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        except ValueError as error:
            raise ValueError("Documents do not contain enough searchable words.") from error
        self.embeddings = None
        if use_embeddings:
            self.embeddings = load_embedding_model().encode(
                [chunk.text for chunk in self.chunks], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False, batch_size=16,
            )

    def bm25(self, query):
        scores = np.zeros(len(self.chunks))
        for term in set(tokens(query)):
            frequency = np.array([count[term] for count in self.counts], dtype=float)
            df = self.document_frequency[term]
            idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
            scores += idf * frequency * 2.5 / (
                frequency + 1.5 * (0.25 + 0.75 * self.lengths / self.average_length)
            )
        return scores

    def overview(self, top_k=5, filenames=None) -> list[SearchHit]:
        """Sample passages across documents and across each document's length.

        Overview selection has no similarity score. It is only used when the
        user explicitly asks for a broad document explanation or summary.
        """
        if top_k <= 0:
            return []
        allowed = set(filenames) if filenames is not None else None
        groups = {}
        for chunk in self.chunks:
            if allowed is None or chunk.filename in allowed:
                groups.setdefault(chunk.document_id, []).append(chunk)
        if not groups:
            return []
        # Round-robin allocation keeps a long document from taking every slot.
        counts = dict.fromkeys(groups, 0)
        for _ in range(top_k):
            candidates = [key for key in groups if counts[key] < len(groups[key])]
            if not candidates:
                break
            chosen = min(candidates, key=lambda key: counts[key])
            counts[chosen] += 1
        samples = []
        for key, chunks in groups.items():
            count = counts[key]
            positions = ([0] if count == 1 else
                         [round(i * (len(chunks) - 1) / (count - 1)) for i in range(count)])
            samples.append([SearchHit(chunks[position], 0.0, 0.0) for position in positions])
        return [group[rank] for rank in range(max(counts.values()))
                for group in samples if rank < len(group)]

    def search(self, query: str, top_k=5, method="hybrid", filenames=None) -> list[SearchHit]:
        if method not in {"bm25", "tfidf", "hybrid", "semantic"}:
            raise ValueError("Unknown retrieval method.")
        if top_k <= 0 or not query.strip():
            return []
        lexical = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        bm25 = self.bm25(query)
        evidence = lexical.copy()
        if method == "tfidf":
            scores = lexical
        elif method == "bm25":
            scores = bm25
        elif method == "hybrid":
            # BM25 scale varies by question. Normalise only for ranking.
            scaled_bm25 = bm25 / (bm25.max() or 1.0)
            scores = 0.5 * scaled_bm25 + 0.5 * lexical
        else:
            if self.embeddings is None:
                raise ValueError("Rebuild the index with embeddings enabled.")
            query_vector = load_embedding_model().encode(
                [query], normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            semantic = np.clip(self.embeddings @ query_vector, -1, 1)
            scaled_bm25 = bm25 / (bm25.max() or 1.0)
            scores = 0.7 * semantic + 0.3 * scaled_bm25
            evidence = semantic
        allowed = set(filenames) if filenames is not None else None
        ranked = np.argsort(-scores, kind="stable")
        return [SearchHit(self.chunks[i], float(scores[i]), float(evidence[i]))
                for i in ranked if scores[i] > 0 and
                (allowed is None or self.chunks[i].filename in allowed)][:top_k]
