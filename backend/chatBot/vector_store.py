"""
Local Vector Store & Semantic Document Chunking Engine (100% Offline).
Provides page-aware chunking, vector indexing, exact entity boosting, 
and cosine-similarity retrieval using in-memory TF-IDF and numpy / scikit-learn without any network calls.
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _extract_pages_from_text(text: str) -> List[int]:
    """Finds all page numbers tagged in a text section."""
    matches = re.findall(r"\[Page\s+(\d+)\]", text, re.IGNORECASE)
    return [int(m) for m in matches] if matches else []


def chunk_document_text(
    text: str,
    chunk_size: int = 1400,
    chunk_overlap: int = 200,
    min_chunk_len: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits document text into page-aware semantic chunks with overlap.
    Preserves page boundaries, certificates, and paragraph structure.
    
    Returns:
        List of dicts: [
            {
                "id": 0, 
                "text": "...", 
                "char_len": 1234, 
                "pages": [1], 
                "primary_page": 1
            }
        ]
    """
    if not text or not text.strip():
        return []

    # Check if text contains page delimiters
    page_sections = re.split(r"(?=---\s*\[Page\s+\d+\]\s*---)", text.strip())
    page_sections = [s.strip() for s in page_sections if s.strip()]

    chunks = []
    chunk_id = 0

    if len(page_sections) > 1 and all(len(s) <= chunk_size * 1.5 for s in page_sections):
        # Case A: Document is neatly divided by pages and each page fits into reasonable chunk size
        # Group small consecutive pages or keep 1 page per chunk
        for sec in page_sections:
            pages = _extract_pages_from_text(sec)
            primary_page = pages[0] if pages else 1
            chunks.append({
                "id": chunk_id,
                "text": sec,
                "char_len": len(sec),
                "pages": pages,
                "primary_page": primary_page
            })
            chunk_id += 1
        return chunks

    # Case B: General document chunking with paragraph/sentence preservation
    paragraphs = re.split(r"\n\s*\n", text.strip())
    raw_blocks = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if len(p_clean) > chunk_size:
            lines = p_clean.split("\n")
            for l in lines:
                l_clean = l.strip()
                if len(l_clean) > chunk_size:
                    sentences = re.split(r"(?<=[.?!])\s+", l_clean)
                    raw_blocks.extend([s.strip() for s in sentences if s.strip()])
                elif l_clean:
                    raw_blocks.append(l_clean)
        else:
            raw_blocks.append(p_clean)

    current_chunk_blocks = []
    current_length = 0

    for block in raw_blocks:
        block_len = len(block)
        if current_length + block_len <= chunk_size:
            current_chunk_blocks.append(block)
            current_length += block_len + 2
        else:
            if current_chunk_blocks:
                chunk_str = "\n\n".join(current_chunk_blocks).strip()
                if len(chunk_str) >= min_chunk_len:
                    pages = _extract_pages_from_text(chunk_str)
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_str,
                        "char_len": len(chunk_str),
                        "pages": pages,
                        "primary_page": pages[0] if pages else 1
                    })
                    chunk_id += 1

                # Retain overlap blocks
                overlap_blocks = []
                overlap_len = 0
                for b in reversed(current_chunk_blocks):
                    if overlap_len + len(b) <= chunk_overlap:
                        overlap_blocks.insert(0, b)
                        overlap_len += len(b) + 2
                    else:
                        break

                current_chunk_blocks = overlap_blocks + [block]
                current_length = sum(len(b) + 2 for b in current_chunk_blocks)
            else:
                current_chunk_blocks = [block]
                current_length = block_len

    if current_chunk_blocks:
        chunk_str = "\n\n".join(current_chunk_blocks).strip()
        if len(chunk_str) >= min_chunk_len or not chunks:
            pages = _extract_pages_from_text(chunk_str)
            chunks.append({
                "id": chunk_id,
                "text": chunk_str,
                "char_len": len(chunk_str),
                "pages": pages,
                "primary_page": pages[0] if pages else 1
            })

    return chunks


# Optional Local Neural Embedding Model (FastEmbed ONNX)
_GLOBAL_EMBED_MODEL = None

def get_neural_embedding_model():
    """Lazily loads local fast ONNX neural embedding model (BAAI/bge-small-en-v1.5).
    Safely bypassed on constrained cloud servers (Render 512MB RAM) to prevent OOM termination.
    """
    global _GLOBAL_EMBED_MODEL
    # Render free plan limit is 512MB; TF-IDF uses <5MB RAM and runs 100% reliably
    if os.getenv("RENDER") or os.getenv("DISABLE_FASTEMBED", "0") == "1":
        return None

    if _GLOBAL_EMBED_MODEL is None:
        try:
            from fastembed import TextEmbedding
            _GLOBAL_EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception as e:
            print(f"[VectorStore] FastEmbed lazy load skipped: {e}")
            _GLOBAL_EMBED_MODEL = False
    return _GLOBAL_EMBED_MODEL if _GLOBAL_EMBED_MODEL is not False else None


class LocalVectorStore:
    """
    Hybrid Local Vector Store for 100% Offline Retrieval-Augmented Generation (RAG).
    Combines:
    1. Local Dense Neural Embeddings (FastEmbed BAAI/bge-small-en-v1.5) for deep semantic comprehension.
    2. Sublinear TF-IDF n-grams (1, 2, 3) for rare vocabulary & keyword exactness.
    3. Exact phrase & entity boosting (+0.50) for student names, codes, and project titles.
    """

    def __init__(self, chunks: List[Dict[str, Any]] = None):
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        
        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes text chunks into dense neural and lexical vector spaces."""
        self.chunks = chunks or []
        if not self.chunks:
            return

        corpus = [c["text"] for c in self.chunks]

        # 1. Compute Dense Neural Embeddings (FastEmbed)
        embed_model = get_neural_embedding_model()
        if embed_model:
            try:
                # Fast parallel ONNX batch inference
                dense_list = list(embed_model.embed(corpus))
                self.dense_embeddings = np.array(dense_list, dtype=np.float32)
                # Normalize for cosine similarity via dot product
                norms = np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                self.dense_embeddings = self.dense_embeddings / norms
            except Exception as e:
                print(f"[VectorStore] Dense embedding index fallback: {e}")
                self.dense_embeddings = None

        # 2. Compute Lexical TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True,
            lowercase=True,
            max_features=8000
        )
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),
                stop_words=None,
                sublinear_tf=False,
                lowercase=True
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.01) -> List[Dict[str, Any]]:
        """
        Hybrid Semantic Search:
        Blends Dense Neural Cosine Similarity (60%) + TF-IDF Lexical Match (40%) + Exact Entity Boost.
        """
        if not self.chunks:
            return []

        clean_q = query.strip()
        if not clean_q:
            return self.chunks[:top_k]

        noise_words = {
            "what", "is", "the", "of", "in", "for", "to", "a", "an", "and", "or", "by", 
            "who", "where", "how", "which", "tell", "me", "about", "give", "show", "find",
            "please", "can", "you", "details", "information", "project", "submitted"
        }
        query_terms = [
            w.lower() for w in re.findall(r"\b[A-Za-z0-9_\-]{2,}\b", clean_q) 
            if w.lower() not in noise_words
        ]

        n_chunks = len(self.chunks)
        dense_scores = np.zeros(n_chunks, dtype=np.float32)
        tfidf_scores = np.zeros(n_chunks, dtype=np.float32)

        # A. Dense Neural Similarity
        embed_model = get_neural_embedding_model()
        if self.dense_embeddings is not None and embed_model:
            try:
                q_dense = list(embed_model.embed([clean_q]))[0]
                q_dense = np.array(q_dense, dtype=np.float32)
                q_norm = np.linalg.norm(q_dense)
                if q_norm > 0:
                    q_dense = q_dense / q_norm
                    dense_scores = np.dot(self.dense_embeddings, q_dense)
            except Exception as e:
                print(f"[VectorStore] Dense search fallback: {e}")

        # B. TF-IDF Lexical Similarity
        if self.vectorizer and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([clean_q])
                tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            except Exception:
                pass

        # C. Hybrid Blending & Exact Entity Boost
        has_dense = self.dense_embeddings is not None
        adjusted_scores = []
        q_lower = clean_q.lower()

        for idx, chunk in enumerate(self.chunks):
            text_lower = chunk["text"].lower()

            if has_dense:
                # 60% Dense Neural + 40% TF-IDF Lexical
                combined = float(0.60 * dense_scores[idx] + 0.40 * tfidf_scores[idx])
            else:
                combined = float(tfidf_scores[idx])

            # Exact full phrase match boost (+0.50)
            if len(clean_q) > 3 and q_lower in text_lower:
                combined += 0.50

            # Keyword term coverage boost (+0.30)
            if query_terms:
                matched_terms = sum(1 for t in query_terms if t in text_lower)
                term_ratio = matched_terms / len(query_terms)
                combined += term_ratio * 0.30

            adjusted_scores.append((combined, idx))

        # Rank descending by adjusted score
        adjusted_scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in adjusted_scores:
            if score >= min_score or len(results) == 0:
                chunk_data = dict(self.chunks[idx])
                chunk_data["score"] = round(score, 4)
                results.append(chunk_data)
            if len(results) >= top_k:
                break

        return results

    def search_by_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Returns all chunks belonging to a specific page number."""
        return [c for c in self.chunks if page_num in c.get("pages", []) or c.get("primary_page") == page_num]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Returns all chunks currently in the store."""
        return self.chunks

    def get_combined_text(self, max_chars: int = 15000) -> str:
        """Returns concatenated text from all chunks up to max_chars."""
        all_text = "\n\n".join(c["text"] for c in self.chunks)
        return all_text[:max_chars]
