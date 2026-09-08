
"""
Document Memory & Semantic Ingestion Lifecycle Manager.
Provides ChatGPT-like memory for uploaded documents without storing bulky raw files in external DBs.

Lifecycle:
1. Document Fingerprinting (SHA-256 hash of file content).
2. Cold Ingestion (1st upload): Extract -> Chunk -> Embed -> Index -> Generate V1 summary -> Cache in memory.
3. Warm Retrieval (2nd+ upload): Instant memory recall -> Reuse Vector Store -> Build deeper, improved response (V2+) -> Save tokens.
"""

import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from .vector_store import chunk_document_text, LocalVectorStore
from .document_parser import extract_universal_document_text


class DocumentMemoryEntry:
    """Represents the in-memory knowledge representation of an uploaded document."""

    def __init__(self, doc_hash: str, filename: str, raw_text: str, chunks: List[Dict[str, Any]], vector_store: LocalVectorStore):
        self.doc_hash: str = doc_hash
        self.filename: str = filename
        self.raw_text: str = raw_text
        self.char_count: int = len(raw_text)
        self.chunks: List[Dict[str, Any]] = chunks
        self.vector_store: LocalVectorStore = vector_store
        self.upload_count: int = 1
        self.created_at: float = time.time()
        self.last_accessed: float = time.time()
        self.summaries: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, str]] = []  # [{"prompt": ..., "response": ..., "version": 1}]
        self.faqs: List[Dict[str, str]] = []  # [{"question": ..., "answer": ...}]

    def set_faqs(self, faqs: List[Dict[str, str]]):
        """Sets generated FAQs for this document entry."""
        self.faqs = faqs

    def get_faqs(self) -> List[Dict[str, str]]:
        """Returns generated FAQs for this document."""
        return self.faqs

    def record_interaction(self, prompt: str, response: str, is_reupload: bool = False):
        """Records an interaction turn for this document in memory."""
        self.last_accessed = time.time()
        version = len(self.summaries) + 1
        entry = {
            "version": version,
            "prompt": prompt,
            "response": response,
            "timestamp": self.last_accessed,
            "is_reupload": is_reupload
        }
        self.summaries.append(entry)
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": response})

    def get_latest_summary(self) -> Optional[str]:
        """Returns the most recent summary/response generated for this document."""
        if self.summaries:
            return self.summaries[-1]["response"]
        return None

    def get_all_past_prompts(self) -> List[str]:
        """Returns list of all past user prompts on this document."""
        return [s["prompt"] for s in self.summaries]


class DocumentMemoryManager:
    """
    Central in-memory Document Knowledge Base and Lifecycle Cache.
    Manages document instances, fingerprints, session bindings, and memory recalls.
    """

    def __init__(self):
        # Maps doc_hash -> DocumentMemoryEntry
        self._memory_store: Dict[str, DocumentMemoryEntry] = {}
        # Secondary index: filename (lowercase) -> latest doc_hash
        self._name_index: Dict[str, str] = {}
        # Session index: session_id -> latest doc_hash
        self._session_index: Dict[str, str] = {}
        # User index: user_id -> list of doc_hashes
        self._user_index: Dict[str, List[str]] = {}

    @staticmethod
    def compute_document_hash(file_bytes: bytes, filename: str) -> str:
        """Computes deterministic SHA-256 fingerprint for document content."""
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        # Combine content hash with normalized filename
        return hasher.hexdigest()[:16]

    def has_document(self, doc_hash: str) -> bool:
        """Checks if document is already remembered in memory."""
        return doc_hash in self._memory_store

    def get_document(self, doc_hash: str) -> Optional[DocumentMemoryEntry]:
        """Retrieves remembered document entry."""
        entry = self._memory_store.get(doc_hash)
        if entry:
            entry.last_accessed = time.time()
        return entry

    def bind_session_document(self, session_id: Optional[str], doc_hash: str):
        """Binds a document to a specific chat session."""
        if session_id:
            self._session_index[str(session_id)] = doc_hash

    def bind_user_document(self, user_id: Optional[str], doc_hash: str):
        """Binds a document to a specific user's document registry."""
        if user_id:
            u_str = str(user_id)
            if u_str not in self._user_index:
                self._user_index[u_str] = []
            if doc_hash not in self._user_index[u_str]:
                self._user_index[u_str].append(doc_hash)

    def get_document_for_session(self, session_id: Optional[str]) -> Optional[DocumentMemoryEntry]:
        """Returns the document associated with the given session ID, if any."""
        if not session_id:
            return None
        doc_hash = self._session_index.get(str(session_id))
        if doc_hash and doc_hash in self._memory_store:
            return self._memory_store[doc_hash]
        return None

    def get_best_document_match(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[DocumentMemoryEntry]:
        """
        Intelligently identifies which document the user is querying:
        1. Explicit document name match in the query (e.g. 'Life Insurers.pdf' or 'Life Insurers').
        2. Active session-bound document.
        3. Highest semantic RAG match across user's documents.
        4. Most recently accessed document.
        """
        if not self._memory_store:
            return None

        clean_q = query.lower().strip()

        # 1. Check for explicit filename mention
        for fn_lower, d_hash in self._name_index.items():
            base_name = fn_lower.rsplit(".", 1)[0]
            if fn_lower in clean_q or (len(base_name) >= 4 and base_name in clean_q):
                entry = self._memory_store.get(d_hash)
                if entry:
                    entry.last_accessed = time.time()
                    return entry

        # 2. Check session-bound document
        session_doc = self.get_document_for_session(session_id)

        # 3. Check semantic search score across candidate documents
        candidate_hashes = list(self._memory_store.keys())
        if user_id and str(user_id) in self._user_index:
            user_hashes = [h for h in self._user_index[str(user_id)] if h in self._memory_store]
            if user_hashes:
                candidate_hashes = user_hashes

        best_score = 0.0
        best_entry = None

        for h in candidate_hashes:
            entry = self._memory_store[h]
            if entry.vector_store:
                results = entry.vector_store.search(query, top_k=1, min_score=0.01)
                if results:
                    score = results[0].get("score", 0.0)
                    if score > best_score:
                        best_score = score
                        best_entry = entry

        # If a document has a strong matching score (> 0.20), choose it
        if best_entry and best_score >= 0.20:
            best_entry.last_accessed = time.time()
            return best_entry

        # Otherwise prioritize the session's active document if available
        if session_doc:
            session_doc.last_accessed = time.time()
            return session_doc

        # Otherwise return the highest scoring candidate, or latest accessed
        if best_entry and best_score > 0.03:
            best_entry.last_accessed = time.time()
            return best_entry

        # Fallback to the latest accessed document
        sorted_entries = sorted(self._memory_store.values(), key=lambda e: e.last_accessed, reverse=True)
        if sorted_entries:
            latest = sorted_entries[0]
            latest.last_accessed = time.time()
            return latest

        return None

    def ingest_or_recall(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 250,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[DocumentMemoryEntry, bool]:
        """
        Main lifecycle entrypoint:
        - If document exists in memory: returns (entry, is_reupload=True) without re-chunking or re-indexing.
        - If document is new: parses text, creates chunks, builds vector store, stores in memory, returns (entry, is_reupload=False).
        Binds session and user mappings automatically.
        """
        doc_hash = self.compute_document_hash(file_bytes, filename)

        # Check if already in memory
        if doc_hash in self._memory_store:
            entry = self._memory_store[doc_hash]
            entry.upload_count += 1
            entry.last_accessed = time.time()
            self._name_index[filename.lower()] = doc_hash
            self.bind_session_document(session_id, doc_hash)
            self.bind_user_document(user_id, doc_hash)
            return entry, True

        # Cold ingestion
        raw_text = extract_universal_document_text(file_bytes, filename)
        if not raw_text or not raw_text.strip():
            raw_text = ""

        chunks = chunk_document_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks and raw_text:
            chunks = [{"id": 0, "text": raw_text[:3000], "char_len": len(raw_text[:3000])}]

        vector_store = LocalVectorStore(chunks)

        entry = DocumentMemoryEntry(
            doc_hash=doc_hash,
            filename=filename,
            raw_text=raw_text,
            chunks=chunks,
            vector_store=vector_store
        )

        self._memory_store[doc_hash] = entry
        self._name_index[filename.lower()] = doc_hash
        self.bind_session_document(session_id, doc_hash)
        self.bind_user_document(user_id, doc_hash)

        return entry, False

    def search_all_documents(self, query: str, top_k_per_doc: int = 3, min_score: float = 0.04) -> List[Dict[str, Any]]:
        """
        Searches across all remembered documents in memory for relevant chunks.
        Returns matching chunks tagged with document filename and relevance scores.
        """
        all_matches = []
        for entry in self._memory_store.values():
            if entry.vector_store:
                results = entry.vector_store.search(query, top_k=top_k_per_doc, min_score=min_score)
                for r in results:
                    r_copy = dict(r)
                    r_copy["filename"] = entry.filename
                    all_matches.append(r_copy)
        all_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_matches[:top_k_per_doc * 2]

    def get_all_summaries_context(self, max_chars: int = 1500) -> str:
        """Returns concise summary snapshot of all currently remembered documents."""
        if not self._memory_store:
            return ""
        lines = []
        for entry in self._memory_store.values():
            summary = entry.get_latest_summary()
            if summary:
                clean_summary = summary.replace("###", "").strip()[:400]
                lines.append(f"• Document `{entry.filename}`:\n{clean_summary}")
        return "\n\n".join(lines)[:max_chars]

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on active document memory."""
        return {
            "total_documents_remembered": len(self._memory_store),
            "documents": [
                {
                    "filename": entry.filename,
                    "doc_hash": entry.doc_hash,
                    "upload_count": entry.upload_count,
                    "chunks_count": len(entry.chunks),
                    "interaction_count": len(entry.summaries),
                }
                for entry in self._memory_store.values()
            ]
        }

    def clear_memory(self):
        """Clears all stored document memories."""
        self._memory_store.clear()
        self._name_index.clear()
        self._session_index.clear()
        self._user_index.clear()


# Global Singleton Document Memory Instance
doc_memory = DocumentMemoryManager()

