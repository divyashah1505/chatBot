"""
Test script for Local LLaMA Document Intelligence, Chunking, Ingestion Pipeline & Document Memory Lifecycle.
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatBot.vector_store import chunk_document_text, LocalVectorStore
from chatBot.document_memory import doc_memory, DocumentMemoryManager
from chatBot.llama_summarizer import (
    generate_pdf_summary_with_llama,
    classify_user_prompt_intent
)

def run_tests():
    print("=" * 60)
    print("[TEST] TESTING INGESTION LIFECYCLE & DOCUMENT MEMORY SYSTEM")
    print("=" * 60)

    # 1. Test Intent Classification
    assert classify_user_prompt_intent("Summarize this document") == "GENERAL_SUMMARY"
    assert classify_user_prompt_intent("extract only the topics name from these document") == "TOPICS"
    assert classify_user_prompt_intent("Give me the list of topics and syllabus") == "TOPICS"
    assert classify_user_prompt_intent("from this documents provide some important points") == "KEY_POINTS"
    assert classify_user_prompt_intent("Give me the key highlights and takeaways") == "KEY_POINTS"
    assert classify_user_prompt_intent("What is the annual revenue of the company?") == "TOPIC_QA"
    print("[OK] Intent Classification passed!")

    # 2. Test Document Chunking
    sample_text = """
    Artificial Intelligence in Healthcare.
    AI is transforming medical diagnostics and drug discovery rapidly.
    Machine learning algorithms can detect anomalies in medical imaging with high precision.

    Natural Language Processing (NLP) in Medicine.
    Clinical records and doctor notes contain unstructured data.
    Large language models can summarize patient records and medical histories accurately.

    Financial Growth and Investment.
    The healthcare AI market is expected to reach $180 billion by 2030.
    Key drivers include telemedicine, robotic surgery, and automated billing workflows.
    """ * 5

    chunks = chunk_document_text(sample_text, chunk_size=400, chunk_overlap=100)
    print(f"[OK] Chunking: Created {len(chunks)} chunks with semantic overlap.")
    assert len(chunks) > 1

    # 3. Test Vector Store & Similarity Search
    vstore = LocalVectorStore(chunks)
    results = vstore.search("What is the financial market size and investment by 2030?", top_k=2)
    print(f"[OK] Vector Search: Top score: {results[0].get('score')} | Chunk ID: {results[0].get('id')}")
    assert any("180 billion" in r["text"] for r in results)

    # 4. Test Document Memory Lifecycle (First Upload vs Re-upload)
    doc_memory.clear_memory()
    doc_bytes = sample_text.encode("utf-8")
    filename = "healthcare_ai_report.txt"

    print("\n--- Testing First Upload (Cold Ingestion & V1 Summary) ---")
    res_upload_1 = generate_pdf_summary_with_llama(
        doc_bytes, filename, "Summarize this document"
    )
    print(res_upload_1[:250] + "...\n")
    assert "healthcare_ai_report.txt" in res_upload_1
    assert doc_memory.get_stats()["total_documents_remembered"] == 1

    print("--- Testing Second Upload of SAME document (Memory Recall & Enhanced V2) ---")
    res_upload_2 = generate_pdf_summary_with_llama(
        doc_bytes, filename, "Give me the important points and key highlights"
    )
    print(res_upload_2[:300] + "...\n")
    assert ("Memory" in res_upload_2) or ("Version 2" in res_upload_2) or ("Recognized" in res_upload_2)

    # Check that stats recorded 2 uploads and multiple interactions without duplicating memory
    stats = doc_memory.get_stats()
    assert stats["total_documents_remembered"] == 1
    doc_entry = stats["documents"][0]
    assert doc_entry["upload_count"] == 2
    assert doc_entry["interaction_count"] == 2
    print(f"[OK] Document Memory verified: Upload Count = {doc_entry['upload_count']}, Interactions = {doc_entry['interaction_count']}")

    print("\n--- Testing Specific Q&A on Cached Document ---")
    res_upload_3 = generate_pdf_summary_with_llama(
        doc_bytes, filename, "What is the market size by 2030?"
    )
    print(res_upload_3[:250] + "...\n")
    assert doc_memory.get_stats()["documents"][0]["interaction_count"] == 3

    print("=" * 60)
    print("[SUCCESS] ALL INGESTION LIFECYCLE & DOCUMENT MEMORY TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
