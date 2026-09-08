import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chatBot.llama_summarizer import classify_user_prompt_intent, generate_pdf_summary_with_llama
from chatBot.vector_store import chunk_document_text, LocalVectorStore

def test_intent_classification():
    print("\n--- Testing Intent Classification & Typo Tolerance ---", flush=True)
    test_cases = [
        ("sumaarize this document", "GENERAL_SUMMARY"),
        ("sammary of this", "GENERAL_SUMMARY"),
        ("overveiw of document", "GENERAL_SUMMARY"),
        ("tldr", "GENERAL_SUMMARY"),
        ("extract all functions", "EXTRACT_FUNCTIONS"),
        ("what topics are covered", "TOPICS"),
        ("important key points", "KEY_POINTS"),
        ("What project was submitted by Divya Shah?", "TOPIC_QA"),
        ("Who is the guide for Swatantra Kumar?", "TOPIC_QA"),
    ]

    for prompt, expected in test_cases:
        actual = classify_user_prompt_intent(prompt)
        status = "[PASS]" if actual == expected else f"[FAIL - got {actual}]"
        print(f"Prompt: '{prompt}' -> {actual} {status}", flush=True)

def test_rag_with_sample_certificates():
    print("\n--- Testing RAG & Synthesis on Sample Academic Certificates ---", flush=True)
    sample_text = """
--- [Page 1] ---
Maliba Campus, GopalVidyanagar, Bardoli - Mahuva Road
Shrimad Rajchandra Institute of Management and Computer Application
Team No :MCA4_001 Team Approval Code :6449
CERTIFICATE
This is to certify that the project entitled EWSign Tool for CS8033 - Project-III submitted by Prashant Patil(202404104610014) for 4th semester as a partial fulfillment for the requirement of the degree of Master of ComputerApplications during the academic year 2025-2026.
Dr. Jitendra Upadhyay Dr. Dharmendra Bhatti
Guide Director
Date: 18 May 2026

--- [Page 91] ---
Maliba Campus, GopalVidyanagar, Bardoli - Mahuva Road
Shrimad Rajchandra Institute of Management and Computer Application
Team No :MCA4_078 Team Approval Code :5616
CERTIFICATE
This is to certify that the project entitled Clothiq:-Ecommerce Men’s Tshirt for CS8033 - Project-III submitted by Divya Shah (202404104610080) for 4th semester as a partial fulfillment for the requirement of the degree of Master of Computer Applications during the academic year 2025-2026.
Ms. Shivani Talaviya Dr. Dharmendra Bhatti
Guide Director
Date: 18 June 2026

--- [Page 112] ---
Maliba Campus, GopalVidyanagar, Bardoli - Mahuva Road
Shrimad Rajchandra Institute of Management and Computer Application
Team No :MCA4_097 Team Approval Code :5832
CERTIFICATE
This is to certify that the project entitled NVIDIA CUDA-Powered Agentic AI for Intelligent GPU Optimization for CS8033 - Project-III submitted by Dev Savaj (202404104610114) for 4th semester as a partial fulfillment for the requirement of the degree of Master of ComputerApplications during the academic year 2025-2026.
Mr. Yash Panchal Dr. Dharmendra Bhatti
Guide Director
Date: 18 May 2026
"""
    raw_bytes = sample_text.encode("utf-8")

    # Test 1: Typo Summary
    print("\n[Test 1] Query: 'sumaarize this document'", flush=True)
    res1 = generate_pdf_summary_with_llama(raw_bytes, "external_certificate.txt", "sumaarize this document")
    print(res1[:500] + "...\n", flush=True)

    # Test 2: Direct Student Query
    print("[Test 2] Query: 'What is the project of Divya Shah?'", flush=True)
    res2 = generate_pdf_summary_with_llama(raw_bytes, "external_certificate.txt", "What is the project of Divya Shah?")
    print(res2, flush=True)

if __name__ == "__main__":
    test_intent_classification()
    test_rag_with_sample_certificates()
