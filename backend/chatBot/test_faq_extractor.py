import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatBot.llama_summarizer import extract_document_faqs, generate_pdf_summary_with_llama
from chatBot.document_memory import doc_memory

sample_text = """
# MCA Degree Admission Policy 2026

## Overview
Master of Computer Applications (MCA) is a 2-year postgraduate program designed for students aiming for advanced software development and AI engineering careers.

## Eligibility Criteria
- Candidate must have completed BCA / B.Sc (CS/IT) or equivalent bachelor degree with minimum 50% aggregate.
- Mathematics at 10+2 level or Graduation level is mandatory.

## Fee Structure
- Tuition Fee: Rs. 45,000 per semester
- Examination & Lab Fee: Rs. 5,000 per year
- Total Course Fee: Rs. 1,90,000 for 2 years.

## Technical Skills Covered
Students will learn Python, React, Data Structures, Machine Learning, PyTorch, Docker, and Cloud Computing.

## Contact Details
For admissions, contact +91 98765 43210 or email admissions@suratcollege.edu.in.
"""

def test_faq_extraction():
    filename = "mca_policy.txt"
    file_bytes = sample_text.encode("utf-8")
    
    # 1. Test raw FAQ extraction
    faqs = extract_document_faqs(sample_text, filename)
    print(f"✅ Extracted {len(faqs)} FAQs:")
    for f in faqs:
        print(f"  • Q: {f['question']}")
    
    assert len(faqs) >= 5, "Should extract at least 5 FAQs"
    
    # 2. Test full gateway ingestion
    summary = generate_pdf_summary_with_llama(file_bytes, filename, "Summarize this document.")
    print("\n--- Summary Output Preview ---")
    print(summary[:400])
    
    # Verify FAQs are in doc memory
    doc_hash = doc_memory.compute_document_hash(file_bytes, filename)
    entry = doc_memory.get_document(doc_hash)
    assert entry is not None
    assert len(entry.get_faqs()) >= 5
    print("\n✅ Document memory entry contains FAQs!")
    
    # 3. Test out-of-scope question
    out_of_scope_reply = generate_pdf_summary_with_llama(file_bytes, filename, "What is the capital of France?")
    print("\n--- Out-Of-Scope Reply ---")
    print(out_of_scope_reply)
    assert "Question Outside Document Scope" in out_of_scope_reply or "outside the scope" in out_of_scope_reply.lower()
    print("\n✅ Out-of-scope boundary test passed!")

if __name__ == "__main__":
    test_faq_extraction()
