import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatBot.llama_summarizer import (
    extract_document_faqs,
    generate_pdf_summary_with_llama,
    _detect_document_category
)
from chatBot.document_memory import doc_memory
from chatBot.chatbot import get_response

life_insurers_sample = """
Updated list of Life Insurers as on 02-11-2022
Data Layout:
Sl.No Name of the Company – Corporate Office Address Regn. No Name of the Chairman / MD & CEO Name of Appointed Actuary Telephone No / Fax No / Web Address

1 Life Insurance Corporation of India Yogakshema, Jeevan Bima Marg, Mumbai – 400 021 512 Mr. M R Kumar Mr. Dinesh Pant Tel: 22-22027060 / 22-66598000 Fax: 22-22028600 Chairman@licindia.com www.licindia.in
2 HDFC Life Insurance Co. Ltd 13th Floor, Lodha Excelus, Apollo Mills Compound, Mahalaxmi, Mumbai 400 011 101 Ms. Vibha Padalkar Ms. Eshwari Murugan Tel: 022-67516666 Fax: 022-6751 6550 service@hdfclife.com www.hdfclife.com
3 Max Life Insurance Co. Ltd DLF Square Building, Jacaranda Marg, DLF City Phase-II, Gurgaon – 122002 104 Mr. Prashant Tripathy Mr. Jose Chathuparambil John Tel: 0124-4121500 Fax: 0124-6659811 ceo@maxlifeinsurance.com www.maxlifeinsurance.com
4 ICICI Prudential Life Insurance Co. Ltd ICICI Prulife Towers, Appasaheb Marathe Marg, Prabhadevi, Mumbai – 400 025 105 Mr. N S Kannan Mr. Souvik Jash Tel: 022-40391992 Fax: 022-66622031 lifeline@iciciprulife.com www.iciciprulife.com

Note: Exide Life Insurance Co. has merged with HDFC Life Insurance Co. effective from the End of Day 14th October 2022.
"""

def test_universal_document_rag():
    print("\n" + "="*70)
    print("TEST 1: Category Detection & FAQ Generation for Life Insurers.pdf")
    print("="*70)
    
    cat_code, cat_label = _detect_document_category(life_insurers_sample, "Life Insurers.pdf")
    print(f"Detected Category: {cat_code} ({cat_label})")
    assert cat_code == "COMPANY_DIRECTORY"
    
    faqs = extract_document_faqs(life_insurers_sample, "Life Insurers.pdf")
    print(f"Generated FAQs ({len(faqs)}):")
    for idx, f in enumerate(faqs, 1):
        print(f"  {idx}. {f['question']}")
    
    faq_questions = [f['question'] for f in faqs]
    assert not any("--- [Page" in q for q in faq_questions)
    assert not any("functions, methods, or APIs" in q for q in faq_questions)
    print("✅ Life Insurers FAQs are clean and domain-appropriate!")

    print("\n" + "="*70)
    print("TEST 2: Document Ingestion & Directory Queries")
    print("="*70)

    file_bytes = life_insurers_sample.encode("utf-8")
    filename = "Life Insurers.pdf"
    session_id = "session_insurance_test"
    user_id = "test_user_1"

    summary_reply = generate_pdf_summary_with_llama(
        file_bytes, filename, "Summarize this document.", user_id=user_id, session_id=session_id
    )
    print("\n--- Document Summary Output ---")
    print(summary_reply[:400] + "...")
    assert "Life Insurance Corporation of India" in summary_reply
    assert "Exide Life Insurance Co. has merged" in summary_reply
    print("✅ Ingestion & Summary passed!")

    # Query A: Contact details
    q_contacts = "What are the contact details, phone numbers, or locations?"
    resp_contacts = get_response(q_contacts, user_id=user_id, session_id=session_id)
    print("\n--- Response to Contact Details ---")
    print(resp_contacts[:450] + "...")
    assert "022-22027060" in resp_contacts or "licindia.in" in resp_contacts
    assert "Insurers Contact Directory" in resp_contacts or "Contact Details" in resp_contacts
    print("✅ Contact Details query passed!")

    # Query B: Important Dates & Merger
    q_dates = "What are the important dates, deadlines, or schedules mentioned?"
    resp_dates = get_response(q_dates, user_id=user_id, session_id=session_id)
    print("\n--- Response to Important Dates ---")
    print(resp_dates)
    assert "02-11-2022" in resp_dates
    assert "14th October 2022" in resp_dates or "Exide Life" in resp_dates
    print("✅ Important Dates query passed!")

    # Query C: Data Layout Details
    q_layout = "What are the details regarding 'Data Layout'?"
    resp_layout = get_response(q_layout, user_id=user_id, session_id=session_id)
    print("\n--- Response to Data Layout ---")
    print(resp_layout)
    assert "Sl.No" in resp_layout or "Corporate Office Address" in resp_layout
    print("✅ Data Layout query passed!")

    # Query D: Coverage question on a directory document
    q_cov = "What are the key coverage benefits and limits?"
    resp_cov = get_response(q_cov, user_id=user_id, session_id=session_id)
    print("\n--- Response to Coverage Query on Directory ---")
    print(resp_cov[:450] + "...")
    assert "official regulatory directory" in resp_cov.lower() or "23 registered life insurance" in resp_cov.lower() or "registered insurers" in resp_cov.lower()
    print("✅ Coverage query explanation passed!")

    # Query E: Specific Insurer Drilldown
    q_lic = "Who is the Chairman and actuary of LIC?"
    resp_lic = get_response(q_lic, user_id=user_id, session_id=session_id)
    print("\n--- Response to LIC Drilldown ---")
    print(resp_lic)
    assert "M R Kumar" in resp_lic
    assert "Dinesh Pant" in resp_lic
    print("✅ Specific Insurer drilldown passed!")

    print("\n" + "="*70)
    print("TEST 3: Multi-Session Isolation (Doc A vs Doc B)")
    print("="*70)
    
    health_policy_sample = """
    # Arogya Supreme Health Insurance Policy
    Policy Number: SBIG-HLT-2024-998877
    Proposer: Divya Shah
    Sum Insured: Rs. 10,00,000
    Gross Premium: Rs. 14,500
    Room Rent Limit: Single Private AC Room
    Pre-Hospitalization: 60 Days
    Post-Hospitalization: 90 Days
    Waiting Period for Cataract: 24 Months
    """
    
    session_health = "session_health_policy"
    generate_pdf_summary_with_llama(
        health_policy_sample.encode("utf-8"), "SBI_Health_Policy.pdf",
        "Summarize this document.", user_id=user_id, session_id=session_health
    )

    health_ans = get_response("What is the sum insured and room rent limit?", user_id=user_id, session_id=session_health)
    print("\n--- session_health Query (Expected SBI Health) ---")
    print(health_ans)
    assert "10,00,000" in health_ans or "1000000" in health_ans or "Private AC Room" in health_ans

    dir_ans = get_response("Who is MD and CEO of Max Life?", user_id=user_id, session_id=session_id)
    print("\n--- session_insurance_test Query (Expected Max Life) ---")
    print(dir_ans)
    assert "Prashant Tripathy" in dir_ans

    print("\n✅ Multi-Session Isolation verified: Session 1 targets Health Policy, Session 2 targets Life Insurers!")
    print("\n" + "="*70)
    print("ALL TESTS PASSED SUCCESSFULLY! 🚀")
    print("="*70)

if __name__ == "__main__":
    test_universal_document_rag()
