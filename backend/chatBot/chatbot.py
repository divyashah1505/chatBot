"""
Chatbot Interface Module (100% Offline & Local).
Connects to Local LLMs (Ollama / LM Studio) or falls back seamlessly to the built-in local AI engine.
Maintains full multi-turn conversational memory, cross-session profile, and uploaded document awareness.
"""
from typing import Optional, List, Dict, Any
from .ai_engine import generate_ai_response
from .knowledge_base import COLLEGE_INFO, PROGRAMS, ADMISSION_GUIDE, SCHOLARSHIPS_INFO
from .candidate_matcher import candidate_matcher
from .document_memory import doc_memory
from .long_term_memory import long_term_memory
from .local_llm_provider import local_llm_provider, is_ollama_running, is_lm_studio_running
from .gemini_provider import gemini_provider

def is_local_llm_running(timeout: float = 0.2) -> bool:
    """Checks if Ollama or LM Studio is currently listening."""
    _, _, is_online = local_llm_provider.detect_active_provider()
    return is_online


def get_available_local_model() -> str:
    """Fetch active model ID dynamically."""
    if gemini_provider.is_available():
        return gemini_provider._active_model or "gemini-3.6-flash"
    _, model_name, _ = local_llm_provider.detect_active_provider()
    return model_name


def build_system_context(message: str, user_id: str = "default_user", session_id: Optional[str] = None) -> str:
    """Provides relevant domain instructions, long-term memory, and session-specific document knowledge into prompt."""
    clean = message.lower()
    context_sections = [
        "You are an advanced, helpful, articulate, and highly capable AI Assistant (like ChatGPT and Gemini).",
        "",
        "### RESPONSE FORMATTING RULES (CRITICAL):",
        "- Always format your answers using clean, standard GitHub Flavored Markdown.",
        "- Structure long or complex responses with clear, descriptive headings (e.g. `### Heading`).",
        "- Use bullet points (`- `) or numbered lists (`1. `, `2. `) for features, steps, and options.",
        "- ALWAYS wrap technical code, shell commands, or syntax snippets in fenced code blocks with the exact language tag (e.g. ```python, ```javascript, ```sql, ```bash, ```json, ```html).",
        "- Use markdown tables (`| Col 1 | Col 2 |`) whenever comparing items, summarizing technical specs, or listing data.",
        "- Use bold text (`**term**`) to emphasize key terms, numbers, concepts, and conclusions.",
        "- Use blockquotes (`> Note:` or `> Tip:`) for key takeaways, insights, or warnings.",
        "- Deliver comprehensive, well-spaced, and engaging responses. Never output raw unstructured walls of text.",
        "- You have persistent memory of past conversations, uploaded documents, and facts across all sessions."
    ]

    # Inject Cross-Session Long-Term Memory (User facts, preferences, frequent topics)
    try:
        mem_ctx = long_term_memory.get_system_prompt_memory_context(user_id)
        if mem_ctx:
            context_sections.append(mem_ctx)
    except Exception as e:
        print(f"[Memory] Context generation skipped: {e}")

    # Add context from the session-bound or matched uploaded document
    target_entry = doc_memory.get_best_document_match(message, session_id=session_id, user_id=user_id)
    if target_entry and target_entry.vector_store:
        matched_chunks = target_entry.vector_store.search(message, top_k=4, min_score=0.01)
        if matched_chunks:
            doc_lines = [f"\n[KNOWLEDGE FROM UPLOADED DOCUMENT: `{target_entry.filename}`]"]
            for m in matched_chunks:
                p_tag = f" (Page {m.get('primary_page')})" if m.get("primary_page") else ""
                doc_lines.append(f"Excerpt{p_tag}:\n{m['text']}")
            doc_lines.append(
                f"[END UPLOADED DOCUMENT KNOWLEDGE FROM `{target_entry.filename}`]\n"
                f"Use the uploaded document knowledge above to answer questions about `{target_entry.filename}` accurately and comprehensively."
            )
            context_sections.append("\n".join(doc_lines))
    elif doc_memory._memory_store:
        doc_matches = doc_memory.search_all_documents(message, top_k_per_doc=2)
        if doc_matches:
            doc_lines = ["\n[KNOWLEDGE FROM USER'S UPLOADED DOCUMENTS IN MEMORY]"]
            for m in doc_matches:
                p_tag = f" (Page {m.get('primary_page')})" if m.get("primary_page") else ""
                doc_lines.append(f"Source Document: `{m['filename']}`{p_tag}\nContent:\n{m['text']}")
            doc_lines.append("[END UPLOADED DOCUMENT KNOWLEDGE]")
            context_sections.append("\n".join(doc_lines))

    # Add college knowledge if related to academics
    if any(k in clean for k in ["college", "mca", "bca", "mba", "bba", "admission", "fee", "syllabus", "facility", "placement", "course", "degree"]):
        context_sections.append(
            f"\nCollege Info:\n- Name: {COLLEGE_INFO['name']}\n- Location: {COLLEGE_INFO['location']}\n"
            f"- Address: {COLLEGE_INFO['address']}\n- Phone: {COLLEGE_INFO['phone']}\n- Email: {COLLEGE_INFO['email']}\n"
            f"Available Programs: MCA, BCA, MBA, BBA.\n"
            f"MCA: {PROGRAMS['mca']['duration']}, Fees: {PROGRAMS['mca']['fees']['per_semester']}.\n"
            f"BCA: {PROGRAMS['bca']['duration']}, Fees: {PROGRAMS['bca']['fees']['per_semester']}.\n"
            f"MBA: {PROGRAMS['mba']['duration']}, Fees: {PROGRAMS['mba']['fees']['per_semester']}.\n"
            f"BBA: {PROGRAMS['bba']['duration']}, Fees: {PROGRAMS['bba']['fees']['per_semester']}."
        )

    return "\n".join(context_sections)


def call_gemini_chat(message: str, history: list = None, user_id: str = "default_user", session_id: Optional[str] = None) -> Optional[str]:
    """
    Calls Google Gemini with multi-turn conversation memory, system prompt, and document context.
    """
    system_prompt = build_system_context(message, user_id=user_id, session_id=session_id)
    messages = []

    # Append past conversation history (last 14 turns)
    if history:
        for item in history[-14:]:
            sender = item.get("sender") or item.get("role", "")
            text = item.get("text") or item.get("content", "")
            if text:
                role = "assistant" if sender in ["bot", "assistant"] else "user"
                messages.append({"role": role, "content": text})

    # Append current user message
    messages.append({"role": "user", "content": message})

    return gemini_provider.generate_chat_response(
        messages=messages,
        system_instruction=system_prompt,
        temperature=0.4,
        max_tokens=2048
    )


def call_llama_chat(message: str, history: list = None, user_id: str = "default_user", session_id: Optional[str] = None) -> str:
    """
    Calls Local LLM (Ollama / LM Studio) with multi-turn conversation memory and session-specific document profile.
    """
    system_prompt = build_system_context(message, user_id=user_id, session_id=session_id)
    messages = [{"role": "system", "content": system_prompt}]

    # Append past conversation history (last 12 turns)
    if history:
        for item in history[-12:]:
            sender = item.get("sender") or item.get("role", "")
            text = item.get("text") or item.get("content", "")
            if text:
                role = "assistant" if sender in ["bot", "assistant"] else "user"
                messages.append({"role": role, "content": text})

    # Append current user message
    messages.append({"role": "user", "content": message})

    completion = local_llm_provider.generate_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    if completion:
        return completion[0]
    return ""


def get_response(
    message: str,
    history: list = None,
    user_id: str = "default_user",
    session_id: Optional[str] = None
) -> str:
    """
    Primary Chatbot Response Gateway.
    1. Priority: Long-Term Memory Meta Queries (frequent questions, user memory).
    2. Priority: Candidate database ranking queries for company owners.
    3. Priority: Medicine / Prescription Direct Lookups.
    4. Primary: Google Gemini Generative AI (Modern actual AI with proper Markdown formatting).
    5. Primary: Local LLMs (Ollama / LM Studio with full multi-turn memory & document awareness).
    6. Fallback: Session-Aware Document Memory Search & Universal Offline RAG Engine.
    7. Fallback: Built-in local offline engine.
    """
    clean_msg = message.lower().strip()

    # Extract and save user profile facts seamlessly
    try:
        long_term_memory.extract_and_save_facts(user_id, message)
    except Exception:
        pass

    # Priority 1: Meta questions about memory & history (e.g. "which questions do I generally ask most?")
    if long_term_memory.is_memory_meta_query(clean_msg):
        try:
            return long_term_memory.answer_memory_query(user_id, clean_msg)
        except Exception as e:
            print(f"[Memory] Error answering meta query: {e}")

    # Priority 2: Candidate database ranking queries for company owners
    if candidate_matcher.is_candidate_query(clean_msg) or candidate_matcher.is_single_candidate_drilldown(clean_msg):
        try:
            return candidate_matcher.evaluate_candidates(clean_msg)
        except Exception as e:
            print(f"[Matcher] Error: {e}")

    # Priority 3: Medicine / Prescription Direct Lookups
    from .prescription_analyzer import local_prescription_analyzer
    if any(k in clean_msg for k in ["medicine", "tablet", "dosage", "prescription", "syrup", "capsule", "side effect", "timing"]):
        med_reply = local_prescription_analyzer.answer_medicine_query(message)
        if med_reply:
            return med_reply

    # Primary 4: Google Gemini AI (State-of-the-Art Generative AI with proper Markdown output)
    if gemini_provider.is_available():
        try:
            gemini_reply = call_gemini_chat(message, history, user_id=user_id, session_id=session_id)
            if gemini_reply and gemini_reply.strip():
                return gemini_reply.strip()
        except Exception as e:
            print(f"[ChatBot] Gemini call error: {e}. Trying local LLM / fallback.")

    # Primary 5: Check if Local LLM server (Ollama or LM Studio) is running
    _, _, is_online = local_llm_provider.detect_active_provider()
    if is_online:
        try:
            llm_reply = call_llama_chat(message, history, user_id=user_id, session_id=session_id)
            if llm_reply and llm_reply.strip():
                return llm_reply.strip()
        except Exception as e:
            print(f"[ChatBot] Local LLM call error: {e}. Using built-in local engine.")

    # Fallback 6: Session-Aware Document Memory Search & Targeted Analysis
    target_entry = doc_memory.get_best_document_match(message, session_id=session_id, user_id=user_id)
    if target_entry:
        from .llama_summarizer import generate_universal_offline_analysis, classify_user_prompt_intent
        intent = classify_user_prompt_intent(clean_msg)
        analysis_reply = generate_universal_offline_analysis(
            target_entry.vector_store,
            target_entry.filename,
            target_entry.raw_text,
            message,
            intent
        )
        target_entry.record_interaction(message, analysis_reply)
        return analysis_reply

    # Fallback 7: Built-in local offline engine
    return generate_ai_response(message, history)