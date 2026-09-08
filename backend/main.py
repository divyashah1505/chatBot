from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio

from chatBot.chatbot import get_response
from chatBot.llama_summarizer import generate_pdf_summary_with_llama
from chatBot.document_memory import doc_memory
from chatBot.long_term_memory import long_term_memory
from chatBot.resume_parser import process_and_save_cv
from chatBot.local_llm_provider import local_llm_provider
from db.models import (
    create_session, 
    add_message_to_session, 
    get_user_sessions, 
    get_session_messages, 
    delete_session,
    get_all_candidates,
    insert_candidate,
    seed_default_candidates_if_empty,
    get_user_memories,
    clear_user_memories,
    get_all_user_documents
)


app = FastAPI(title="Company Owner AI Assistant Backend")


# CORS configuration - allows any other website (e.g. localhost:3000, live server, or custom domains) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"^https?://.*",
)



@app.on_event("startup")
async def startup_event():
    # Seed default candidate CVs into MongoDB if empty
    await asyncio.to_thread(seed_default_candidates_if_empty)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None


class CandidateCreateRequest(BaseModel):
    name: str
    title: str
    experience_years: float
    seniority: Optional[str] = "Mid"
    email: str
    phone: str
    location: Optional[str] = "Surat"
    education: str
    skills: List[str]
    project_highlights: str
    bio: str
    status: Optional[str] = "Available"


@app.get("/")
async def home():
    return {"message": "AI Assistant Backend is running!"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint for external websites to verify API status."""
    return {"status": "ok", "service": "ChatBot API", "version": "1.0.0"}



@app.get("/api/models/status")
async def get_models_status():
    """Returns status of local LLM providers (Ollama, LM Studio, Offline RAG)."""
    return local_llm_provider.get_provider_status()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    history = []

    try:
        if not session_id:
            title = request.message[:30] + "..." if len(request.message) > 30 else request.message
            session_id = await asyncio.to_thread(create_session, request.user_id, title)
        else:
            history = await asyncio.to_thread(get_session_messages, session_id, request.user_id)
    except Exception as e:
        print(f"[Chat] Session lookup skipped: {e}")
        session_id = session_id or "local_session"

    # Run AI engine logic in thread pool with user_id and session_id for session-aware RAG
    response = await asyncio.to_thread(get_response, request.message, history, request.user_id, session_id)

    # Save messages safely
    try:
        if session_id:
            await asyncio.gather(
                asyncio.to_thread(add_message_to_session, session_id, "user", request.message),
                asyncio.to_thread(add_message_to_session, session_id, "bot", response),
            )
    except Exception as e:
        print(f"[Chat] Message save skipped: {e}")

    return {
        "reply": response,
        "session_id": session_id or "local_session"
    }


from chatBot.prescription_analyzer import local_prescription_analyzer

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt", ".md", ".json", ".log", ".xml",
    ".rtf", ".odt", ".ods", ".yaml", ".yml", ".html", ".htm", ".tsv", ".ini", ".conf", ".sql",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

@app.post("/api/analyze-prescription")
@app.post("/api/summarize-pdf")
@app.post("/api/summarize-document")
async def summarize_pdf_endpoint(
    user_id: str = Form(...),
    session_id: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """Upload any document or handwritten prescription image (PDF, Image, Word, Excel, CSV, Text) + custom user prompt and get a verified answer with persistent memory."""
    try:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return {
                "reply": f"⚠️ Unsupported file format `{ext}`. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                "session_id": session_id or "local_session",
                "filename": file.filename,
                "faqs": []
            }

        file_bytes = await file.read()
        user_prompt = prompt.strip() if isinstance(prompt, str) and prompt.strip() else "Summarize this document."
        user_id_val = str(user_id) if isinstance(user_id, str) and user_id else "default_user"
        session_id_val = str(session_id) if isinstance(session_id, str) and session_id else None

        # Check if uploaded file is a prescription image
        if ext in IMAGE_EXTENSIONS or any(k in user_prompt.lower() for k in ["prescription", "medicine", "doctor", "rx", "dosage"]):
            prescription_result = await asyncio.to_thread(
                local_prescription_analyzer.analyze_prescription, file_bytes, file.filename, user_prompt
            )
            summary = prescription_result.get("reply", "")
            faqs = prescription_result.get("faqs", [])
        else:
            # Standard Universal Document Summarization & Vector RAG with Session binding
            summary = await asyncio.to_thread(
                generate_pdf_summary_with_llama, file_bytes, file.filename, user_prompt, user_id_val, session_id_val
            )
            doc_hash = doc_memory.compute_document_hash(file_bytes, file.filename)
            entry = doc_memory.get_document(doc_hash)
            faqs = entry.get_faqs() if entry else []

        # Create or reuse chat session if available (offline-safe)
        try:
            if not session_id_val:
                title = f"Rx: {file.filename[:25]}" if ext in IMAGE_EXTENSIONS else f"Doc: {file.filename[:25]}"
                session_id_val = await asyncio.to_thread(create_session, user_id_val, title)

            # Bind session to document in doc_memory
            doc_hash = doc_memory.compute_document_hash(file_bytes, file.filename)
            if session_id_val:
                doc_memory.bind_session_document(session_id_val, doc_hash)
            doc_memory.bind_user_document(user_id_val, doc_hash)

            user_msg = f"📎 {file.filename} — {user_prompt}"
            if session_id_val:
                await asyncio.gather(
                    asyncio.to_thread(add_message_to_session, session_id_val, "user", user_msg),
                    asyncio.to_thread(add_message_to_session, session_id_val, "bot", summary),
                )
        except Exception as e:
            print(f"[Doc] Session/message save skipped in offline mode: {e}")
            session_id_val = session_id_val or "local_session"

        return {
            "reply": summary,
            "session_id": session_id_val or "local_session",
            "filename": file.filename,
            "faqs": faqs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "reply": f"⚠️ An error occurred while processing **{file.filename}**: {str(e)}",
            "session_id": session_id or "local_session",
            "filename": file.filename,
            "faqs": []
        }



@app.get("/api/documents/faqs")
async def get_document_faqs(filename: Optional[str] = None):
    """Retrieves generated FAQs for active or specified remembered document."""
    if filename:
        fn_lower = filename.lower()
        if fn_lower in doc_memory._name_index:
            doc_hash = doc_memory._name_index[fn_lower]
            entry = doc_memory.get_document(doc_hash)
            if entry:
                return {"filename": entry.filename, "faqs": entry.get_faqs()}
    
    # Return latest document FAQs if available
    if doc_memory._memory_store:
        latest_entry = list(doc_memory._memory_store.values())[-1]
        return {"filename": latest_entry.filename, "faqs": latest_entry.get_faqs()}
    
    return {"filename": None, "faqs": []}


# Long-Term User Memory Management Endpoints (ChatGPT-Style)
@app.get("/api/user/memory/{user_id}")
async def get_user_memory_details(user_id: str):
    """Retrieve long-term memories, frequent questions, and uploaded documents for user."""
    frequent_stats = await asyncio.to_thread(long_term_memory.get_frequent_queries_and_topics, user_id)
    memories = await asyncio.to_thread(get_user_memories, user_id)
    user_docs = await asyncio.to_thread(get_all_user_documents, user_id)
    return {
        "user_id": user_id,
        "analytics": frequent_stats,
        "saved_memories": memories,
        "uploaded_documents": user_docs
    }

@app.delete("/api/user/memory/{user_id}")
async def delete_user_memories_endpoint(user_id: str):
    """Clear all persistent memories for user."""
    success = await asyncio.to_thread(clear_user_memories, user_id)
    return {"status": "success" if success else "error", "message": "User memories cleared."}


# Document Memory Management Endpoints
@app.get("/api/documents/memory")
async def get_document_memory_stats():
    """Retrieve stats about currently remembered documents in memory."""
    return doc_memory.get_stats()

@app.delete("/api/documents/memory")
async def clear_document_memory():
    """Clear in-memory document cache and reset lifecycle."""
    doc_memory.clear_memory()
    return {"status": "success", "message": "Document memory cache cleared successfully."}



# Session Management Endpoints
@app.get("/api/sessions/{user_id}")
async def get_sessions(user_id: str):
    sessions = await asyncio.to_thread(get_user_sessions, user_id)
    return sessions

@app.get("/api/sessions/{user_id}/{session_id}")
async def get_session_details(user_id: str, session_id: str):
    messages = await asyncio.to_thread(get_session_messages, session_id, user_id)
    return {"messages": messages}

@app.delete("/api/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    success = await asyncio.to_thread(delete_session, session_id)
    if success:
        return {"status": "success"}
    return {"status": "error"}


# Candidate Management Endpoints
@app.get("/api/candidates")
async def get_candidates():
    candidates = await asyncio.to_thread(get_all_candidates)
    return {"count": len(candidates), "candidates": candidates}

@app.post("/api/candidates")
async def create_candidate(candidate: CandidateCreateRequest):
    success = await asyncio.to_thread(insert_candidate, candidate.dict())
    if success:
        return {"status": "success", "message": "Candidate added successfully"}
    return {"status": "error", "message": "Failed to add candidate"}

@app.post("/api/candidates/upload-cv")
async def upload_candidate_cv(file: UploadFile = File(...)):
    """Upload PDF or Text CV file, automatically parse details and store into MongoDB."""
    try:
        file_bytes = await file.read()
        candidate = await asyncio.to_thread(process_and_save_cv, file_bytes, file.filename)
        return {
            "status": "success",
            "message": f"Successfully parsed and stored CV for {candidate['name']}",
            "candidate": candidate
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Host on 0.0.0.0 so other devices on local network or local ports can access it
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)