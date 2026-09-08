# Backend: AI Chatbot & Candidate Matching Engine

FastAPI server powered by a local, offline NLP matching engine and MongoDB database.

---

## 📂 Backend Structure

```
backend/
├── chatBot/
│   ├── ai_engine.py         # Semantic NLP, vector similarity, and query router
│   ├── candidate_matcher.py # Multi-dimensional candidate evaluation & scoring
│   ├── resume_parser.py     # PDF/Text CV extractor & MongoDB ingestion
│   ├── knowledge_base.py    # Structured domain knowledge
│   └── chatbot.py           # Interface bridge
├── db/
│   ├── connection.py        # MongoDB client & collection handles
│   └── models.py            # Chat sessions & candidate database models
├── resumes/                 # Drop PDF/TXT candidate resumes here
├── import_cvs.py            # Batch CV import script
├── main.py                  # FastAPI server & REST endpoints
├── requirements.txt         # Python dependencies
└── .env                     # MongoDB connection URI
```

---

## 🚀 Running the Backend

```powershell
cd backend
.\venv\Scripts\uvicorn.exe main:app --reload
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Send message and receive list-wise AI response |
| `GET` | `/api/candidates` | List all candidate profiles in MongoDB |
| `POST` | `/api/candidates` | Add candidate record via JSON |
| `POST` | `/api/candidates/upload-cv` | Upload PDF/TXT resume and auto-parse into DB |
| `GET` | `/api/sessions/{user_id}` | Fetch chat history sessions |
| `GET` | `/api/sessions/{user_id}/{session_id}` | Fetch session message transcript |
| `DELETE` | `/api/sessions/{session_id}` | Delete a chat session |
