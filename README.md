# AI Chatbot & Candidate CV Identification Engine 🤖💼

A 100% local, intelligent **Conversational AI and Candidate CV Matching System** built with Python (FastAPI, scikit-learn, TF-IDF Vector Semantic Matching, MongoDB) and React.

---

## 🌟 Key Features

1. **Zero External API Dependency (No Gemini / OpenAI API Key Needed)**:
   - Runs completely offline/self-hosted on your machine.
   - 0 API costs, 0 quota limits, and sub-15ms response latency.

2. **Company Owner Candidate & CV Identification System**:
   - Matches candidate CVs stored in MongoDB against natural language queries.
   - Evaluates technical skills depth, years of experience, production projects, and education.
   - Computes a dynamic multi-dimensional **Match Score (0% – 100%)** with visual star ratings.

3. **Multi-Domain Candidate Support**:
   - **AI/ML & Data Science**: PyTorch, TensorFlow, LLMs, NLP, LangChain, Transformers, MLOps.
   - **CVD (Computer Vision Developers)**: OpenCV, YOLOv9, Deep Learning Vision, MediaPipe, Edge AI.
   - **BDE (Business Development Executives)**: B2B Sales, Lead Generation, CRM, Cold Outreach, Client Acquisition.
   - **Sales (Corporate & Enterprise Sales)**: High-ticket SaaS deals, Key Account Management, Salesforce.
   - **Full Stack & Backend**: React, Node.js, FastAPI, Python, MongoDB, PostgreSQL.
   - **DevOps & Cloud**: AWS, Docker, Kubernetes, Terraform, CI/CD pipelines.
   - **Digital Marketing**: SEO, Google Ads, Meta Ads, Growth Hacking.

4. **Multi-Turn Context & Dialogue Memory**:
   - Understands conversational context across turns (e.g. asking about a candidate or course, followed by *"What are the fees?"* or *"Show their full resume"*).

5. **Automated Resume & CV Ingestion (PDF / TXT)**:
   - Built-in resume parser extracts Name, Email, Phone, Skills, Experience, and Bio from uploaded `.pdf` / `.txt` files directly into MongoDB.

6. **Clean List-Wise Structured Responses**:
   - Formatted in clean numbered sections, bullet points (`•`), and indented details for maximum readability.

---

## 🧠 Candidate Matching & Ranking Algorithm

$$\text{Candidate Match Score} = 0.45 \times S_{\text{skill}} + 0.30 \times S_{\text{exp}} + 0.15 \times S_{\text{title}} + 0.10 \times S_{\text{project}}$$

| Component | Weight | Criteria Evaluated |
| :--- | :--- | :--- |
| **Skill Depth ($S_{\text{skill}}$)** | **45%** | Matches exact and related technical skills from the domain taxonomy. |
| **Experience ($S_{\text{exp}}$)** | **30%** | Seniority alignment (Senior 5+ yrs, Mid 3+ yrs, Junior/Freshers 0-2 yrs). |
| **Title Match ($S_{\text{title}}$)** | **15%** | Alignment with candidate's professional designation. |
| **Project Impact ($S_{\text{project}}$)** | **10%** | Real-world production systems (RAG, microservices, 60 FPS vision pipelines). |

---

## 📂 Project Structure

```
ChatBot/
├── README.md                    # Main Project Documentation (This File)
├── PROJECT_DOCUMENTATION.md     # Detailed Technical Reference
├── backend/
│   ├── chatBot/
│   │   ├── ai_engine.py         # Semantic NLP, vector similarity, and query router
│   │   ├── candidate_matcher.py # Multi-dimensional candidate evaluation & scoring
│   │   ├── resume_parser.py     # PDF/Text CV extractor & MongoDB ingestion
│   │   ├── knowledge_base.py    # Structured domain knowledge
│   │   └── chatbot.py           # Interface bridge
│   ├── db/
│   │   ├── connection.py        # MongoDB client & collection handles
│   │   └── models.py            # Chat sessions & candidate database models
│   ├── resumes/                 # Drop PDF/TXT candidate resumes here
│   ├── import_cvs.py            # Batch CV import script
│   ├── main.py                  # FastAPI server & REST endpoints
│   ├── requirements.txt         # Python dependencies
│   ├── README.md                # Backend API Documentation
│   └── .env                     # MongoDB connection URI
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # React chat interface with voice typing & sessions
│   │   └── App.css              # Styling with white-space pre-wrap list rendering
│   └── package.json
```

---

## 🚀 How to Run

### 1. Start the Backend Server:
```powershell
cd backend
.\venv\Scripts\uvicorn.exe main:app --reload
```
*Backend runs on `http://127.0.0.1:8000`*

### 2. Start the Frontend Application:
```powershell
cd frontend
npm run dev
```
*Frontend runs on `http://localhost:5173`*

---

## 💬 Example Chatbot Commands

### 🔍 Candidate Search:
- `I want AI ML best candidates in their skills and experience wise`
- `Show me BDE candidates with good experience`
- `Find CVD computer vision developers`
- `Show me enterprise sales candidates`
- `Find developers who know Python and React with 3+ years experience`
- `Who are our digital marketing candidates?`

### 📄 Candidate Resumes & Drilldown:
- `Show full resume of Aarav Mehta`
- `Show full resume of Harsh Varma`
- `Show full resume of Manish Patel`
- `Show full resume of Pooja Kapadia`

### 🎓 Academic & College Inquiries:
- `Tell me about MCA degree`
- `What is the fee structure for BCA?`
- `What is the eligibility criteria for MBA?`
- `How can I apply for admission?`
- `Where is the college located in Surat?`
- `What are the contact details and phone number?`

---

## 📥 How to Add More Candidate CVs

### Option A: Upload CV via API (PDF / TXT):
```bash
curl -X POST "http://127.0.0.1:8000/api/candidates/upload-cv" \
  -F "file=@/path/to/resume.pdf"
```

### Option B: Batch Import from `resumes/` Folder:
1. Place candidate `.pdf` or `.txt` CVs inside `backend/resumes/`.
2. Run the script:
```powershell
cd backend
.\venv\Scripts\python.exe import_cvs.py
```

### Option C: Direct JSON REST API:
```bash
curl -X POST "http://127.0.0.1:8000/api/candidates" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Karan Singhania",
    "title": "Senior AI / Deep Learning Specialist",
    "experience_years": 5.0,
    "seniority": "Senior",
    "email": "karan.singhania@example.com",
    "phone": "+91 98765 43210",
    "location": "Surat",
    "education": "M.Tech in AI",
    "skills": ["AI/ML", "PyTorch", "LLMs", "NLP", "LangChain", "Docker"],
    "project_highlights": "Built enterprise RAG pipelines and fine-tuned open-source LLMs.",
    "bio": "Senior AI engineer with 5 years experience in deep learning."
  }'
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic
- **NLU & AI**: Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity, N-grams), Regular Expressions, Multi-factor Ranking
- **Resume Processing**: PyPDF, Python-Multipart
- **Database**: MongoDB Atlas (PyMongo)
- **Frontend**: React 18, Vite, Web Speech API (Voice Typing)
