from datetime import datetime
from bson import ObjectId
from db.connection import sessions_collection, candidates_collection

# --- Session & Chat Management ---

def create_session(user_id, title):
    try:
        session = {
            "user_id": user_id,
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "messages": []
        }
        result = sessions_collection.insert_one(session)
        return str(result.inserted_id)
    except Exception as e:
        print(f"[DB] Offline/Skipped session creation: {e}")
        return f"local_session_{int(datetime.utcnow().timestamp())}"

def add_message_to_session(session_id, sender, text):
    try:
        if not session_id or session_id.startswith("local_session_"):
            return
        message = {
            "sender": sender,
            "text": text,
            "timestamp": datetime.utcnow()
        }
        
        sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
    except Exception as e:
        print(f"[DB] Offline/Skipped message saving: {e}")

def get_user_sessions(user_id):
    try:
        cursor = sessions_collection.find(
            {"user_id": user_id},
            {"messages": 0}
        ).sort("updated_at", -1)
        
        sessions = []
        for doc in cursor:
            sessions.append({
                "id": str(doc["_id"]),
                "title": doc.get("title", "New Chat"),
                "updated_at": doc["updated_at"].isoformat() if isinstance(doc.get("updated_at"), datetime) else ""
            })
        return sessions
    except Exception as e:
        print(f"[DB] Offline/Skipped get_user_sessions: {e}")
        return []

def get_session_messages(session_id, user_id):
    try:
        session = sessions_collection.find_one({
            "_id": ObjectId(session_id),
            "user_id": user_id
        })
        if session:
            messages = session.get("messages", [])
            for msg in messages:
                if isinstance(msg.get("timestamp"), datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()
            return messages
        return []
    except Exception:
        return []

def delete_session(session_id):
    try:
        sessions_collection.delete_one({"_id": ObjectId(session_id)})
        return True
    except Exception:
        return False


# --- Long-Term User Memory & Cross-Session Document Tracking ---

def get_user_all_past_messages(user_id):
    """Retrieves all user and bot messages across all chat sessions for long-term memory analysis."""
    try:
        sessions = sessions_collection.find({"user_id": user_id})
        all_messages = []
        for s in sessions:
            for m in s.get("messages", []):
                all_messages.append({
                    "session_id": str(s["_id"]),
                    "session_title": s.get("title", ""),
                    "sender": m.get("sender", ""),
                    "text": m.get("text", ""),
                    "timestamp": m.get("timestamp", datetime.utcnow())
                })
        return all_messages
    except Exception as e:
        print(f"[DB] get_user_all_past_messages fallback: {e}")
        return []

def save_user_memory(user_id, fact_key, fact_value, category="preference"):
    """Saves or updates a persistent memory fact about the user."""
    try:
        from db.connection import user_memory_collection
        user_memory_collection.update_one(
            {"user_id": user_id, "key": fact_key},
            {
                "$set": {
                    "value": fact_value,
                    "category": category,
                    "updated_at": datetime.utcnow()
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        return True
    except Exception as e:
        print(f"[DB] save_user_memory fallback: {e}")
        return False

def get_user_memories(user_id):
    """Retrieves all persistent memory facts saved for the user."""
    try:
        from db.connection import user_memory_collection
        cursor = user_memory_collection.find({"user_id": user_id})
        memories = []
        for doc in cursor:
            memories.append({
                "id": str(doc["_id"]),
                "key": doc.get("key", ""),
                "value": doc.get("value", ""),
                "category": doc.get("category", "general"),
                "updated_at": doc.get("updated_at", "").isoformat() if isinstance(doc.get("updated_at"), datetime) else ""
            })
        return memories
    except Exception as e:
        print(f"[DB] get_user_memories fallback: {e}")
        return []

def clear_user_memories(user_id):
    """Clears all long-term memories for a specific user."""
    try:
        from db.connection import user_memory_collection
        user_memory_collection.delete_many({"user_id": user_id})
        return True
    except Exception:
        return False

def save_document_metadata(user_id, filename, doc_hash, summary, total_chunks, category="GENERAL"):
    """Stores metadata of uploaded documents permanently for cross-session access."""
    try:
        from db.connection import documents_collection
        documents_collection.update_one(
            {"user_id": user_id, "doc_hash": doc_hash},
            {
                "$set": {
                    "filename": filename,
                    "summary": summary[:1500] if summary else "",
                    "total_chunks": total_chunks,
                    "category": category,
                    "last_accessed": datetime.utcnow()
                },
                "$inc": {"upload_count": 1},
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
        return True
    except Exception as e:
        print(f"[DB] save_document_metadata fallback: {e}")
        return False

def get_all_user_documents(user_id):
    """Fetches all documents previously uploaded by the user across any session."""
    try:
        from db.connection import documents_collection
        cursor = documents_collection.find({"user_id": user_id}).sort("last_accessed", -1)
        docs = []
        for d in cursor:
            docs.append({
                "id": str(d["_id"]),
                "filename": d.get("filename", ""),
                "doc_hash": d.get("doc_hash", ""),
                "summary": d.get("summary", ""),
                "category": d.get("category", "GENERAL"),
                "upload_count": d.get("upload_count", 1),
                "last_accessed": d.get("last_accessed", "").isoformat() if isinstance(d.get("last_accessed"), datetime) else ""
            })
        return docs
    except Exception as e:
        print(f"[DB] get_all_user_documents fallback: {e}")
        return []


# --- Candidate / CV Management ---

SAMPLE_CANDIDATES = [
    {
        "name": "Aarav Mehta",
        "title": "Senior AI / Machine Learning Engineer",
        "experience_years": 5.5,
        "seniority": "Senior",
        "email": "aarav.mehta@techvision.com",
        "phone": "+91 98234 56789",
        "location": "Surat / Bangalore",
        "education": "M.Tech in Artificial Intelligence (IIT Bombay)",
        "skills": [
            "AI/ML", "Machine Learning", "Deep Learning", "Python", "PyTorch", 
            "TensorFlow", "NLP", "LLMs", "LangChain", "Transformers", 
            "MLOps", "Docker", "AWS SageMaker", "FastAPI"
        ],
        "project_highlights": (
            "Architected an enterprise RAG system indexing 10M+ documents with sub-second retrieval. "
            "Fine-tuned open-source Llama 3 & Mistral models for automated code review, reducing developer turnaround by 40%. "
            "Deployed production MLOps pipeline on AWS SageMaker with automated model drift detection."
        ),
        "bio": "Experienced AI/ML Engineer with 5.5+ years of hands-on expertise building production NLP, LLM, and Deep Learning applications at enterprise scale.",
        "status": "Available / Looking for Senior AI roles"
    },
    {
        "name": "Priya Sharma",
        "title": "Lead Computer Vision & Data Scientist",
        "experience_years": 6.0,
        "seniority": "Lead / Senior",
        "email": "priya.sharma@aimodels.io",
        "phone": "+91 98345 67890",
        "location": "Surat / Pune",
        "education": "B.Tech in Computer Science & PG Diploma in Data Science",
        "skills": [
            "AI/ML", "Computer Vision", "Deep Learning", "OpenCV", "PyTorch", 
            "Python", "YOLOv8", "Image Segmentation", "Scikit-Learn", 
            "Pandas", "SQL", "Model Optimization"
        ],
        "project_highlights": (
            "Developed autonomous defect inspection system using YOLOv8 with 98.4% precision on high-speed manufacturing lines. "
            "Built predictive customer retention models saving $2.2M annually. "
            "Published research paper on low-latency edge inference for embedded vision devices."
        ),
        "bio": "Lead Data Scientist and Computer Vision specialist with 6 years experience in real-time image processing, neural networks, and statistical machine learning.",
        "status": "Available / Immediate Joiner"
    },
    {
        "name": "Rohan Patel",
        "title": "AI/ML Developer & Python Specialist",
        "experience_years": 3.0,
        "seniority": "Mid-level",
        "email": "rohan.patel@devhub.in",
        "phone": "+91 98456 78901",
        "location": "Surat / Ahmedabad",
        "education": "B.E. in Information Technology",
        "skills": [
            "AI/ML", "Python", "Scikit-Learn", "PyTorch", "NLP", 
            "BERT", "FastAPI", "Hugging Face", "MongoDB", "Docker", "Git"
        ],
        "project_highlights": (
            "Built intelligent resume parser and semantic candidate matching engine using TF-IDF and Sentence-Transformers. "
            "Created multilingual customer sentiment analysis dashboard processing 50k social media mentions daily."
        ),
        "bio": "Dynamic Mid-level AI/ML developer with 3 years experience specializing in NLP, modern Python frameworks, and clean RESTful API integration.",
        "status": "Available / 2 weeks notice"
    },
    {
        "name": "Sneha Joshi",
        "title": "Junior AI & Data Analyst",
        "experience_years": 1.5,
        "seniority": "Junior / Entry-Level",
        "email": "sneha.joshi@careers.net",
        "phone": "+91 98567 89012",
        "location": "Surat",
        "education": "MCA (Master of Computer Applications)",
        "skills": [
            "AI/ML", "Python", "NumPy", "Pandas", "Scikit-Learn", 
            "Data Visualization", "Matplotlib", "Seaborn", "SQL", "Tableau"
        ],
        "project_highlights": (
            "Implemented LSTM-based time-series forecasting model for retail sales with 92% directional accuracy. "
            "Designed automated ETL pipelines and interactive executive dashboards in Tableau and Python."
        ),
        "bio": "Enthusiastic Junior AI Engineer and MCA graduate with strong foundations in data structures, statistical analysis, and machine learning pipelines.",
        "status": "Available / Immediate Joiner"
    },
    {
        "name": "Vikram Desai",
        "title": "Senior Full Stack & AI Integrations Engineer",
        "experience_years": 5.0,
        "seniority": "Senior",
        "email": "vikram.desai@stackbuilders.com",
        "phone": "+91 98678 90123",
        "location": "Surat / Mumbai",
        "education": "B.Tech in Computer Engineering",
        "skills": [
            "Full Stack", "Python", "FastAPI", "React", "Node.js", 
            "Next.js", "MongoDB", "PostgreSQL", "Tailwind CSS", "LangChain", "Docker"
        ],
        "project_highlights": (
            "Architected full-stack enterprise SaaS platform serving 150,000+ monthly active users with React and FastAPI backend. "
            "Integrated intelligent generative AI workflows using LangChain and vector databases."
        ),
        "bio": "Senior Full Stack Engineer with 5 years experience creating scalable web applications and seamless AI integrations.",
        "status": "Available / 1 month notice"
    },
    {
        "name": "Ananya Roy",
        "title": "Full Stack Developer (MERN)",
        "experience_years": 3.5,
        "seniority": "Mid-level",
        "email": "ananya.roy@webcraft.io",
        "phone": "+91 98789 01234",
        "location": "Surat / Hyderabad",
        "education": "BCA & MCA",
        "skills": [
            "Full Stack", "React", "Node.js", "Express.js", "MongoDB", 
            "JavaScript", "TypeScript", "Redux", "REST APIs", "Tailwind CSS"
        ],
        "project_highlights": (
            "Developed end-to-end multi-vendor e-commerce portal with real-time inventory synchronization using Socket.io and Stripe payments. "
            "Built responsive admin dashboard with granular role-based access control."
        ),
        "bio": "Passionate Full Stack Developer with 3.5 years of experience delivering high-performance responsive web applications using the MERN stack.",
        "status": "Available / Immediate Joiner"
    },
    {
        "name": "Kabir Shah",
        "title": "Senior DevOps & Cloud Infrastructure Engineer",
        "experience_years": 5.5,
        "seniority": "Senior",
        "email": "kabir.shah@cloudinfra.tech",
        "phone": "+91 98890 12345",
        "location": "Surat / Remote",
        "education": "B.E. in Computer Science",
        "skills": [
            "DevOps", "AWS", "Docker", "Kubernetes", "CI/CD", 
            "Terraform", "Linux", "Jenkins", "Python", "Prometheus", "GitOps"
        ],
        "project_highlights": (
            "Engineered zero-downtime CI/CD deployment pipelines using GitHub Actions and ArgoCD on Kubernetes. "
            "Optimized AWS cloud infrastructure with Terraform, slashing monthly cloud hosting costs by 38%."
        ),
        "bio": "Senior DevOps Engineer with 5.5 years specializing in cloud automation, microservices orchestration, containerization, and infrastructure as code.",
        "status": "Available / 2 weeks notice"
    },
    {
        "name": "Neha Verma",
        "title": "Senior Frontend UI/UX & React Developer",
        "experience_years": 4.0,
        "seniority": "Senior",
        "email": "neha.verma@uicraft.dev",
        "phone": "+91 98901 23456",
        "location": "Surat / Delhi",
        "education": "B.Tech in Information Technology",
        "skills": [
            "Frontend", "React", "Next.js", "TypeScript", "Tailwind CSS", 
            "Redux Toolkit", "Figma", "Web Performance", "Jest", "UI/UX"
        ],
        "project_highlights": (
            "Created component library and design system adopted by 12 cross-functional teams, speeding up feature delivery by 50%. "
            "Achieved perfect 100 Lighthouse performance score on core fintech consumer web application."
        ),
        "bio": "Frontend engineer with 4 years experience crafting pixel-perfect, accessible, and ultra-fast user interfaces in React and Next.js.",
        "status": "Available / Immediate Joiner"
    },
    {
        "name": "Manish Patel",
        "title": "Senior Business Development Executive (BDE)",
        "experience_years": 4.5,
        "seniority": "Senior",
        "email": "manish.patel@growthleads.io",
        "phone": "+91 98111 22233",
        "location": "Surat / Ahmedabad",
        "education": "BBA & MBA in Marketing",
        "skills": [
            "BDE", "Business Development", "B2B Sales", "Lead Generation", 
            "Client Acquisition", "Cold Calling", "CRM", "Pipeline Management", "Negotiation", "HubSpot"
        ],
        "project_highlights": (
            "Closed $1.4M in annual enterprise software contracts. "
            "Generated 150+ qualified B2B leads monthly via LinkedIn Sales Navigator and structured cold outreach campaigns."
        ),
        "bio": "High-performing BDE with 4.5 years experience in B2B sales pipelines, client acquisition, and high-value contract negotiations.",
        "status": "Available / Immediate Joiner"
    },
    {
        "name": "Pooja Kapadia",
        "title": "Enterprise Sales & Key Account Manager",
        "experience_years": 6.0,
        "seniority": "Lead / Senior",
        "email": "pooja.kapadia@corpsales.com",
        "phone": "+91 98222 33344",
        "location": "Surat / Mumbai",
        "education": "MBA in Sales & Marketing",
        "skills": [
            "Sales", "Enterprise Sales", "B2B", "Account Management", 
            "Revenue Growth", "Closing Deals", "Client Relationship", "Salesforce", "Cold Outreach"
        ],
        "project_highlights": (
            "Exceeded quarterly revenue targets by 135% for 3 consecutive years. "
            "Managed and scaled enterprise SaaS client accounts worth over $2.8M with 96% client retention rate."
        ),
        "bio": "Senior Enterprise Sales Manager with 6 years experience in B2B sales, corporate negotiations, and revenue expansion.",
        "status": "Available / 1 month notice"
    },
    {
        "name": "Harsh Varma",
        "title": "Senior Computer Vision Developer (CVD)",
        "experience_years": 4.5,
        "seniority": "Senior",
        "email": "harsh.varma@visionai.tech",
        "phone": "+91 98333 44455",
        "location": "Surat / Bangalore",
        "education": "B.Tech in Computer Science (AI/ML Specialization)",
        "skills": [
            "CVD", "Computer Vision", "OpenCV", "YOLOv9", "YOLOv8", 
            "Deep Learning Vision", "Object Detection", "Image Segmentation", "PyTorch Vision", "Python", "MediaPipe"
        ],
        "project_highlights": (
            "Engineered automated fabric quality and defect inspection system using YOLOv9 with 99.1% detection rate on Surat textile lines. "
            "Developed real-time multi-camera facial recognition and biometric attendance pipeline processing 60 FPS video streams."
        ),
        "bio": "Specialist Computer Vision Developer (CVD) with 4.5 years hands-on experience in neural vision models, real-time edge processing, and OpenCV.",
        "status": "Available / Immediate Joiner"
    },
    {
        "name": "Kavita Trivedi",
        "title": "Junior BDE & Inside Sales Specialist",
        "experience_years": 2.0,
        "seniority": "Junior / Entry-Level",
        "email": "kavita.trivedi@salesboost.in",
        "phone": "+91 98444 55566",
        "location": "Surat",
        "education": "BBA (Bachelor of Business Administration)",
        "skills": [
            "BDE", "Sales", "Inside Sales", "Lead Generation", 
            "Cold Emailing", "Client Onboarding", "CRM", "Communication"
        ],
        "project_highlights": (
            "Generated 60+ qualified sales discovery calls per month for a SaaS product. "
            "Improved demo-to-trial conversion rate by 18% through personalized follow-up workflows."
        ),
        "bio": "Energetic Junior BDE with 2 years of experience in lead prospecting, outbound sales campaigns, and customer discovery.",
        "status": "Available / 2 weeks notice"
    },
    {
        "name": "Amit Rathi",
        "title": "Digital Marketing & Growth Lead",
        "experience_years": 3.5,
        "seniority": "Mid-level",
        "email": "amit.rathi@growthpulse.com",
        "phone": "+91 98555 66677",
        "location": "Surat / Pune",
        "education": "B.Com & Certified Digital Marketer",
        "skills": [
            "Marketing", "Digital Marketing", "SEO", "Google Ads", 
            "Meta Ads", "Growth Hacking", "Content Strategy", "Analytics"
        ],
        "project_highlights": (
            "Scaled inbound organic lead pipeline by 240% in 6 months using technical SEO and content clusters. "
            "Managed $45k/month performance ad budgets with an average 4.5x ROAS."
        ),
        "bio": "Results-oriented Growth & Digital Marketing specialist with 3.5 years experience in organic SEO, paid ads, and B2B customer acquisition.",
        "status": "Available / Immediate Joiner"
    }
]

def seed_default_candidates_if_empty():
    """Seeds the database with sample candidate CVs if the collection is empty."""
    try:
        count = candidates_collection.count_documents({})
        if count == 0:
            candidates_collection.insert_many(SAMPLE_CANDIDATES)
            print(f"Successfully seeded {len(SAMPLE_CANDIDATES)} candidate CVs into MongoDB.")
    except Exception as e:
        print(f"Error seeding candidate CVs: {e}")

def get_all_candidates():
    """Retrieves all candidate profiles from MongoDB."""
    try:
        candidates = list(candidates_collection.find({}, {"_id": 0}))
        if not candidates:
            # Fallback to in-memory sample if database is unreachable
            return SAMPLE_CANDIDATES
        return candidates
    except Exception as e:
        print(f"Error fetching candidates from DB: {e}")
        return SAMPLE_CANDIDATES

def insert_candidate(candidate_data):
    """Inserts a new candidate CV into MongoDB."""
    try:
        candidates_collection.insert_one(candidate_data)
        return True
    except Exception as e:
        print(f"Error inserting candidate: {e}")
        return False

def get_candidate_by_name(name):
    """Fetches candidate profile by name."""
    try:
        cand = candidates_collection.find_one(
            {"name": {"$regex": f"^{name}$", "$options": "i"}},
            {"_id": 0}
        )
        return cand
    except Exception:
        return None