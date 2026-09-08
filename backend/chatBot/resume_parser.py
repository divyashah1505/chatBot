"""
Intelligent CV / Resume Parser and Ingestion Module.
Extracts structured candidate data (Name, Email, Phone, Skills, Experience, Projects)
from uploaded PDF or text resume files and stores them in MongoDB.
"""

import re
import io
from typing import Dict, List, Optional
import pypdf
from db.models import insert_candidate
from chatBot.candidate_matcher import SKILL_TAXONOMY


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text content from PDF bytes."""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def parse_resume_text(raw_text: str, default_name: str = "Candidate") -> Dict:
    """
    Parses resume text and extracts candidate fields:
    - Email, Phone
    - Detected Skills
    - Estimated Experience Years
    - Title / Role
    - Project Highlights
    """
    clean_text = raw_text.strip()
    
    # 1. Extract Email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", clean_text)
    email = email_match.group(0) if email_match else "contact@candidate.io"

    # 2. Extract Phone Number
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+91[-.\s]?\d{10}|\b\d{10}\b", clean_text)
    phone = phone_match.group(0) if phone_match else "+91 90000 00000"

    # 3. Extract Name (Heuristic: first non-empty line or default)
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    name = lines[0] if lines and len(lines[0].split()) <= 4 and not "@" in lines[0] else default_name

    # 4. Detect Skills from Taxonomy
    detected_skills = []
    text_lower = clean_text.lower()
    
    for category, skill_list in SKILL_TAXONOMY.items():
        for skill in skill_list:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                formatted_skill = skill.title()
                if formatted_skill not in detected_skills:
                    detected_skills.append(formatted_skill)

    if not detected_skills:
        detected_skills = ["Software Development", "Problem Solving", "Git"]

    # 5. Extract Years of Experience
    exp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years?|yrs?|year|yr)\s*(?:of)?\s*(?:experience|exp)", text_lower)
    if exp_match:
        try:
            exp_years = float(exp_match.group(1))
        except ValueError:
            exp_years = 2.0
    else:
        # Heuristic fallback based on seniority keywords
        if "senior" in text_lower or "lead" in text_lower:
            exp_years = 5.0
        elif "junior" in text_lower or "intern" in text_lower:
            exp_years = 1.0
        else:
            exp_years = 3.0

    # 6. Seniority Label
    if exp_years >= 5.0:
        seniority = "Senior"
    elif exp_years >= 2.5:
        seniority = "Mid-level"
    else:
        seniority = "Junior"

    # 7. Infer Title
    if "ai" in text_lower or "machine learning" in text_lower or "deep learning" in text_lower:
        title = f"{seniority} AI / Machine Learning Engineer"
    elif "devops" in text_lower or "cloud" in text_lower or "aws" in text_lower:
        title = f"{seniority} DevOps & Cloud Engineer"
    elif "react" in text_lower and "node" in text_lower:
        title = f"{seniority} Full Stack Developer"
    elif "frontend" in text_lower or "react" in text_lower:
        title = f"{seniority} Frontend Developer"
    else:
        title = f"{seniority} Software Engineer"

    # 8. Education
    edu = "Bachelor's Degree in Computer Science / IT"
    if "m.tech" in text_lower or "master" in text_lower or "mca" in text_lower:
        edu = "Master's Degree (M.Tech / MCA / M.Sc)"
    elif "bca" in text_lower:
        edu = "Bachelor of Computer Applications (BCA)"
    elif "b.tech" in text_lower or "b.e." in text_lower:
        edu = "B.Tech / B.E. in Computer Science"

    candidate_doc = {
        "name": name,
        "title": title,
        "experience_years": exp_years,
        "seniority": seniority,
        "email": email,
        "phone": phone,
        "location": "Surat / Remote",
        "education": edu,
        "skills": detected_skills[:12],
        "project_highlights": clean_text[:400].replace("\n", " ") + "...",
        "bio": f"{title} with {exp_years} years of experience in {', '.join(detected_skills[:4])}.",
        "status": "Available / Newly Added"
    }

    return candidate_doc


def process_and_save_cv(file_bytes: bytes, filename: str) -> Dict:
    """Processes uploaded CV file (PDF or TXT) and inserts it into MongoDB."""
    default_name = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        raise ValueError("Could not extract any readable text from the uploaded CV file.")

    candidate_data = parse_resume_text(text, default_name=default_name)
    success = insert_candidate(candidate_data)
    
    if not success:
        raise RuntimeError("Failed to insert candidate CV document into MongoDB database.")

    return candidate_data
