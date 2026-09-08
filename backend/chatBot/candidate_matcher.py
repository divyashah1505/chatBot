"""
Candidate Identification & Multi-Dimensional Evaluation Engine for Company Owners.
Matches, filters, and ranks candidate CVs stored in MongoDB based on:
1. Technical Skills relevance & depth
2. Years of experience & seniority level
3. Real-world project achievements & production systems
4. Education and candidate suitability
All responses are formatted in clean, structured list-wise bullet points.
"""

import re
from typing import List, Dict, Optional, Tuple
from db.models import get_all_candidates, get_candidate_by_name

# Skill taxonomy & synonyms dictionary
SKILL_TAXONOMY = {
    "bde": [
        "bde", "business development", "business development executive", "bdm", "lead generation",
        "client acquisition", "cold calling", "b2b sales", "crm", "sales funnel", "pipeline management",
        "outreach", "negotiation", "prospecting", "client onboarding"
    ],
    "sales": [
        "sales", "inside sales", "corporate sales", "account executive", "sales executive",
        "enterprise sales", "closing deals", "revenue growth", "prospecting", "b2b", "b2c",
        "account manager", "sales manager", "key account management", "salesforce"
    ],
    "cvd": [
        "cvd", "computer vision developer", "computer vision", "opencv", "yolo", "yolov8", "yolov9",
        "image processing", "deep learning vision", "object detection", "image segmentation",
        "pytorch vision", "ocr", "facial recognition", "mediapipe", "video analytics"
    ],
    "ai_ml": [
        "ai", "ml", "ai/ml", "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "llm", "llms", "large language model", "generative ai", "genai", "pytorch",
        "tensorflow", "scikit-learn", "keras", "transformers", "langchain", "opencv", "yolo", "hugging face", "rag"
    ],
    "python": [
        "python", "django", "fastapi", "flask", "numpy", "pandas", "scipy", "scikit-learn", "pytorch"
    ],
    "full_stack": [
        "full stack", "fullstack", "mern", "mean", "frontend", "backend", "web developer", "react", "node", "next.js"
    ],
    "frontend": [
        "frontend", "front end", "react", "react.js", "next.js", "vue", "angular", "javascript", "typescript", "tailwind", "css", "html", "ui/ux", "figma"
    ],
    "backend": [
        "backend", "back end", "node.js", "express", "fastapi", "django", "flask", "java", "spring boot", "golang", "microservices", "rest api", "graphql"
    ],
    "devops": [
        "devops", "cloud", "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "ci/cd", "terraform", "jenkins", "linux", "gitops"
    ],
    "database": [
        "database", "mongodb", "postgresql", "mysql", "redis", "sql", "nosql", "vector database"
    ],
    "marketing": [
        "digital marketing", "marketing", "seo", "sem", "google ads", "meta ads", "social media", "growth hacking", "content strategy"
    ],
    "hr": [
        "hr", "human resources", "talent acquisition", "recruiter", "hiring", "talent sourcing", "recruitment"
    ]
}


class CandidateMatcher:
    """
    Intelligent CV search and candidate evaluation engine.
    """
    def __init__(self):
        pass

    def extract_experience_criteria(self, message: str) -> Optional[float]:
        """Extracts minimum years of experience from query (e.g., '3+ years', '5 yrs', 'senior')."""
        clean = message.lower()
        
        # Regex for 'X+ years', 'X years', 'X yr', 'X yrs'
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years?|yrs?|year|yr)", clean)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
                
        if "senior" in clean or "lead" in clean:
            return 4.0
        if "mid" in clean or "intermediate" in clean:
            return 2.5
        if "junior" in clean or "entry" in clean or "fresher" in clean or "freshers" in clean:
            return 0.0
            
        return None

    def extract_requested_skills(self, message: str) -> List[str]:
        """Identifies target skills/domains requested by the owner."""
        clean = message.lower()
        matched_categories = []
        
        # Check against skill taxonomy
        for category, terms in SKILL_TAXONOMY.items():
            for term in terms:
                pattern = r"\b" + re.escape(term) + r"\b"
                if re.search(pattern, clean):
                    matched_categories.append(category)
                    break
                    
        return matched_categories

    def is_candidate_query(self, message: str) -> bool:
        """Determines if the owner's message is asking for candidate/CV identification."""
        clean = message.lower().strip()

        # 1. Ignore self-introductions and personal questions
        self_intro_patterns = [
            r"\bmy name is\b", r"\bi am\b", r"\bi'm\b", r"\bi work as\b",
            r"\bwhat is\b", r"\bhow to\b", r"\bexplain\b", r"\bhelp me (?:make|create|write)\b"
        ]
        if any(re.search(pat, clean) for pat in self_intro_patterns):
            return False

        # 2. Check for explicit search / recruitment keywords
        explicit_cand_phrases = [
            "find candidate", "find candidates", "search candidate", "search candidates",
            "show candidate", "show candidates", "list candidate", "list candidates",
            "top candidate", "top candidates", "best candidate", "best candidates",
            "recommend candidate", "recommend candidates", "candidate with", "candidates with",
            "filter candidate", "filter candidates", "hire", "hiring", "applicant", "applicants",
            "who can we hire", "find developers", "search developers", "show developers",
            "find engineers", "search engineers", "show engineers", "find me", "shortlist"
        ]
        if any(phrase in clean for phrase in explicit_cand_phrases):
            return True

        # 3. Check for Search Verb + Role/Skill pattern (e.g. "show react developers", "find python developers")
        search_verbs = ["find", "search", "show", "list", "get", "need", "looking for", "suggest"]
        target_nouns = ["candidate", "candidates", "developer", "developers", "engineer", "engineers", "profiles", "resumes", "cvs"]
        
        has_verb = any(re.search(r"\b" + re.escape(v) + r"\b", clean) for v in search_verbs)
        has_noun = any(re.search(r"\b" + re.escape(n) + r"\b", clean) for n in target_nouns)

        if has_verb and has_noun:
            return True

        # 4. Criteria-based searches (e.g. "React candidates with 3+ years")
        if ("years" in clean or "yrs" in clean or "senior" in clean or "fresher" in clean) and has_noun:
            return True

        return False

    def is_single_candidate_drilldown(self, message: str) -> Optional[str]:
        """Detects if the user is asking for details about a specific candidate by name."""
        clean = message.lower().strip()

        # Ignore self-introductions
        if "my name is" in clean or "i am" in clean or "i'm" in clean:
            return None

        # Look for explicit drilldown intent (e.g., "show resume of Vikram", "details of Neha")
        drilldown_intent = any(k in clean for k in ["resume of", "cv of", "profile of", "details of", "about candidate", "show", "tell me about"])
        if not drilldown_intent:
            return None

        candidates = get_all_candidates()
        for cand in candidates:
            cand_name = cand["name"].lower()
            if re.search(r"\b" + re.escape(cand_name) + r"\b", clean):
                return cand["name"]

        return None

    def calculate_match_score(self, candidate: Dict, query: str, req_exp: Optional[float], req_categories: List[str]) -> Tuple[float, Dict]:
        """Calculates multi-dimensional score (0 - 100%)."""
        cand_skills = [s.lower() for s in candidate.get("skills", [])]
        cand_title = candidate.get("title", "").lower()
        cand_exp = candidate.get("experience_years", 0.0)
        cand_projects = candidate.get("project_highlights", "").lower()
        cand_bio = candidate.get("bio", "").lower()
        
        clean_q = query.lower()

        # 1. Skill Score (out of 100)
        skill_score = 0.0
        if req_categories:
            cat_scores = []
            for cat in req_categories:
                related_terms = SKILL_TAXONOMY.get(cat, [])
                matched_skills = sum(1 for term in related_terms if any(term == s or term in s for s in cand_skills))
                matched_text = sum(1 for term in related_terms if term in cand_projects or term in cand_bio or term in cand_title)
                
                cat_score = min(100.0, (matched_skills * 25.0) + (matched_text * 10.0))
                cat_scores.append(cat_score)
                
            skill_score = sum(cat_scores) / max(1, len(cat_scores))
        else:
            words = [w for w in re.findall(r"\w+", clean_q) if len(w) >= 2]
            matches = sum(1 for w in words if any(w in s for s in cand_skills))
            skill_score = min(100.0, matches * 35.0)

        # 2. Experience Score (out of 100)
        if req_exp is not None:
            if req_exp == 0.0:  # Freshers / Juniors
                if cand_exp <= 2.5:
                    exp_score = 100.0
                else:
                    exp_score = max(40.0, 100.0 - (cand_exp - 2.5) * 15)
            else:
                if cand_exp >= req_exp:
                    exp_score = min(100.0, 85.0 + ((cand_exp - req_exp) * 5.0))
                else:
                    exp_score = max(20.0, (cand_exp / req_exp) * 80.0)
        else:
            if "best" in clean_q or "top" in clean_q or "experienced" in clean_q or "wise" in clean_q:
                exp_score = min(100.0, 65.0 + (cand_exp * 6.0))
            else:
                exp_score = 80.0

        # 3. Title Match (out of 100)
        title_score = 0.0
        query_words = re.findall(r"\w+", clean_q)
        for qw in query_words:
            if len(qw) >= 2 and qw in cand_title:
                title_score += 35.0
        title_score = min(100.0, title_score)

        # 4. Project & Bio Relevance (out of 100)
        proj_score = 0.0
        for qw in query_words:
            if len(qw) >= 2 and (qw in cand_projects or qw in cand_bio):
                proj_score += 25.0
        proj_score = min(100.0, proj_score)

        final_score = (
            (0.45 * skill_score) +
            (0.30 * exp_score) +
            (0.15 * title_score) +
            (0.10 * proj_score)
        )
        
        breakdown = {
            "skill_score": round(skill_score, 1),
            "exp_score": round(exp_score, 1),
            "title_score": round(title_score, 1),
            "proj_score": round(proj_score, 1)
        }
        
        return round(final_score, 1), breakdown

    def get_star_rating(self, score: float) -> str:
        """Returns visual star rating string."""
        if score >= 90:
            return "⭐⭐⭐⭐⭐ Top Match"
        elif score >= 78:
            return "⭐⭐⭐⭐ Strong Fit"
        elif score >= 60:
            return "⭐⭐⭐ Good Candidate"
        else:
            return "⭐⭐ Potential Match"

    def format_candidate_summary(self, rank: int, candidate: Dict, score: float) -> str:
        """Generates executive summary block for a candidate in clean list format."""
        stars = self.get_star_rating(score)
        top_skills = ", ".join(candidate.get("skills", [])[:6])
        
        return (
            f"🔹 Candidate #{rank}: {candidate['name']} ({score}% Match • {stars})\n"
            f"  • Role: {candidate['title']}\n"
            f"  • Experience: {candidate['experience_years']} Years ({candidate.get('seniority', 'Mid')})\n"
            f"  • Top Skills: {top_skills}\n"
            f"  • Education: {candidate.get('education', 'N/A')}\n"
            f"  • Key Project: {candidate.get('project_highlights', 'N/A')}\n"
            f"  • Contact: {candidate.get('email', 'N/A')} | {candidate.get('phone', 'N/A')}\n"
            f"  • Availability: {candidate.get('status', 'Available')}"
        )

    def format_full_candidate_profile(self, candidate: Dict) -> str:
        """Generates complete detailed CV profile of a candidate in clean list format."""
        all_skills = "\n".join([f"  • {s}" for s in candidate.get("skills", [])])
        return (
            f"📄 Full Candidate Profile: {candidate['name']}\n\n"
            f"1. Professional Details:\n"
            f"  • Designation: {candidate['title']}\n"
            f"  • Total Experience: {candidate['experience_years']} Years\n"
            f"  • Seniority Level: {candidate.get('seniority', 'Mid-Level')}\n"
            f"  • Location: {candidate.get('location', 'N/A')}\n"
            f"  • Education: {candidate.get('education', 'N/A')}\n"
            f"  • Availability: {candidate.get('status', 'Available')}\n\n"
            f"2. Core Technical Skills:\n"
            f"{all_skills}\n\n"
            f"3. Key Projects & Production Achievements:\n"
            f"  • {candidate.get('project_highlights', 'N/A')}\n\n"
            f"4. Contact Details:\n"
            f"  • Email: {candidate.get('email', 'N/A')}\n"
            f"  • Phone: {candidate.get('phone', 'N/A')}\n\n"
            f"💡 Suggested Actions:\n"
            f"  • Ask to schedule an interview\n"
            f"  • Review another candidate resume"
        )

    def evaluate_candidates(self, query: str) -> str:
        """Main entry point for candidate evaluation and ranking in list format."""
        single_cand_name = self.is_single_candidate_drilldown(query)
        if single_cand_name:
            cand = get_candidate_by_name(single_cand_name)
            if cand:
                return self.format_full_candidate_profile(cand)

        candidates = get_all_candidates()
        if not candidates:
            return "No candidates found in the database. Please add candidate CVs to your database."

        req_exp = self.extract_experience_criteria(query)
        req_categories = self.extract_requested_skills(query)

        scored_list = []
        for cand in candidates:
            score, breakdown = self.calculate_match_score(cand, query, req_exp, req_categories)
            scored_list.append((cand, score, breakdown))

        scored_list.sort(key=lambda x: x[1], reverse=True)

        top_candidates = [c for c in scored_list if c[1] >= 40]
        if not top_candidates:
            top_candidates = scored_list[:3]
        else:
            top_candidates = top_candidates[:4]

        criteria_notes = []
        if req_categories:
            criteria_notes.append(f"Domain/Skills: {', '.join([c.upper() for c in req_categories])}")
        if req_exp is not None:
            criteria_notes.append(f"Experience: {req_exp}+ Years")

        criteria_str = f" ({' | '.join(criteria_notes)})" if criteria_notes else ""
        
        response_blocks = [
            f"📋 Top Ranked Candidates from Database{criteria_str}:\n"
        ]

        for i, (cand, score, _) in enumerate(top_candidates, 1):
            block = self.format_candidate_summary(i, cand, score)
            response_blocks.append(block)

        response_blocks.append(
            "💡 Available Commands:\n"
            "  • 'Show full resume of [Candidate Name]'\n"
            "  • 'Filter candidates with 5+ years experience'\n"
            "  • 'Find candidates with [Specific Skill]'"
        )

        return "\n\n".join(response_blocks)


candidate_matcher = CandidateMatcher()

def process_candidate_query(message: str) -> str:
    """Wrapper function to evaluate recruitment queries."""
    return candidate_matcher.evaluate_candidates(message)
