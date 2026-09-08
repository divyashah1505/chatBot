"""
Custom AI Conversational Engine (Mini ChatGPT) for Company Owner & Academic Assistant.
Runs 100% locally with zero external API dependencies.
All responses are formatted in clean, structured list-wise bullet points.
"""

import re
import random
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_base import (
    COLLEGE_INFO,
    PROGRAMS,
    ADMISSION_GUIDE,
    SCHOLARSHIPS_INFO
)
from .candidate_matcher import candidate_matcher


# Training Corpus for Semantic Intent Matching
INTENT_CORPUS = [
    # GREETING
    ("greeting", "hello"),
    ("greeting", "hi"),
    ("greeting", "hey"),
    ("greeting", "good morning"),
    ("greeting", "good afternoon"),
    ("greeting", "good evening"),
    ("greeting", "hey there"),
    ("greeting", "hello assistant"),
    ("greeting", "namaste"),
    ("greeting", "hi how are you"),
    
    # IDENTITY & CAPABILITIES
    ("identity", "who are you"),
    ("identity", "what is your name"),
    ("identity", "tell me about yourself"),
    ("identity", "what can you do"),
    ("identity", "how can you help me"),
    ("identity", "what are your capabilities"),
    ("identity", "are you an ai"),
    ("identity", "who created you"),
    
    # COURSES LIST
    ("courses_list", "what courses do you offer"),
    ("courses_list", "list of courses"),
    ("courses_list", "available programs"),
    ("courses_list", "which degrees are available"),
    ("courses_list", "what can i study here"),
    ("courses_list", "tell me all the branches and courses in the college"),
    ("courses_list", "ug and pg programs"),

    # ADMISSION GENERAL
    ("admission_general", "how to take admission"),
    ("admission_general", "admission procedure"),
    ("admission_general", "how can i apply for admission"),
    ("admission_general", "application process"),
    ("admission_general", "what is the admission criteria"),
    ("admission_general", "how do i register for college"),
    ("admission_general", "documents required for admission"),
    ("admission_general", "admission dates and deadlines"),

    # SCHOLARSHIPS
    ("scholarships", "are there any scholarships available"),
    ("scholarships", "scholarship criteria and discounts"),
    ("scholarships", "financial aid and concessions"),
    ("scholarships", "mysy digital gujarat scholarship details"),
    ("scholarships", "fee reduction for meritorious students"),

    # CAMPUS & FACILITIES
    ("facilities", "what facilities are available on campus"),
    ("facilities", "tell me about the library and labs"),
    ("facilities", "is there a hostel facility"),
    ("facilities", "sports, cafeteria and amenities"),
    ("facilities", "campus infrastructure and wifi"),

    # LOCATION
    ("location", "where is the college located"),
    ("location", "what is the college address"),
    ("location", "location of the campus in surat"),
    ("location", "how to reach the college"),
    ("location", "which city is the college in"),

    # CONTACT
    ("contact", "how can i contact the college"),
    ("contact", "give me the phone number and email"),
    ("contact", "contact details of admission office"),
    ("contact", "office hours and helpline number"),
    ("contact", "college email id and website"),

    # PLACEMENT GENERAL
    ("placement_general", "how are the placements in this college"),
    ("placement_general", "average salary package and campus drives"),
    ("placement_general", "which companies visit for placement"),
    ("placement_general", "highest package and placement cell"),

    # GRATITUDE
    ("gratitude", "thank you"),
    ("gratitude", "thanks a lot"),
    ("gratitude", "thank you so much for the help"),
    ("gratitude", "great, thanks"),
    ("gratitude", "appreciate it"),

    # GOODBYE
    ("goodbye", "bye"),
    ("goodbye", "goodbye"),
    ("goodbye", "see you later"),
    ("goodbye", "have a nice day"),
    ("goodbye", "good night"),

    # SMALL TALK
    ("smalltalk_howareyou", "how are you"),
    ("smalltalk_howareyou", "how are you doing"),
    ("smalltalk_howareyou", "how is it going"),
]


class CollegeAIEngine:
    """
    Intelligent Conversational AI Engine.
    Handles semantic vector similarity, candidate database matching, entity extraction,
    context tracking, and response generation in structured list format.
    """
    def __init__(self):
        self.intent_labels = [label for label, _ in INTENT_CORPUS]
        self.intent_texts = [text for _, text in INTENT_CORPUS]
        
        # Build TF-IDF Vectorizer with character and word n-grams
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            lowercase=True,
            stop_words='english'
        )
        self.intent_vectors = self.vectorizer.fit_transform(self.intent_texts)
        
        self.programs = PROGRAMS
        self.college_info = COLLEGE_INFO
        
        self.program_aliases = {
            "mca": ["mca", "master of computer application", "master of computer applications", "masters in computer", "masters in cs", "m.c.a", "postgraduate computer"],
            "bca": ["bca", "bachelor of computer application", "bachelor of computer applications", "bachelors in computer", "b.c.a", "undergraduate computer"],
            "mba": ["mba", "master of business administration", "masters in business", "m.b.a", "postgraduate management"],
            "bba": ["bba", "bachelor of business administration", "bachelors in business", "b.b.a", "undergraduate management"]
        }
        
        self.aspect_patterns = {
            "overview": [
                r"tell me about", r"what is", r"overview", r"details", r"information", r"about", r"explain", r"describe", r"course structure"
            ],
            "eligibility": [
                r"eligib", r"who can apply", r"criteria", r"requirement", r"qualification", r"percentage", r"marks", r"minimum", r"prerequisite", r"12th", r"graduation"
            ],
            "fees": [
                r"fee", r"fees", r"cost", r"how much", r"tuition", r"price", r"charge", r"installment", r"structure", r"payment"
            ],
            "syllabus": [
                r"syllabus", r"subject", r"subjects", r"curriculum", r"course work", r"learn", r"topics", r"semesters", r"modules"
            ],
            "careers": [
                r"career", r"job", r"jobs", r"scope", r"opportunities", r"roles", r"future", r"become", r"work as"
            ],
            "placements": [
                r"placement", r"placements", r"package", r"packages", r"salary", r"recruiter", r"companies", r"highest", r"average"
            ],
            "duration": [
                r"how long", r"duration", r"years", r"semesters", r"how many years", r"time period"
            ],
            "admission": [
                r"admission", r"apply", r"registration", r"enroll", r"how to join", r"entry"
            ]
        }

    def _preprocess(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\.\,\?\-]", " ", text)
        return text

    def extract_program_entity(self, message: str) -> Optional[str]:
        clean_msg = message.lower()
        for prog_key, aliases in self.program_aliases.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, clean_msg):
                    return prog_key
        return None

    def extract_aspect(self, message: str) -> Optional[str]:
        clean_msg = message.lower()
        aspect_priority = ["fees", "eligibility", "syllabus", "placements", "careers", "duration", "admission", "overview"]
        
        for aspect in aspect_priority:
            patterns = self.aspect_patterns.get(aspect, [])
            for pat in patterns:
                if re.search(r"\b" + pat, clean_msg):
                    return aspect
        return None

    def find_context_from_history(self, history: Optional[List[Dict]]) -> Optional[str]:
        if not history:
            return None
            
        for msg in reversed(history):
            text = msg.get("text", "")
            prog = self.extract_program_entity(text)
            if prog:
                return prog
        return None

    def compute_semantic_intent(self, message: str) -> Tuple[Optional[str], float]:
        try:
            query_vec = self.vectorizer.transform([message])
            similarities = cosine_similarity(query_vec, self.intent_vectors).flatten()
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score > 0.35:
                return self.intent_labels[best_idx], float(best_score)
            return None, float(best_score)
        except Exception:
            return None, 0.0

    def generate_program_response(self, prog_key: str, aspect: Optional[str] = None) -> str:
        """Generates dynamic, rich natural language explanations in list format."""
        prog = self.programs.get(prog_key)
        if not prog:
            return "I couldn't find details for that program. We offer MCA, BCA, MBA, and BBA courses."
            
        full_name = prog["full_name"]
        
        if aspect == "eligibility":
            return (
                f"📋 Eligibility Criteria for {full_name}:\n\n"
                f"• Qualification: {prog['eligibility']}\n"
                f"• Program Duration: {prog['duration']}\n"
                f"• Course Level: {prog['level']}\n\n"
                f"💡 Next Inquiries:\n"
                f"  • Ask for fee structure\n"
                f"  • Ask for syllabus & subjects"
            )
            
        elif aspect == "fees":
            fee_info = prog["fees"]
            return (
                f" Fee Structure for {full_name}:\n\n"
                f"• Semester Tuition: {fee_info['per_semester']}\n"
                f"• Estimated Annual Fee: {fee_info['annual']}\n"
                f"• Payment Policy: {fee_info['details']}\n"
                f"• Scholarship Support: Government (MYSY, Digital Gujarat) & Merit discounts available."
            )
            
        elif aspect == "syllabus":
            syllabus_lines = "\n".join([f"  • {line}" for line in prog["syllabus"]])
            return (
                f"📚 Curriculum & Syllabus for {full_name}:\n\n"
                f"{syllabus_lines}\n\n"
                f"• Note: Curriculum is aligned with modern tech & management industry standards."
            )
            
        elif aspect == "careers" or aspect == "placements":
            careers_lines = "\n".join([f"  • {c}" for c in prog["careers"]])
            return (
                f" Career Opportunities & Placements for {full_name}:\n\n"
                f"1. Potential Career Roles:\n"
                f"{careers_lines}\n\n"
                f"2. Placement Track Record:\n"
                f"  • Package: {prog['placements']}\n"
                f"  • Campus Support: Mock interviews, resume workshops, and direct company drives."
            )
            
        elif aspect == "duration":
            return (
                f" Program Duration for {full_name}:\n\n"
                f"• Total Duration: {prog['duration']}\n"
                f"• Academic Level: {prog['level']}"
            )
            
        elif aspect == "admission":
            return (
                f"📝 Admission Procedure for {full_name}:\n\n"
                f"1. Check Eligibility:\n"
                f"  • {prog['eligibility']}\n\n"
                f"2. Application Submission:\n"
                f"  • Online portal or visit our Surat campus admission office.\n\n"
                f"3. Document Verification & Merit List\n\n"
                f"4. Fee Confirmation:\n"
                f"  • Pay semester fee ({prog['fees']['per_semester']}) to secure seat."
            )
            
        else:
            # Complete comprehensive overview in clean list format
            syllabus_brief = "\n".join([f"  • {line}" for line in prog["syllabus"][:2]])
            career_brief = ", ".join(prog["careers"][:4])
            
            return (
                f"🎓 {full_name} ({prog['level']}):\n\n"
                f"1. Program Overview:\n"
                f"  • {prog['overview']}\n\n"
                f"2. Key Program Highlights:\n"
                f"  • Duration: {prog['duration']}\n"
                f"  • Eligibility: {prog['eligibility']}\n"
                f"  • Fee Structure: {prog['fees']['per_semester']}\n"
                f"  • Career Scope: {career_brief}\n"
                f"  • Placement Record: {prog['placements']}\n\n"
                f"3. Core Modules Preview:\n"
                f"{syllabus_brief}\n\n"
                f" Quick Questions You Can Ask:\n"
                f"  • 'What are the fees?'\n"
                f"  • 'What is the full syllabus?'\n"
                f"  • 'How do I apply for admission?'"
            )

    def generate_response(self, message: str, history: Optional[List[Dict]] = None) -> str:
        """Main AI response pipeline."""
        clean_msg = self._preprocess(message)
        
        if not clean_msg:
            return "How can I assist you today? You can search candidate CVs for your company or inquire about courses and admissions."

        # Priority 1: Candidate CV matching for Company Owner
        if candidate_matcher.is_candidate_query(clean_msg) or candidate_matcher.is_single_candidate_drilldown(clean_msg):
            return candidate_matcher.evaluate_candidates(clean_msg)

        # Priority 2: Degree Program Entity
        program_entity = self.extract_program_entity(clean_msg)
        aspect = self.extract_aspect(clean_msg)
        
        # Context Memory check
        if not program_entity and aspect in ["fees", "eligibility", "syllabus", "careers", "placements", "duration", "overview"]:
            context_program = self.find_context_from_history(history)
            if context_program:
                program_entity = context_program

        if program_entity:
            return self.generate_program_response(program_entity, aspect)

        # Priority 3: Semantic Intent Matching
        intent, confidence = self.compute_semantic_intent(clean_msg)
        
        if intent == "greeting" or "my name is" in clean_msg or "i am" in clean_msg:
            # Check if user introduced their name
            name_match = re.search(r"\bmy name is\s+([A-Za-z]+)", clean_msg)
            user_name = name_match.group(1).title() if name_match else ""
            
            if user_name:
                return (
                    f"Hello **{user_name}**! Nice to meet you.\n\n"
                    "I am your AI Assistant. Here is what I can help you with:\n"
                    "• **Document Intelligence & Summarization**: Upload any document and I will summarize it or extract key points.\n"
                    "• **CV & Profile Assistance**: Help you draft resumes, cover letters, and career profiles.\n"
                    "• **General Q&A**: Ask me any question or explore technical topics.\n\n"
                    "How can I assist you today?"
                )

            return (
                "Hello! Welcome to your AI Assistant.\n\n"
                "Here is what I can help you with:\n"
                "1. Document Analysis & Summarization: Upload any document (PDF, Word, Excel, etc.) for instant insights.\n"
                "2. Candidate & CV Database Search: Find developers, engineers, and technical profiles.\n"
                "3. Academic Programs & General Advisory: Ask any question or explore degree programs.\n\n"
                "How can I assist you today?"
            )
            
        elif intent == "identity":
            return (
                "💼🤖 I am your Intelligent Personal & Company AI Assistant.\n\n"
                "My Core Capabilities:\n"
                "1. Candidate CV Identification & Ranking (Skills, Experience, Projects, Star Ratings)\n"
                "2. Full Resume & Candidate Profile Inspection\n"
                "3. Academic Course Advisory (MCA, BCA, MBA, BBA)\n"
                "4. Admission, Eligibility & Fee Consultation\n"
                "5. Campus Facilities & Contact Inquiries\n\n"
                "Try asking: 'I want AI ML best candidates' or 'Tell me about MCA degree'."
            )
            
        elif intent == "courses_list":
            return (
                "📋 Available Degree Programs:\n\n"
                "1. Master of Computer Applications (MCA)\n"
                "  • Level: Postgraduate (2 Years / 4 Semesters)\n"
                "  • Focus: Advanced Software Engineering, Cloud, AI & Data Systems\n\n"
                "2. Bachelor of Computer Applications (BCA)\n"
                "  • Level: Undergraduate (3 Years / 6 Semesters)\n"
                "  • Focus: Programming Fundamentals, Web Design & Databases\n\n"
                "3. Master of Business Administration (MBA)\n"
                "  • Level: Postgraduate (2 Years / 4 Semesters)\n"
                "  • Focus: Marketing, Finance, HR & Strategic Management\n\n"
                "4. Bachelor of Business Administration (BBA)\n"
                "  • Level: Undergraduate (3 Years / 6 Semesters)\n"
                "  • Focus: Business Administration & Entrepreneurship\n\n"
                "Which course would you like to explore?"
            )
            
        elif intent == "admission_general":
            steps_text = "\n".join([f"  {step}" for step in ADMISSION_GUIDE["steps"]])
            docs_text = "\n".join([f"  • {doc}" for doc in ADMISSION_GUIDE["documents"][:4]])
            return (
                f" College Admission Guide:\n\n"
                f"1. Admission Steps:\n"
                f"{steps_text}\n\n"
                f"2. Required Documents:\n"
                f"{docs_text}\n\n"
                f"3. Key Notes:\n"
                f"  • {ADMISSION_GUIDE['deadlines']}\n"
                f"  • Contact Admission Cell: {COLLEGE_INFO['phone']}"
            )
            
        elif intent == "scholarships":
            return (
                f"🎓 Scholarship Schemes Available:\n\n"
                f"1. Merit Scholarships (Up to 25% fee waiver for top rankers)\n"
                f"2. Government Schemes (MYSY, Digital Gujarat SC/ST/OBC/SEBC)\n"
                f"3. Sports & Sibling Concessions\n\n"
                f"• How to Apply: Submit academic transcripts and income proof to the administration desk."
            )
            
        elif intent == "facilities":
            fac_lines = "\n".join([f"  • {f}" for f in COLLEGE_INFO["facilities"]])
            return (
                f"🏫 Campus Facilities at {COLLEGE_INFO['name']}:\n\n"
                f"{fac_lines}\n\n"
                f"• Campus Location: Surat, Gujarat"
            )
            
        elif intent == "location":
            return (
                f" Campus Location:\n\n"
                f"• College: {COLLEGE_INFO['name']}\n"
                f"• Address: {COLLEGE_INFO['address']}\n"
                f"• City: {COLLEGE_INFO['location']}\n"
                f"• Transport: Well-connected via Surat City Bus & BRTS routes."
            )
            
        elif intent == "contact":
            return (
                f"📞 Contact Information:\n\n"
                f"• Address: {COLLEGE_INFO['address']}\n"
                f"• Phone: {COLLEGE_INFO['phone']}\n"
                f"• Email: {COLLEGE_INFO['email']}\n"
                f"• Website: {COLLEGE_INFO['website']}\n"
                f"• Office Hours: {COLLEGE_INFO['office_hours']}"
            )
            
        elif intent == "placement_general":
            return (
                "💼 Placement Statistics & Support:\n\n"
                "• Active Training & Placement Cell with 50+ recruiting partners\n"
                "• Average Salary Package: ₹3.5 LPA to ₹7.5 LPA\n"
                "• Highest Package: Up to ₹9.0 LPA\n"
                "• Regular pre-placement aptitude training and mock interviews"
            )
            
        elif intent == "gratitude":
            return (
                "😊 You're very welcome!\n\n"
                "Feel free to ask whenever you need:\n"
                "• More candidate evaluations\n"
                "• Detailed resume inspections\n"
                "• Information on courses and fees"
            )
            
        elif intent == "goodbye":
            return "👋 Goodbye! Have a wonderful and productive day ahead."
            
        elif intent == "smalltalk_howareyou":
            return "😊 I'm doing great, thank you! Ready to assist you with candidate identification or course inquiries."

        # Intelligent Fallback with List Options
        return (
            "🤖 Here are the main tasks I can help you with:\n\n"
            "1. 💼 Candidate & CV Identification:\n"
            "  • 'I want AI ML best candidates in their skills and experience wise'\n"
            "  • 'Find React developers with 3+ years experience'\n"
            "  • 'Show full resume of Aarav Mehta'\n\n"
            "2. 🎓 Academic Program Inquiries:\n"
            "  • 'Tell me about MCA degree'\n"
            "  • 'What are the fees for BCA?'\n"
            "  • 'How do I apply for admission?'\n\n"
            "Please type your query to get started!"
        )


ai_engine = CollegeAIEngine()

def generate_ai_response(message: str, history: Optional[List[Dict]] = None) -> str:
    """Entry point for generating AI response."""
    return ai_engine.generate_response(message, history)
