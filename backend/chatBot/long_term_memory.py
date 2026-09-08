"""
ChatGPT-Style Long-Term Memory & Cross-Session Knowledge Engine (100% Local & Offline).

Manages:
1. Cross-Session Memory Recall: Access past documents, conversations, and topics across new chats.
2. Frequent Query & Topic Analytics: Identifies what the user generally asks most often.
3. User Profile & Fact Memory: Persists user-stated facts and preferences.
4. 100% Local Storage: Uses MongoDB and in-memory local caches without any third-party cloud APIs.
"""

import re
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from db.models import (
    get_user_all_past_messages,
    save_user_memory,
    get_user_memories,
    clear_user_memories,
    get_all_user_documents,
    save_document_metadata
)
from .document_memory import doc_memory


class LongTermMemoryManager:
    """
    Central Long-Term Memory Manager for Cross-Session Intelligence.
    Provides ChatGPT-like persistent memory across all user chat sessions.
    """

    @staticmethod
    def extract_and_save_facts(user_id: str, message: str) -> List[str]:
        """Extracts user profile facts (e.g., name, role, skills) from conversational statements."""
        clean = message.strip()
        saved_facts = []

        # 1. Name detection ("My name is ...", "I am ...")
        name_m = re.search(r"(?:my name is|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", clean, re.IGNORECASE)
        if name_m:
            name_val = name_m.group(1).strip()
            if name_val.lower() not in ["asking", "looking", "trying", "wondering", "working", "searching", "a", "an"]:
                save_user_memory(user_id, "User Name", name_val, category="profile")
                saved_facts.append(f"Name: {name_val}")

        # 2. Role / Profession / Study ("I am a developer", "I am an MCA student")
        role_m = re.search(r"i am (?:a|an)\s+([A-Za-z0-9\s\-]{3,35})(?:\.|\,|$|\s+at|\s+in)", clean, re.IGNORECASE)
        if role_m:
            role_val = role_m.group(1).strip()
            if role_val.lower() not in ["asking", "here", "just", "user", "trying", "wondering"]:
                save_user_memory(user_id, "Role / Background", role_val.title(), category="profile")
                saved_facts.append(f"Role: {role_val}")

        return saved_facts

    @staticmethod
    def get_frequent_queries_and_topics(user_id: str) -> Dict[str, Any]:
        """
        Analyzes all past messages across all sessions to extract:
        - Most frequent exact and normalized questions.
        - Dominant topic themes.
        - Interaction statistics.
        """
        all_msgs = get_user_all_past_messages(user_id)
        user_queries = [m["text"].strip() for m in all_msgs if m.get("sender") == "user"]

        if not user_queries:
            # Fallback to active document memory prompts if DB is empty
            user_queries = []
            for entry in doc_memory._memory_store.values():
                user_queries.extend(entry.get_all_past_prompts())

        if not user_queries:
            return {
                "total_queries": 0,
                "top_questions": [],
                "top_topics": [],
                "frequent_keywords": []
            }

        # Normalize questions (ignore punctuation and casing)
        norm_queries = []
        topic_counter = Counter()
        keyword_counter = Counter()

        stop_words = {
            "what", "is", "the", "of", "in", "for", "to", "a", "an", "and", "or", "by", 
            "who", "where", "how", "which", "tell", "me", "about", "give", "show", "find",
            "please", "can", "you", "this", "my", "i", "do", "does", "with", "from"
        }

        for q in user_queries:
            # Clean text
            clean_q = re.sub(r"[^\w\s]", "", q.lower()).strip()
            if clean_q:
                norm_queries.append(q)

            # Categorize topic
            q_low = q.lower()
            if any(k in q_low for k in ["project", "certificate", "student", "cs8033", "clothiq", "ewsign", "guide", "team"]):
                topic_counter["Academic Projects & Certificates"] += 1
            elif any(k in q_low for k in ["candidate", "resume", "cv", "hire", "developer", "experience", "skill"]):
                topic_counter["Candidate & Talent Search"] += 1
            elif any(k in q_low for k in ["summarize", "summary", "overview", "tldr", "analyze", "explain"]):
                topic_counter["Document Summarization & Analysis"] += 1
            elif any(k in q_low for k in ["college", "mca", "bca", "admission", "fee", "syllabus"]):
                topic_counter["College & Admission Inquiries"] += 1
            else:
                topic_counter["General Document & AI Queries"] += 1

            # Keywords
            words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", q.lower()) if w not in stop_words]
            for w in words:
                keyword_counter[w] += 1

        query_counts = Counter(norm_queries)
        top_questions = [{"query": q, "count": c} for q, c in query_counts.most_common(5)]
        top_topics = [{"topic": t, "count": c} for t, c in topic_counter.most_common(4)]
        top_keywords = [w for w, c in keyword_counter.most_common(8)]

        return {
            "total_queries": len(user_queries),
            "top_questions": top_questions,
            "top_topics": top_topics,
            "frequent_keywords": top_keywords
        }

    @staticmethod
    def is_memory_meta_query(message: str) -> bool:
        """Detects if the user is asking about their past chats, frequent queries, or remembered facts."""
        clean = message.lower().strip()
        
        patterns = [
            "which question", "which questions", "what question", "what questions",
            "most asked", "generally ask", "often ask", "frequently ask",
            "what did i ask", "what did we discuss", "what did we talk",
            "what do you remember", "remember about me", "my previous chat",
            "my previous chats", "past chat", "past chats", "past questions",
            "what documents did i upload", "my uploaded documents", "my memory",
            "show my history", "what do you know about me", "summarize my chats"
        ]
        return any(p in clean for p in patterns)

    @classmethod
    def answer_memory_query(cls, user_id: str, message: str) -> str:
        """
        Generates a direct, accurate answer to user queries about their cross-session history,
        frequent questions, uploaded documents, and saved memory profile.
        """
        stats = cls.get_frequent_queries_and_topics(user_id)
        memories = get_user_memories(user_id)
        user_docs = get_all_user_documents(user_id)

        # Fallback doc check from in-memory store
        if not user_docs and doc_memory._memory_store:
            user_docs = [
                {"filename": e.filename, "upload_count": e.upload_count, "summary": e.get_latest_summary()}
                for e in doc_memory._memory_store.values()
            ]

        parts = [
            "###  Your Long-Term Memory & Conversation Insights",
            ">  *I have full memory of your previous chats, uploaded documents, and frequent topics across all sessions.*\n"
        ]

        # 1. Frequently Asked Questions
        if stats["top_questions"]:
            parts.append("####  Questions You Generally Ask Most Often:")
            for idx, item in enumerate(stats["top_questions"], 1):
                count_str = f"*(asked {item['count']} time{'s' if item['count'] > 1 else ''})*"
                parts.append(f"{idx}. **\"{item['query']}\"** {count_str}")
            parts.append("")
        else:
            parts.append("####  Questions You Generally Ask:")
            parts.append("• *You are starting your conversation history with me! As you ask more questions, I will track and summarize your most frequent topics.*\n")

        # 2. Main Discussion Topics
        if stats["top_topics"]:
            parts.append("####  Your Primary Topics & Areas of Interest:")
            for item in stats["top_topics"]:
                parts.append(f"• **{item['topic']}**: `{item['count']} query(ies)`")
            parts.append("")

        # 3. Uploaded Documents Remembered
        if user_docs:
            parts.append("####  Remembered Uploaded Documents (Cross-Session):")
            for d in user_docs[:5]:
                parts.append(f"• **`{d['filename']}`** *(Upload count: {d.get('upload_count', 1)})*")
            parts.append("")

        # 4. User Profile & Preferences
        if memories:
            parts.append("#### 👤 What I Remember About You:")
            for m in memories:
                parts.append(f"• **{m['key']}**: `{m['value']}`")
            parts.append("")

        # 5. Storage explanation note
        parts.extend([
            "---",
            " **Where is this data stored?**",
            "• **Location**: 100% locally on your machine in your local **MongoDB database** (`user_memory`, `sessions`, and `uploaded_documents` collections) + in-memory Vector store.",
            "• **Privacy**: No external 3rd-party cloud APIs are used. Everything is stored locally on your system."
        ])

        return "\n".join(parts)

    @classmethod
    def get_system_prompt_memory_context(cls, user_id: str) -> str:
        """
        Builds a compact memory context block to inject into the Local LLaMA system prompt,
        ensuring cross-session awareness in any new chat.
        """
        stats = cls.get_frequent_queries_and_topics(user_id)
        memories = get_user_memories(user_id)
        user_docs = get_all_user_documents(user_id)

        lines = ["[LONG-TERM USER MEMORY & CROSS-SESSION CONTEXT]"]

        # Add profile facts
        if memories:
            for m in memories:
                lines.append(f"- User {m['key']}: {m['value']}")

        # Add frequent topics
        if stats["top_topics"]:
            topic_names = ", ".join([t["topic"] for t in stats["top_topics"][:3]])
            lines.append(f"- User Primary Interests: {topic_names}")

        # Add top questions
        if stats["top_questions"]:
            top_q_list = "; ".join([f"\"{q['query']}\"" for q in stats["top_questions"][:3]])
            lines.append(f"- User Frequently Asked Questions: {top_q_list}")

        # Add remembered documents
        if user_docs:
            doc_names = ", ".join([d["filename"] for d in user_docs[:4]])
            lines.append(f"- Uploaded Documents in Memory: {doc_names}")

        lines.append(
            "[END USER MEMORY]\n"
            "Use the memory context above if the user asks about their history, previous questions, documents, or personal facts."
        )

        return "\n".join(lines)


# Global Singleton Long-Term Memory Instance
long_term_memory = LongTermMemoryManager()
