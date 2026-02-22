"""
J.A.R.V.I.S. User Profile
============================
Automatically extracts and stores facts about the user from conversations.

When the user says "My name is Swapnil" or "I work at Google", Jarvis
detects these as personal facts and stores them in a ChromaDB collection.

Extraction uses pattern matching (lightweight, no extra model needed):
  - "My name is X" / "I'm X" / "Call me X"
  - "I live in X" / "I'm from X"
  - "I work at X" / "I'm a X" (job)
  - "I like X" / "I love X" / "I hate X"
  - "I'm X years old" / "My age is X"

Facts are:
  - Deduplicated: "My name is Swapnil" won't create a second name entry
  - Searchable: "What's the user's name?" finds the name fact
  - Persistent: Survives restarts
  - Injected into every NLU prompt so Jarvis always knows who you are
"""

import os
import re
import time

import chromadb

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("memory.profile")

# Patterns that indicate the user is sharing a personal fact.
# Each tuple: (compiled regex, fact category, extraction group index)
FACT_PATTERNS = [
    # Name
    (re.compile(r"my name is (.+?)(?:\.|,|$)", re.IGNORECASE), "name"),
    (re.compile(r"i'm (.+?)(?:\.|,|$)", re.IGNORECASE), "name"),
    (re.compile(r"call me (.+?)(?:\.|,|$)", re.IGNORECASE), "name"),

    # Location
    (re.compile(r"i live in (.+?)(?:\.|,|$)", re.IGNORECASE), "location"),
    (re.compile(r"i'm from (.+?)(?:\.|,|$)", re.IGNORECASE), "location"),
    (re.compile(r"i am from (.+?)(?:\.|,|$)", re.IGNORECASE), "location"),

    # Work / Job
    (re.compile(r"i work (?:at|for) (.+?)(?:\.|,|$)", re.IGNORECASE), "job"),
    (re.compile(r"i'm (?:a|an) (.+?)(?:\.|,|$)", re.IGNORECASE), "job"),
    (re.compile(r"i am (?:a|an) (.+?)(?:\.|,|$)", re.IGNORECASE), "job"),

    # Preferences
    (re.compile(r"i (?:really )?like (.+?)(?:\.|,|$)", re.IGNORECASE), "likes"),
    (re.compile(r"i (?:really )?love (.+?)(?:\.|,|$)", re.IGNORECASE), "likes"),
    (re.compile(r"i (?:really )?enjoy (.+?)(?:\.|,|$)", re.IGNORECASE), "likes"),
    (re.compile(r"my favorite (.+?)(?:\.|,|$)", re.IGNORECASE), "favorite"),
    (re.compile(r"i (?:really )?(?:hate|dislike) (.+?)(?:\.|,|$)", re.IGNORECASE), "dislikes"),

    # Age
    (re.compile(r"i'm (\d+) years old", re.IGNORECASE), "age"),
    (re.compile(r"i am (\d+) years old", re.IGNORECASE), "age"),
    (re.compile(r"my age is (\d+)", re.IGNORECASE), "age"),
]


class UserProfile:
    """
    Stores and retrieves personal facts about the user.
    """

    def __init__(self):
        config = load_config()
        mem_cfg = config["memory"]
        profile_cfg = mem_cfg["profile"]

        storage_dir = os.path.expanduser(mem_cfg["storage_dir"])
        os.makedirs(storage_dir, exist_ok=True)

        self.max_facts: int = profile_cfg["max_facts"]

        # Reuse the same ChromaDB client path as ConversationStore
        self.client = chromadb.PersistentClient(path=storage_dir)

        self.collection = self.client.get_or_create_collection(
            name=profile_cfg["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"UserProfile ready: {self.collection.count()} facts stored"
        )

    def extract_and_save(self, user_text: str) -> list[str]:
        """
        Scan user text for personal facts and save any found.

        Args:
            user_text: What the user just said.

        Returns:
            List of facts that were extracted and saved (for logging).
        """
        extracted = []

        for pattern, category in FACT_PATTERNS:
            match = pattern.search(user_text)
            if match:
                value = match.group(1).strip()

                # Skip very short or clearly wrong extractions
                if len(value) < 2 or len(value) > 100:
                    continue

                # Skip if value looks like a question or command
                if value.endswith("?") or value.startswith("can") or value.startswith("will"):
                    continue

                fact_text = f"{category}: {value}"

                # Check if we already have this category stored
                # If so, update it instead of adding a duplicate
                self._upsert_fact(category, value, fact_text)
                extracted.append(fact_text)

        if extracted:
            logger.info(f"📋 Profile updated: {extracted}")

        return extracted

    def _upsert_fact(self, category: str, value: str, fact_text: str):
        """
        Insert or update a fact. If a fact with the same category exists,
        replace it (e.g., if user changes their name).
        """
        fact_id = f"fact_{category}"

        # Try to delete existing fact in this category
        try:
            self.collection.delete(ids=[fact_id])
        except Exception:
            pass  # Didn't exist, that's fine

        self.collection.add(
            ids=[fact_id],
            documents=[fact_text],
            metadatas=[{
                "category": category,
                "value": value,
                "timestamp": time.time(),
            }],
        )

    def get_all_facts(self) -> list[dict]:
        """
        Get all stored user facts.

        Returns:
            List of dicts with keys: category, value, fact_text.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        facts = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                facts.append({
                    "category": metadata.get("category", "unknown"),
                    "value": metadata.get("value", ""),
                    "fact_text": doc,
                })

        return facts

    def get_facts_text(self) -> str:
        """
        Get all facts as a formatted string for prompt injection.

        Returns:
            String like:
              "Known facts about the user:
               - name: Swapnil
               - location: India
               - likes: AI and coding"
            Or empty string if no facts stored.
        """
        facts = self.get_all_facts()
        if not facts:
            return ""

        lines = ["Known facts about the user:"]
        for f in facts:
            lines.append(f"- {f['fact_text']}")

        return "\n".join(lines)

    def search_facts(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Search facts by semantic similarity.
        Useful for answering "What's my name?" type questions.
        """
        if self.collection.count() == 0:
            return []

        n = min(n_results, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        facts = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - (distance / 2.0)

                facts.append({
                    "category": metadata.get("category", ""),
                    "value": metadata.get("value", ""),
                    "fact_text": doc,
                    "relevance_score": round(similarity, 3),
                })

        return facts