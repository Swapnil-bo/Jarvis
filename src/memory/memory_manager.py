"""
J.A.R.V.I.S. Memory Manager
==============================
The brain's memory system. Orchestrates:
  1. Saving exchanges after each conversation cycle
  2. Extracting user facts from what the user says
  3. Building a memory context block to inject into NLU prompts

Context injection flow:
  User says: "What did I ask you yesterday?"
  
  Memory Manager builds:
    ┌─────────────────────────────────────────────┐
    │ [MEMORY CONTEXT]                             │
    │                                              │
    │ User profile:                                │
    │ - name: Swapnil                              │
    │ - likes: AI and coding                       │
    │                                              │
    │ Relevant past exchanges:                     │
    │ [Turn 12] User: "How does RAG work?"         │
    │           Jarvis: "RAG stands for..."        │
    │ [Turn 15] User: "Can you explain embeddings?"│
    │           Jarvis: "Embeddings are..."        │
    └─────────────────────────────────────────────┘
  
  This block gets prepended to Phi-3's system prompt.
  Result: Jarvis answers with awareness of who you are and what you've discussed.
"""

from src.memory.conversation_store import ConversationStore
from src.memory.user_profile import UserProfile
from src.utils.config import load_config
from src.utils.logger import get_logger, log_memory

logger = get_logger("memory.manager")


class MemoryManager:
    """
    Orchestrates all memory operations for the voice pipeline.
    """

    def __init__(self):
        config = load_config()
        mem_cfg = config["memory"]
        self.ctx_cfg = mem_cfg["context_injection"]

        self.enabled: bool = mem_cfg.get("enabled", True)

        if not self.enabled:
            logger.info("Memory system: DISABLED")
            self.conversations = None
            self.profile = None
            return

        # Initialize stores (both share the same ChromaDB directory)
        self.conversations = ConversationStore()
        self.profile = UserProfile()

        log_memory(logger)
        logger.info("✅ Memory system online")

    def after_exchange(self, user_text: str, jarvis_response: str) -> None:
        """
        Called after each complete conversation cycle.
        Saves the exchange and extracts any user facts.

        Args:
            user_text: What the user said (transcription).
            jarvis_response: What Jarvis replied.
        """
        if not self.enabled:
            return

        # Save the full exchange for future retrieval
        self.conversations.save_exchange(user_text, jarvis_response)

        # Scan for personal facts ("My name is...", "I work at...", etc.)
        self.profile.extract_and_save(user_text)

    def build_context(self, user_text: str) -> str:
        """
        Build a memory context block to inject into the NLU prompt.

        Searches past conversations for relevant exchanges and
        includes the user's profile facts.

        Args:
            user_text: The current user query.

        Returns:
            Formatted string of memory context, or empty string if
            no relevant memories found.
        """
        if not self.enabled:
            return ""

        sections = []

        # --- User Profile ---
        if self.ctx_cfg.get("include_profile", True):
            profile_text = self.profile.get_facts_text()
            if profile_text:
                sections.append(profile_text)

        # --- Relevant Past Conversations ---
        if self.ctx_cfg.get("include_conversation", True):
            relevant = self.conversations.search(user_text)
            if relevant:
                lines = ["Relevant past exchanges:"]
                for ex in relevant:
                    turn = ex.get("turn_number", "?")
                    lines.append(
                        f"[Turn {turn}] User: \"{ex['user_text']}\"\n"
                        f"           Jarvis: \"{ex['jarvis_response']}\""
                    )
                sections.append("\n".join(lines))

        if not sections:
            return ""

        # Combine into a single context block
        context = "[MEMORY CONTEXT]\n" + "\n\n".join(sections)

        # Rough token estimate: ~4 chars per token
        max_chars = self.ctx_cfg.get("max_memory_tokens", 300) * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[...memory truncated]"

        logger.debug(f"  Memory context: {len(context)} chars injected")
        return context

    def get_stats(self) -> dict:
        """Get memory system stats for logging."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "total_exchanges": self.conversations.collection.count(),
            "total_facts": self.profile.collection.count(),
        }