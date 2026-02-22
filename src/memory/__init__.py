"""
J.A.R.V.I.S. Memory System — Phase 2
======================================
Gives Jarvis persistent memory across sessions:
  - ConversationStore: Searchable history of all exchanges
  - UserProfile: Auto-extracted facts about the user
  - MemoryManager: Orchestrates search, save, and context injection
"""

from src.memory.memory_manager import MemoryManager

__all__ = ["MemoryManager"]