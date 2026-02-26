"""
J.A.R.V.I.S. Tool: Web Search
================================
Searches the web using DuckDuckGo (free, no API key) and summarizes
results using Phi-3 for a TTS-friendly spoken answer.

Flow:
  1. User asks a question → Router classifies as web_search
  2. DuckDuckGo returns top 3 results with snippets
  3. Phi-3 reads snippets and generates a concise spoken answer
  4. Answer is spoken by TTS

Dependencies:
  pip install duckduckgo-search

RAM impact: ~0MB (uses existing Phi-3 via Ollama for summarization).
"""

import requests
from datetime import datetime

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("tools.web_search")

# Summarization prompt — tells Phi-3 to create a TTS-friendly answer
SUMMARIZE_PROMPT = """You are Jarvis, a voice assistant. Based on the search results below, give a concise spoken answer to the user's question.

CURRENT DATE: {current_date}

RULES:
- Answer in 1 to 3 short sentences maximum.
- Write exactly as you would speak aloud. No markdown, no bullet points, no special characters.
- DO NOT read out URLs, links, or website names.
- If the results don't answer the question, say so briefly.
- NEVER make up information not found in the results.
- STOP after answering. Do not generate instructions or continue writing.

SEARCH RESULTS:
{results}

USER QUESTION: {question}

YOUR SPOKEN ANSWER:"""


class WebSearchTool:
    """
    Web search using DuckDuckGo + Phi-3 summarization.
    """

    def __init__(self):
        config = load_config()
        nlu_cfg = config["nlu"]
        self.base_url: str = nlu_cfg["base_url"]
        self.model: str = nlu_cfg["model"]
        self._ddgs = None

    def _get_ddgs(self):
        """Lazy-load DuckDuckGo search client."""
        if self._ddgs is None:
            try:
                from ddgs import DDGS
                self._ddgs = DDGS()
            except ImportError:
                try:
                    from duckduckgo_search import DDGS
                    self._ddgs = DDGS()
                except ImportError:
                    logger.error("ddgs not installed! Run: pip install ddgs")
                    self._ddgs = None
        return self._ddgs

    def execute(self, action: str, params: dict = None) -> str:
        """
        Execute a web search action.

        Args:
            action: Usually 'search' or 'lookup'.
            params: Must contain 'query' key.

        Returns:
            Summarized answer string for TTS.
        """
        params = params or {}

        # Extract query from various param keys Phi-3 might use
        query = ""
        for key in ("query", "search", "question", "q", "text", "term", "topic"):
            if key in params:
                query = str(params[key]).strip()
                if query:
                    break

        if not query:
            return "What would you like me to search for?"

        logger.info(f"🔍 Searching: \"{query}\"")

        # Step 1: Search DuckDuckGo
        snippets = self._search_ddg(query)

        if not snippets:
            return f"I couldn't find any results for {query}. Check your internet connection."

        # Step 2: Summarize with Phi-3
        answer = self._summarize(query, snippets)

        return answer

    def _search_ddg(self, query: str) -> str:
        """
        Search DuckDuckGo and return formatted snippets.

        Returns:
            Formatted string of top 3 results, or empty string on failure.
        """
        ddgs = self._get_ddgs()

        if ddgs is None:
            # Fallback: try DuckDuckGo instant answer API (no pip needed)
            return self._search_ddg_api(query)

        try:
            results = list(ddgs.text(query, max_results=3))

            if not results:
                logger.warning("DuckDuckGo returned 0 results, trying fallback API")
                return self._search_ddg_api(query)

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body = r.get("body", r.get("text", ""))
                formatted.append(f"Result {i}: {title}. {body}")

            snippet_text = "\n".join(formatted)
            logger.info(f"🔍 Got {len(results)} results from DuckDuckGo")
            return snippet_text

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            # Try fallback API
            return self._search_ddg_api(query)

    def _search_ddg_api(self, query: str) -> str:
        """
        Fallback: DuckDuckGo instant answer API (no pip dependency needed).
        Less comprehensive but works without duckduckgo-search package.
        """
        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
                timeout=8,
            )
            data = response.json()

            snippets = []

            # Abstract (main answer)
            if data.get("AbstractText"):
                snippets.append(f"Answer: {data['AbstractText']}")

            # Related topics
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    snippets.append(f"Related: {topic['Text']}")

            return "\n".join(snippets) if snippets else ""

        except Exception as e:
            logger.warning(f"DuckDuckGo API fallback failed: {e}")
            return ""

    def _summarize(self, question: str, snippets: str) -> str:
        """
        Use Phi-3 to summarize search results into a spoken answer.
        """
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        prompt = SUMMARIZE_PROMPT.format(
            current_date=current_date, 
            results=snippets, 
            question=question
        )

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": "",
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "temperature": 0.3,
                        "num_predict": 100,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()

            raw = response.json().get("response", "").strip()

            # Clean up Phi-3 hallucination artifacts
            for poison in ["---", "###", "REFERENCE:", "Instruction:", "```", "SEARCH RESULTS:", "USER QUESTION:"]:
                if poison in raw:
                    raw = raw[:raw.index(poison)].strip()

            if not raw:
                # If summarization failed, return first snippet directly
                first_line = snippets.split("\n")[0]
                return first_line[:200] if len(first_line) > 200 else first_line

            return raw

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            # Return raw first snippet as fallback
            first_line = snippets.split("\n")[0]
            return first_line[:200] if len(first_line) > 200 else first_line