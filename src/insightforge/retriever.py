"""Custom LangChain retriever that selects the most relevant statistics documents.

The knowledge base is small and numeric, so a transparent keyword-scoring retriever is
more accurate (and free) than embedding similarity: questions such as "median order
value" or "how did the North region do" map directly to topics and entity names.
"""

import re

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "what",
    "which",
    "how",
    "did",
    "do",
    "does",
    "me",
    "show",
    "tell",
    "about",
    "our",
    "my",
    "we",
    "by",
    "with",
    "it",
    "this",
    "that",
    "be",
    "give",
    "can",
    "you",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


class StatisticsRetriever(BaseRetriever):
    """Rank fact documents by keyword, entity-name and content overlap with the query."""

    documents: list[Document]
    k: int = 4
    keyword_weight: float = 3.0
    content_weight: float = 1.0
    fallback_topics: tuple[str, ...] = ("overview", "insights")

    def score(self, query: str, document: Document) -> float:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return 0.0
        keywords = {k.lower() for k in document.metadata.get("keywords", [])}
        content_tokens = set(tokenize(document.page_content))
        topic = document.metadata.get("topic", "")
        score = self.keyword_weight * len(query_tokens & keywords)
        score += self.content_weight * len(query_tokens & content_tokens)
        # Entity documents (e.g. "region:North") win when the entity is named.
        if ":" in topic:
            entity_tokens = set(tokenize(topic.split(":", 1)[1]))
            if entity_tokens and entity_tokens <= query_tokens:
                score += 5.0
        return score

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scored = [(self.score(query, doc), i, doc) for i, doc in enumerate(self.documents)]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "score": s})
            for s, _, doc in scored[: self.k]
        ]
        if not results:
            results = [
                doc for doc in self.documents if doc.metadata.get("topic") in self.fallback_topics
            ]
        return results
