"""The InsightForge RAG assistant: retriever + chained prompts + conversation memory."""

import time
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from insightforge.knowledge_base import build_knowledge_base
from insightforge.monitoring import InteractionLogger
from insightforge.prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT
from insightforge.retriever import StatisticsRetriever


@dataclass
class AssistantResponse:
    question: str
    standalone_question: str
    answer: str
    sources: list[Document] = field(default_factory=list)
    latency_seconds: float = 0.0


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('topic', 'fact')}]\n{doc.page_content}" for doc in documents
    )


class InsightForgeAssistant:
    """Answers business questions from retrieved statistics, remembering each session.

    With ``llm=None`` the assistant runs offline: it still condenses follow-ups
    heuristically and retrieves facts, then returns those facts verbatim.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        llm: BaseChatModel | None = None,
        k: int = 4,
        logger: InteractionLogger | None = None,
        model_label: str = "offline",
        memory_turns: int = 6,
    ) -> None:
        self.llm = llm
        self.model_label = model_label if llm is not None else "offline"
        self.documents = build_knowledge_base(sales)
        self.retriever = StatisticsRetriever(documents=self.documents, k=k)
        self.logger = logger
        self.memory_turns = memory_turns
        self._histories: dict[str, list[BaseMessage]] = {}
        self.chain = self._build_chain()

    # ----- memory (windowed conversation buffer per session) ------------------
    def get_history(self, session_id: str) -> list[BaseMessage]:
        return self._histories.setdefault(session_id, [])

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        history = self.get_history(session_id)
        history.extend([HumanMessage(question), AIMessage(answer)])
        del history[: max(0, len(history) - 2 * self.memory_turns)]

    def clear_memory(self, session_id: str = "default") -> None:
        self._histories.pop(session_id, None)

    # ----- chain --------------------------------------------------------------
    def _condense_step(self) -> Runnable:
        llm_condense = CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser() if self.llm else None

        def condense(inputs: dict) -> str:
            history = inputs.get("chat_history") or []
            question = inputs["question"]
            if not history:
                return question
            if llm_condense is not None:
                return llm_condense.invoke(inputs).strip()
            previous = [m.content for m in history if m.type == "human"]
            return f"{previous[-1]} {question}" if previous else question

        return RunnableLambda(condense)

    def _answer_step(self) -> Runnable:
        if self.llm is not None:
            return ANSWER_PROMPT | self.llm | StrOutputParser()

        def offline_answer(inputs: dict) -> str:
            return (
                "_Offline mode (no LLM configured) - the most relevant statistics are:_\n\n"
                + inputs["context"]
            )

        return RunnableLambda(offline_answer)

    def _build_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(standalone_question=self._condense_step())
            .assign(sources=(lambda x: x["standalone_question"]) | self.retriever)
            .assign(context=lambda x: format_documents(x["sources"]))
            .assign(answer=self._answer_step())
        )

    # ----- public API ---------------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> AssistantResponse:
        start = time.perf_counter()
        result = self.chain.invoke(
            {"question": question, "chat_history": list(self.get_history(session_id))}
        )
        self._remember(session_id, question, result["answer"])
        response = AssistantResponse(
            question=question,
            standalone_question=result["standalone_question"],
            answer=result["answer"],
            sources=result["sources"],
            latency_seconds=round(time.perf_counter() - start, 3),
        )
        if self.logger is not None:
            self.logger.log(response, session_id=session_id, model=self.model_label)
        return response
