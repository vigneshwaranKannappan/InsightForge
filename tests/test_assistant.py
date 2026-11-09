from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from insightforge.analytics import (
    dimension_performance,
    key_insights,
    load_sales,
    period_performance,
    prepare_sales,
    statistical_summary,
)
from insightforge.assistant import InsightForgeAssistant
from insightforge.evaluation import accuracy, build_eval_set, heuristic_grade, run_evaluation
from insightforge.knowledge_base import build_knowledge_base
from insightforge.monitoring import InteractionLogger, summarize_interactions
from insightforge.retriever import StatisticsRetriever

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "sales.csv"


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    return prepare_sales(load_sales(DATA_FILE))


def test_period_performance_growth(sales: pd.DataFrame) -> None:
    months = period_performance(sales, "month")
    assert months["month"].tolist() == ["2026-01", "2026-02", "2026-03", "2026-04"]
    assert pd.isna(months.loc[0, "growth_pct"])
    assert months["revenue"].sum() == sales["revenue"].sum()


def test_dimension_performance_shares_sum_to_100(sales: pd.DataFrame) -> None:
    regions = dimension_performance(sales, "region")
    assert regions["revenue_share_pct"].sum() == pytest.approx(100, abs=0.5)
    assert regions["revenue"].is_monotonic_decreasing


def test_age_bands_and_statistics(sales: pd.DataFrame) -> None:
    assert set(sales["age_band"]) <= {"18-24", "25-34", "35-44", "45-54", "55+"}
    stats = statistical_summary(sales)
    assert stats.loc["revenue", "median"] == sales["revenue"].median()
    assert len(key_insights(sales)) == 6


def _retrieve(sales: pd.DataFrame, query: str) -> list[str]:
    retriever = StatisticsRetriever(documents=build_knowledge_base(sales), k=3)
    return [d.metadata["topic"] for d in retriever.invoke(query)]


def test_retriever_finds_relevant_topics(sales: pd.DataFrame) -> None:
    assert _retrieve(sales, "What is the median and standard deviation?")[0] == "statistics"
    assert _retrieve(sales, "How did the North region perform?")[0] == "region:North"
    assert "month_trend" in _retrieve(sales, "Show the monthly sales trend")
    assert _retrieve(sales, "Tell me about Laptop Pro")[0] == "product:Laptop Pro"


def test_retriever_falls_back_to_overview(sales: pd.DataFrame) -> None:
    topics = _retrieve(sales, "zzz qqq")
    assert topics == ["overview", "insights"]


def test_assistant_uses_memory_to_condense_follow_ups(sales: pd.DataFrame) -> None:
    llm = FakeListChatModel(
        responses=[
            "**Answer:** South leads.",
            "What is the revenue of the North region?",
            "**Answer:** North revenue is ...",
        ]
    )
    assistant = InsightForgeAssistant(sales, llm=llm, model_label="fake")

    first = assistant.ask("Which region performs best?", session_id="s1")
    assert first.standalone_question == "Which region performs best?"
    assert first.answer.startswith("**Answer:**")

    second = assistant.ask("And North?", session_id="s1")
    assert second.standalone_question == "What is the revenue of the North region?"
    assert second.sources[0].metadata["topic"] == "region:North"
    assert len(assistant.get_history("s1")) == 4

    assistant.clear_memory("s1")
    assert assistant.get_history("s1") == []


def test_memory_window_keeps_recent_turns(sales: pd.DataFrame) -> None:
    assistant = InsightForgeAssistant(sales, llm=None, memory_turns=2)
    for question in ("total revenue?", "best region?", "best product?"):
        assistant.ask(question, session_id="w")
    history = assistant.get_history("w")
    assert len(history) == 4
    assert history[0].content == "best region?"


def test_offline_assistant_returns_retrieved_facts(sales: pd.DataFrame, tmp_path: Path) -> None:
    logger = InteractionLogger(tmp_path / "log.jsonl")
    assistant = InsightForgeAssistant(sales, llm=None, logger=logger)
    response = assistant.ask("What is the total revenue?")
    assert "Offline mode" in response.answer
    assert "Business overview" in response.answer

    log = logger.load()
    assert len(log) == 1
    assert summarize_interactions(log)["interactions"] == 1


def test_offline_evaluation_passes_fact_check(sales: pd.DataFrame) -> None:
    assistant = InsightForgeAssistant(sales, llm=None)
    examples = build_eval_set(sales)
    results = run_evaluation(assistant, examples)
    assert accuracy(results, "qa_eval_grade") is None
    assert accuracy(results, "fact_check_grade") == 1.0


def test_qa_eval_chain_grades_with_llm(sales: pd.DataFrame) -> None:
    examples = build_eval_set(sales)[:2]
    assistant = InsightForgeAssistant(sales, llm=None)
    grader = FakeListChatModel(responses=["GRADE: CORRECT", "GRADE: INCORRECT"])
    results = run_evaluation(assistant, examples, grader=grader)
    assert results["qa_eval_grade"].tolist() == ["CORRECT", "INCORRECT"]


def test_heuristic_grade(sales: pd.DataFrame) -> None:
    example = build_eval_set(sales)[0]
    assert heuristic_grade(example, "nothing useful") == "INCORRECT"
