"""Model evaluation with LangChain's QAEvalChain.

Ground-truth answers are computed with pandas, so the evaluation set always matches the
data. An LLM grader (QAEvalChain) compares each assistant answer with the ground truth.
Without an LLM, a transparent heuristic grader checks that the key facts appear.
"""

import re
from dataclasses import dataclass

import pandas as pd
from langchain_core.language_models import BaseChatModel

from insightforge.analytics import (
    calculate_kpis,
    dimension_performance,
    period_performance,
)
from insightforge.assistant import InsightForgeAssistant


@dataclass
class EvalExample:
    query: str
    answer: str
    key_facts: list[str]


def build_eval_set(sales: pd.DataFrame) -> list[EvalExample]:
    kpis = calculate_kpis(sales)
    months = period_performance(sales, "month")
    best_month = months.loc[months["revenue"].idxmax()]
    product = dimension_performance(sales, "product").iloc[0]
    region = dimension_performance(sales, "region")
    segment = dimension_performance(sales, "customer_segment").iloc[0]
    age = dimension_performance(sales, "age_band").iloc[0]

    def inr(value: float) -> str:
        return f"INR {value:,.0f}"

    return [
        EvalExample(
            "What is the total revenue?",
            f"Total revenue is {inr(kpis['total_revenue'])}.",
            [f"{kpis['total_revenue']:,.0f}"],
        ),
        EvalExample(
            "What is the median order value?",
            f"The median order value is {inr(kpis['median_order_value'])}.",
            [f"{kpis['median_order_value']:,.0f}"],
        ),
        EvalExample(
            "What is the standard deviation of order revenue?",
            f"The order revenue standard deviation is {inr(kpis['revenue_std_dev'])}.",
            [f"{kpis['revenue_std_dev']:,.0f}"],
        ),
        EvalExample(
            "Which month had the highest revenue?",
            f"{best_month['month']} had the highest revenue at {inr(best_month['revenue'])}.",
            [str(best_month["month"])],
        ),
        EvalExample(
            "Which product generates the most revenue?",
            f"{product['product']} generates the most revenue at {inr(product['revenue'])}.",
            [str(product["product"])],
        ),
        EvalExample(
            "Which region has the highest revenue and which has the lowest?",
            f"{region.iloc[0]['region']} is highest and {region.iloc[-1]['region']} is lowest.",
            [str(region.iloc[0]["region"]), str(region.iloc[-1]["region"])],
        ),
        EvalExample(
            "Which customer segment contributes the most revenue?",
            f"The {segment['customer_segment']} segment contributes the most revenue "
            f"({segment['revenue_share_pct']}%).",
            [str(segment["customer_segment"])],
        ),
        EvalExample(
            "Which customer age group generates the most revenue?",
            f"Customers aged {age['age_band']} generate the most revenue.",
            [str(age["age_band"])],
        ),
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def heuristic_grade(example: EvalExample, prediction: str) -> str:
    normalized = _normalize(prediction)
    found = all(_normalize(fact) in normalized for fact in example.key_facts)
    return "CORRECT" if found else "INCORRECT"


def _parse_grade(raw: str) -> str:
    text = raw.strip().upper()
    if "INCORRECT" in text:
        return "INCORRECT"
    return "CORRECT" if "CORRECT" in text else "UNKNOWN"


def run_evaluation(
    assistant: InsightForgeAssistant,
    examples: list[EvalExample],
    grader: BaseChatModel | None = None,
) -> pd.DataFrame:
    """Ask every evaluation question in a fresh session and grade the answers."""
    predictions = []
    for index, example in enumerate(examples):
        session = f"eval-{index}"
        assistant.clear_memory(session)
        response = assistant.ask(example.query, session_id=session)
        assistant.clear_memory(session)
        predictions.append(response)

    heuristic = [heuristic_grade(e, p.answer) for e, p in zip(examples, predictions, strict=True)]

    if grader is not None:
        from langchain_classic.evaluation.qa import QAEvalChain

        eval_chain = QAEvalChain.from_llm(grader)
        graded = eval_chain.evaluate(
            [{"query": e.query, "answer": e.answer} for e in examples],
            [{"result": p.answer} for p in predictions],
            question_key="query",
            answer_key="answer",
            prediction_key="result",
        )
        llm_grades = [_parse_grade(g.get("results") or g.get("text", "")) for g in graded]
    else:
        llm_grades = [None] * len(examples)

    return pd.DataFrame(
        {
            "question": [e.query for e in examples],
            "expected": [e.answer for e in examples],
            "predicted": [p.answer for p in predictions],
            "qa_eval_grade": llm_grades,
            "fact_check_grade": heuristic,
            "latency_seconds": [p.latency_seconds for p in predictions],
        }
    )


def accuracy(results: pd.DataFrame, column: str) -> float | None:
    graded = results[column].dropna()
    if graded.empty:
        return None
    return float((graded == "CORRECT").mean())
