"""InsightForge Streamlit UI: dashboard, AI assistant chat, evaluation and monitoring.

Run:  uv run streamlit run app.py
"""

from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from insightforge import charts
from insightforge.analytics import (
    calculate_kpis,
    dimension_performance,
    key_insights,
    load_sales,
    period_performance,
    prepare_sales,
    statistical_summary,
)
from insightforge.assistant import InsightForgeAssistant
from insightforge.evaluation import accuracy, build_eval_set, run_evaluation
from insightforge.llm import get_chat_model, load_settings
from insightforge.monitoring import InteractionLogger, summarize_interactions
from insightforge.prompts import EXECUTIVE_SUMMARY_QUESTION

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "sales.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "interactions.jsonl"

st.set_page_config(page_title="InsightForge", page_icon="📊", layout="wide")


@st.cache_data
def load_data(source: str | bytes) -> pd.DataFrame:
    if isinstance(source, bytes):
        from io import BytesIO

        return prepare_sales(load_sales(BytesIO(source)))
    return prepare_sales(load_sales(source))


def get_assistant(sales: pd.DataFrame, data_key: str) -> InsightForgeAssistant:
    if st.session_state.get("assistant_key") != data_key:
        settings = load_settings()
        st.session_state.assistant = InsightForgeAssistant(
            sales,
            llm=get_chat_model(settings),
            logger=InteractionLogger(LOG_FILE),
            model_label=settings.label,
        )
        st.session_state.assistant_key = data_key
        st.session_state.messages = []
    return st.session_state.assistant


# ----- sidebar ----------------------------------------------------------------
st.session_state.setdefault("session_id", uuid4().hex[:8])
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.title("📊 InsightForge")
    st.caption("AI-powered business intelligence assistant")
    try:
        settings = load_settings()
        settings_error = None
    except ValueError as error:
        settings, settings_error = None, str(error)
    if settings_error:
        st.error(settings_error)
    elif settings.is_configured:
        st.success(f"LLM: {settings.label}")
    else:
        st.warning(
            "Offline mode: no LLM configured. Answers show retrieved statistics only. "
            "Set LLM_PROVIDER and an API key in .env to enable AI answers."
        )

    upload = st.file_uploader("Use a different sales CSV", type="csv")
    if st.button("Clear conversation memory"):
        if "assistant" in st.session_state:
            st.session_state.assistant.clear_memory(st.session_state.session_id)
        st.session_state.messages = []

try:
    if upload is not None:
        raw = upload.getvalue()
        sales = load_data(raw)
        data_key = f"upload:{upload.name}:{len(raw)}"
    else:
        sales = load_data(str(DEFAULT_DATA))
        data_key = "default"
except ValueError as error:
    st.error(f"Could not load data: {error}")
    st.stop()

if settings_error:
    st.stop()
assistant = get_assistant(sales, data_key)

dashboard_tab, chat_tab, eval_tab, monitor_tab = st.tabs(
    ["Dashboard", "Ask InsightForge", "Evaluation", "Monitoring"]
)

# ----- dashboard --------------------------------------------------------------
with dashboard_tab:
    kpis = calculate_kpis(sales)
    cols = st.columns(5)
    cols[0].metric("Total revenue", f"₹{kpis['total_revenue']:,.0f}")
    cols[1].metric("Orders", kpis["total_orders"])
    cols[2].metric("Units sold", kpis["total_units"])
    cols[3].metric("Avg order value", f"₹{kpis['average_order_value']:,.0f}")
    cols[4].metric("Median order value", f"₹{kpis['median_order_value']:,.0f}")

    st.subheader("Key insights")
    for insight in key_insights(sales):
        st.markdown(f"- {insight}")

    if st.button("Generate AI executive summary", disabled=assistant.llm is None):
        with st.spinner("Analysing..."):
            summary = assistant.ask(EXECUTIVE_SUMMARY_QUESTION, session_id="summary")
            assistant.clear_memory("summary")
        st.markdown(summary.answer)

    st.subheader("Sales trends over time")
    left, right = st.columns([2, 1])
    left.plotly_chart(charts.sales_trend_figure(sales), width="stretch")
    right.dataframe(period_performance(sales, "quarter"), hide_index=True)

    st.subheader("Product and regional analysis")
    left, right = st.columns(2)
    left.plotly_chart(charts.product_performance_figure(sales), width="stretch")
    right.plotly_chart(charts.regional_figure(sales), width="stretch")
    st.plotly_chart(charts.region_product_heatmap(sales), width="stretch")

    st.subheader("Customer demographics and segmentation")
    left, mid, right = st.columns(3)
    left.plotly_chart(charts.segment_figure(sales), width="stretch")
    mid.plotly_chart(charts.demographics_figure(sales), width="stretch")
    right.plotly_chart(charts.age_distribution_figure(sales), width="stretch")

    with st.expander("Statistical measures"):
        st.dataframe(statistical_summary(sales))
    with st.expander("Performance tables"):
        for dimension in ("product", "region", "customer_segment", "age_band"):
            st.markdown(f"**{dimension.replace('_', ' ').title()}**")
            st.dataframe(dimension_performance(sales, dimension), hide_index=True)
    with st.expander("Raw data"):
        st.dataframe(sales, hide_index=True)

# ----- chat -------------------------------------------------------------------
with chat_tab:
    st.caption(
        "Ask about trends, products, regions, segments, demographics or statistics. "
        "Follow-up questions use conversation memory."
    )
    examples = [
        "Which region performs best?",
        "What about the lowest one?",
        "What is the median order value and standard deviation?",
        "How should we grow Tablet Air sales?",
    ]
    example_cols = st.columns(len(examples))
    clicked = None
    for col, example in zip(example_cols, examples, strict=True):
        if col.button(example, width="stretch"):
            clicked = example

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved statistics"):
                    st.caption(f"Standalone question: {message['standalone']}")
                    for source in message["sources"]:
                        st.code(source, language=None)

    prompt = st.chat_input("Ask a business question") or clicked
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving statistics and generating answer..."):
                try:
                    response = assistant.ask(prompt, session_id=st.session_state.session_id)
                except Exception as error:  # noqa: BLE001 - surface provider errors in the UI
                    st.error(f"LLM call failed: {error}")
                    st.stop()
            st.markdown(response.answer)
            sources = [f"[{d.metadata['topic']}]\n{d.page_content}" for d in response.sources]
            with st.expander("Retrieved statistics"):
                st.caption(f"Standalone question: {response.standalone_question}")
                for source in sources:
                    st.code(source, language=None)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": sources,
                "standalone": response.standalone_question,
            }
        )

# ----- evaluation -------------------------------------------------------------
with eval_tab:
    st.write(
        "Ground-truth answers are computed with pandas. **QAEvalChain** (LLM grader) and a "
        "deterministic fact check compare them with the assistant's answers."
    )
    eval_set = build_eval_set(sales)
    st.dataframe(
        pd.DataFrame([{"question": e.query, "expected": e.answer} for e in eval_set]),
        hide_index=True,
    )
    if st.button("Run evaluation"):
        with st.spinner(f"Evaluating {len(eval_set)} questions..."):
            try:
                st.session_state.eval_results = run_evaluation(
                    assistant, eval_set, grader=assistant.llm
                )
            except Exception as error:  # noqa: BLE001 - surface provider errors in the UI
                st.error(f"Evaluation failed: {error}")
    results = st.session_state.get("eval_results")
    if results is not None:
        qa_acc = accuracy(results, "qa_eval_grade")
        fact_acc = accuracy(results, "fact_check_grade")
        c1, c2, c3 = st.columns(3)
        c1.metric("QAEvalChain accuracy", "n/a (offline)" if qa_acc is None else f"{qa_acc:.0%}")
        c2.metric("Fact-check accuracy", f"{fact_acc:.0%}")
        c3.metric("Avg latency (s)", f"{results['latency_seconds'].mean():.2f}")
        st.dataframe(results, hide_index=True)

# ----- monitoring -------------------------------------------------------------
with monitor_tab:
    log = InteractionLogger(LOG_FILE).load()
    stats = summarize_interactions(log)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interactions", stats["interactions"])
    c2.metric("Sessions", stats["sessions"])
    c3.metric("Avg latency (s)", f"{stats['avg_latency']:.2f}")
    c4.metric("p95 latency (s)", f"{stats['p95_latency']:.2f}")
    if not log.empty:
        topics = log.explode("source_topics")["source_topics"].value_counts().head(10)
        st.bar_chart(topics, horizontal=True)
        st.dataframe(log.sort_values("timestamp", ascending=False), hide_index=True)
    else:
        st.info("No interactions logged yet. Ask a question in the assistant tab.")
