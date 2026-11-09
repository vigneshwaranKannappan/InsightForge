# InsightForge

AI-powered Business Intelligence assistant (Advanced Generative AI capstone).
pandas computes every number; a LangChain RAG pipeline retrieves those statistics and an
LLM explains them as insights and recommendations; Streamlit presents dashboards, chat,
evaluation and monitoring.

## Architecture

```
data/sales.csv
  -> analytics.py        trusted pandas metrics (KPIs, time periods, products, regions,
                         segments, age bands, median / std dev / quartiles)
  -> knowledge_base.py   metrics -> LangChain Documents ("fact sheets", one per topic/entity)
  -> retriever.py        StatisticsRetriever: custom BaseRetriever (keyword + entity scoring)
  -> assistant.py        prompt chain: condense follow-up (memory) -> retrieve -> answer
       prompts.py        system / condense / answer prompts (grounding rules, answer format)
       llm.py            OpenAI or Anthropic chat model from .env (blank = offline mode)
       monitoring.py     JSONL interaction log (latency, retrieved topics, sessions)
  -> evaluation.py       QAEvalChain grading vs pandas ground truth + deterministic fact check
  -> charts.py           Plotly: sales trend, products, regions, heatmap, segments, demographics
  -> app.py              Streamlit UI: Dashboard | Ask InsightForge | Evaluation | Monitoring
```

| Capstone requirement | Where |
|---|---|
| Data preparation / knowledge base | `analytics.py`, `knowledge_base.py` |
| Sales by time period, product & region, demographics, statistics | `analytics.py` |
| pandas processing + custom retriever | `retriever.py` |
| Prompt engineering + chained prompts | `prompts.py`, `assistant.py` |
| RAG system | `assistant.py` |
| Memory | `assistant.py` (windowed per-session buffer, follow-up condensing) |
| QAEvalChain evaluation | `evaluation.py`, Evaluation tab, `notebooks/05_evaluation.py` |
| Visualizations | `charts.py`, Dashboard tab |
| Streamlit UI + monitoring | `app.py` |

## Setup

```powershell
uv sync
copy .env.example .env    # then set LLM_PROVIDER and the matching API key
```

`.env` options: `LLM_PROVIDER=openai|anthropic`, `LLM_MODEL` (optional override),
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. With `LLM_PROVIDER` blank the app runs **offline**:
retrieval, dashboards and the fact-check evaluation work; answers show the retrieved
statistics instead of LLM prose, and QAEvalChain is skipped.

## Run

```powershell
uv run streamlit run app.py                    # the application
uv run python notebooks/04_rag_assistant.py    # knowledge base, retriever, chain, memory
uv run python notebooks/05_evaluation.py       # QAEvalChain evaluation -> reports/
uv run pytest
uv run ruff check .
```

Earlier lessons: `notebooks/01_sales_analysis.py`, `02_visualize_sales.py`,
`03_reusable_analytics.py`.

### If Windows Application Control blocks `.venv\Scripts\python.exe`

Run the uv-managed base interpreter with the venv's packages on the path:

```powershell
$env:PYTHONPATH = "$PWD\.venv\Lib\site-packages;$PWD\src"
$py = (uv python find --system 3.12)
& $py -m pytest
& $py -m streamlit run app.py
```
