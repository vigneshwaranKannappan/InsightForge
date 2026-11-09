# %% [markdown]
# # Lesson 4: Knowledge base, custom retriever, prompt chain and memory
#
# Flow of one question:
#
# ```
# question + chat memory
#   -> condense prompt (LLM rewrites follow-ups as standalone questions)
#   -> StatisticsRetriever (picks the most relevant pandas fact sheets)
#   -> answer prompt (LLM explains ONLY the retrieved facts)
#   -> answer saved to memory
# ```
#
# Works offline (no API key): the retrieved statistics are returned verbatim.
# Configure `.env` (LLM_PROVIDER + API key) for natural-language answers.
#
# ```powershell
# uv run python notebooks/04_rag_assistant.py
# ```

# %%
from pathlib import Path

from insightforge.analytics import load_sales, prepare_sales
from insightforge.assistant import InsightForgeAssistant
from insightforge.knowledge_base import build_knowledge_base
from insightforge.llm import get_chat_model, load_settings
from insightforge.retriever import StatisticsRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sales = prepare_sales(load_sales(PROJECT_ROOT / "data" / "sales.csv"))

# %% [markdown]
# ## 1. Knowledge base: pandas statistics as documents

# %%
documents = build_knowledge_base(sales)
print(f"{len(documents)} fact documents")
for doc in documents[:5]:
    print(f"- {doc.metadata['topic']}")
print("\nExample document:\n" + documents[0].page_content)

# %% [markdown]
# ## 2. Custom retriever

# %%
retriever = StatisticsRetriever(documents=documents, k=3)
for query in ["median order value", "How did the West region do?", "monthly trend"]:
    topics = [d.metadata["topic"] for d in retriever.invoke(query)]
    print(f"{query!r} -> {topics}")

# %% [markdown]
# ## 3. Ask questions with memory

# %%
settings = load_settings()
print(f"\nModel: {settings.label}")
assistant = InsightForgeAssistant(sales, llm=get_chat_model(settings), model_label=settings.label)

for question in ["Which region generates the most revenue?", "And which product sells best there?"]:
    response = assistant.ask(question, session_id="lesson-4")
    print(f"\nQ: {question}\nStandalone: {response.standalone_question}\n{response.answer}")

# %% [markdown]
# ## Your exercises
#
# 1. Add a keyword to a document and see how retrieval changes.
# 2. Change `k` in the retriever and compare answers.
# 3. Edit `ANSWER_SYSTEM_PROMPT` in `prompts.py` to request a different answer format.
