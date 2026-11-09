# %% [markdown]
# # Lesson 5: Evaluate the assistant with QAEvalChain
#
# Ground-truth answers come from pandas. `QAEvalChain` asks an LLM grader whether each
# assistant answer matches the ground truth. A deterministic fact check runs too, so
# the lesson also works offline.
#
# ```powershell
# uv run python notebooks/05_evaluation.py
# ```

# %%
from pathlib import Path

import pandas as pd

from insightforge.analytics import load_sales, prepare_sales
from insightforge.assistant import InsightForgeAssistant
from insightforge.evaluation import accuracy, build_eval_set, run_evaluation
from insightforge.llm import get_chat_model, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sales = prepare_sales(load_sales(PROJECT_ROOT / "data" / "sales.csv"))

settings = load_settings()
llm = get_chat_model(settings)
assistant = InsightForgeAssistant(sales, llm=llm, model_label=settings.label)

# %%
examples = build_eval_set(sales)
results = run_evaluation(assistant, examples, grader=llm)

pd.set_option("display.max_colwidth", 60)
print(results[["question", "qa_eval_grade", "fact_check_grade", "latency_seconds"]])

qa = accuracy(results, "qa_eval_grade")
print(f"\nModel: {settings.label}")
print(f"QAEvalChain accuracy: {'n/a (offline)' if qa is None else f'{qa:.0%}'}")
print(f"Fact-check accuracy: {accuracy(results, 'fact_check_grade'):.0%}")

REPORT = PROJECT_ROOT / "reports" / "evaluation_results.csv"
results.to_csv(REPORT, index=False)
print(f"Saved: {REPORT}")
