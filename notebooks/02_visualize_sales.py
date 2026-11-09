# ruff: noqa: I001
# %% [markdown]
# # Lesson 2: Visualize sales and explain business insights
#
# A chart should answer a business question. In this lesson we create four charts:
#
# 1. How is revenue changing over time?
# 2. Which regions generate the most revenue?
# 3. Which products generate the most revenue?
# 4. Which customer segments contribute the most revenue?
#
# Run the full lesson from the project root:
#
# ```powershell
# uv run python notebooks/02_visualize_sales.py
# ```

# %%
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_FILE = PROJECT_ROOT / "data" / "sales.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sales = pd.read_csv(DATA_FILE, parse_dates=["date"])
sales["revenue"] = sales["units"] * sales["unit_price"]
sales["month"] = sales["date"].dt.to_period("M").astype(str)

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Monthly revenue trend
#
# We group orders by month and add their revenue. A line chart is suitable because
# time has a natural order.

# %%
monthly_revenue = sales.groupby("month", as_index=False)["revenue"].sum()

fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(data=monthly_revenue, x="month", y="revenue", marker="o", ax=ax)
ax.set(title="Monthly Revenue Trend", xlabel="Month", ylabel="Revenue (INR)")
ax.ticklabel_format(style="plain", axis="y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "monthly_revenue.png", dpi=150)
plt.close(fig)

print("Monthly revenue")
print(monthly_revenue.to_string(index=False))

# %% [markdown]
# ## 2. Revenue by region
#
# Bar charts make comparisons between categories easy. Sorting the bars helps the
# reader identify the strongest and weakest regions immediately.

# %%
regional_revenue = (
    sales.groupby("region", as_index=False)["revenue"]
    .sum()
    .sort_values("revenue", ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=regional_revenue, x="region", y="revenue", hue="region", ax=ax)
ax.set(title="Revenue by Region", xlabel="Region", ylabel="Revenue (INR)")
ax.ticklabel_format(style="plain", axis="y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "regional_revenue.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 3. Product performance
#
# Product names can be long, so a horizontal bar chart gives the labels more room.

# %%
product_revenue = (
    sales.groupby("product", as_index=False)["revenue"]
    .sum()
    .sort_values("revenue", ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=product_revenue, x="revenue", y="product", hue="product", ax=ax)
ax.set(title="Revenue by Product", xlabel="Revenue (INR)", ylabel="Product")
ax.ticklabel_format(style="plain", axis="x")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "product_revenue.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 4. Customer-segment performance

# %%
segment_revenue = (
    sales.groupby("customer_segment", as_index=False)["revenue"]
    .sum()
    .sort_values("revenue", ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=segment_revenue,
    x="customer_segment",
    y="revenue",
    hue="customer_segment",
    ax=ax,
)
ax.set(title="Revenue by Customer Segment", xlabel="Customer segment", ylabel="Revenue (INR)")
ax.ticklabel_format(style="plain", axis="y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "segment_revenue.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Convert facts into written insights
#
# The code below uses calculated values to create dependable statements. Later, an
# LLM will turn structured results like these into richer natural-language answers.

# %%
best_month = monthly_revenue.loc[monthly_revenue["revenue"].idxmax()]
best_region = regional_revenue.iloc[0]
best_product = product_revenue.iloc[-1]
best_segment = segment_revenue.iloc[0]

print("\nBusiness insights")
print(f"- {best_month['month']} was the strongest month at INR {best_month['revenue']:,.0f}.")
print(f"- {best_region['region']} was the strongest region at INR {best_region['revenue']:,.0f}.")
print(f"- {best_product['product']} led product revenue at INR {best_product['revenue']:,.0f}.")
print(
    f"- {best_segment['customer_segment']} was the strongest customer segment "
    f"at INR {best_segment['revenue']:,.0f}."
)
print(f"\nSaved charts to: {FIGURES_DIR}")

# %% [markdown]
# ## Your exercises
#
# 1. Which month had the lowest revenue?
# 2. Create a chart showing total units sold by product.
# 3. Add data labels to one bar chart.
# 4. In one sentence, explain the most important finding to a sales manager.

