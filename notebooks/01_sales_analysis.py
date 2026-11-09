# ruff: noqa: I001
# %% [markdown]
# # Lesson 1: Explore sales data with pandas
#
# This notebook is our first small piece of InsightForge. We will:
#
# 1. Load a CSV file.
# 2. Understand rows and columns.
# 3. Create a calculated sales field.
# 4. Answer basic business questions.
#
# In VS Code, open this file and click **Run Cell** above each `# %%` section.
# You can also run the complete lesson from PowerShell:
#
# ```powershell
# uv run python notebooks/01_sales_analysis.py
# ```

# %%
from pathlib import Path

import pandas as pd


# This works whether the script is run from the project root or inside VS Code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "sales.csv"

sales = pd.read_csv(DATA_FILE, parse_dates=["date"])

print(sales.head())

# %% [markdown]
# ## Understand the dataset
#
# A pandas **DataFrame** is a table. Each row is an order and each column stores
# one property of that order.

# %%
print(f"Rows: {sales.shape[0]}")
print(f"Columns: {sales.shape[1]}")
print("\nColumn names:")
print(sales.columns.tolist())
print("\nData types:")
print(sales.dtypes)

# %% [markdown]
# ## Check data quality
#
# Missing values can make calculations misleading. The result below should be
# zero for every column in our prepared capstone dataset.

# %%
print(sales.isna().sum())

# %% [markdown]
# ## Calculate revenue
#
# Revenue for an order is its number of units multiplied by the price of one unit.

# %%
sales["revenue"] = sales["units"] * sales["unit_price"]

print(sales[["order_id", "product", "units", "unit_price", "revenue"]].head())

# %% [markdown]
# ## Core business metrics
#
# These are facts that pandas calculates. A future LLM will explain these facts in
# natural language rather than guessing the numbers itself.

# %%
total_revenue = sales["revenue"].sum()
average_order_value = sales["revenue"].mean()
median_order_value = sales["revenue"].median()
revenue_std_dev = sales["revenue"].std()

print(f"Total revenue: INR {total_revenue:,.0f}")
print(f"Average order value: INR {average_order_value:,.2f}")
print(f"Median order value: INR {median_order_value:,.2f}")
print(f"Revenue standard deviation: INR {revenue_std_dev:,.2f}")

# %% [markdown]
# ## Which regions perform best?
#
# `groupby` divides rows into regional groups. `sum` then totals the revenue in
# each group.

# %%
revenue_by_region = (
    sales.groupby("region", as_index=False)["revenue"]
    .sum()
    .sort_values("revenue", ascending=False)
)

print(revenue_by_region.to_string(index=False))

# %% [markdown]
# ## Which products perform best?

# %%
product_performance = (
    sales.groupby("product", as_index=False)
    .agg(units_sold=("units", "sum"), revenue=("revenue", "sum"))
    .sort_values("revenue", ascending=False)
)

print(product_performance.to_string(index=False))

# %% [markdown]
# ## Your exercises
#
# Try these before looking ahead:
#
# 1. Find the region with the lowest revenue.
# 2. Calculate revenue by customer segment.
# 3. Find the order with the largest revenue.
# 4. Change one order's unit count and observe which metrics change.

