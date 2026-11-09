# %% [markdown]
# # Lesson 3: Use the reusable analytics engine
#
# The first two lessons performed calculations directly inside notebooks. We now
# import reusable, tested functions from `src/insightforge/analytics.py`.
#
# This separation matters because notebooks, Streamlit, and our future AI assistant
# should all use the same calculations.

# %%
from pathlib import Path

from insightforge.analytics import (
    calculate_kpis,
    load_sales,
    monthly_revenue,
    prepare_sales,
    revenue_by,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sales = load_sales(PROJECT_ROOT / "data" / "sales.csv")
sales = prepare_sales(sales)

# %% [markdown]
# ## Headline metrics

# %%
kpis = calculate_kpis(sales)
for name, value in kpis.items():
    print(f"{name}: {value:,.2f}" if isinstance(value, float) else f"{name}: {value}")

# %% [markdown]
# ## Reuse one function for different business questions
#
# A *dimension* is the category used to divide a metric. Region, product, and
# customer segment are all dimensions; revenue is the metric.

# %%
print("\nRevenue by region")
print(revenue_by(sales, "region").to_string(index=False))

print("\nRevenue by customer segment")
print(revenue_by(sales, "customer_segment").to_string(index=False))

print("\nMonthly revenue")
print(monthly_revenue(sales).to_string(index=False))

# %% [markdown]
# ## Your exercises
#
# 1. Use `revenue_by` to calculate category performance.
# 2. Store the best-performing product row in a variable named `best_product`.
# 3. Why does `prepare_sales` return a copy instead of modifying the input table?
# 4. Run `uv run pytest` and examine which business rules the tests protect.

