"""Trusted calculations for the InsightForge business intelligence assistant."""

from pathlib import Path
from typing import TypedDict

import pandas as pd

REQUIRED_COLUMNS = {
    "order_id",
    "date",
    "product",
    "category",
    "region",
    "customer_age",
    "customer_segment",
    "units",
    "unit_price",
}

ALLOWED_DIMENSIONS = {"product", "category", "region", "customer_segment"}


class SalesKPIs(TypedDict):
    """The headline metrics displayed by InsightForge."""

    total_revenue: float
    total_orders: int
    total_units: int
    average_order_value: float
    median_order_value: float
    revenue_std_dev: float


def load_sales(file_path: str | Path) -> pd.DataFrame:
    """Load a sales CSV and validate that its required columns are present."""
    sales = pd.read_csv(file_path, parse_dates=["date"])
    missing_columns = REQUIRED_COLUMNS.difference(sales.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Sales data is missing required columns: {missing}")

    return sales


AGE_BINS = [0, 24, 34, 44, 54, 200]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]


def prepare_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with calculated revenue, month, quarter and age-band columns."""
    prepared = sales.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["revenue"] = prepared["units"] * prepared["unit_price"]
    prepared["month"] = prepared["date"].dt.to_period("M").astype(str)
    prepared["quarter"] = prepared["date"].dt.to_period("Q").astype(str)
    prepared["age_band"] = pd.cut(
        prepared["customer_age"], bins=AGE_BINS, labels=AGE_LABELS
    ).astype(str)
    return prepared


def calculate_kpis(sales: pd.DataFrame) -> SalesKPIs:
    """Calculate headline sales metrics from prepared sales data."""
    _require_revenue(sales)
    return {
        "total_revenue": float(sales["revenue"].sum()),
        "total_orders": int(sales["order_id"].nunique()),
        "total_units": int(sales["units"].sum()),
        "average_order_value": float(sales["revenue"].mean()),
        "median_order_value": float(sales["revenue"].median()),
        "revenue_std_dev": float(sales["revenue"].std()),
    }


def revenue_by(sales: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate revenue by an approved business dimension."""
    _require_revenue(sales)
    if dimension not in ALLOWED_DIMENSIONS:
        allowed = ", ".join(sorted(ALLOWED_DIMENSIONS))
        raise ValueError(f"Unsupported dimension '{dimension}'. Choose from: {allowed}")

    return (
        sales.groupby(dimension, as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )


def monthly_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by calendar month in chronological order."""
    _require_revenue(sales)
    if "month" not in sales.columns:
        raise ValueError("Sales data must be prepared before calculating monthly revenue")

    return sales.groupby("month", as_index=False)["revenue"].sum().sort_values("month")


def period_performance(sales: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Revenue, orders, units and period-over-period growth by month or quarter."""
    _require_columns(sales, {"revenue", period})
    if period not in {"month", "quarter"}:
        raise ValueError("period must be 'month' or 'quarter'")

    result = (
        sales.groupby(period, as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units=("units", "sum"),
        )
        .sort_values(period)
        .reset_index(drop=True)
    )
    result["growth_pct"] = (result["revenue"].pct_change() * 100).round(1)
    return result


def dimension_performance(sales: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Revenue, orders, units, average order value and revenue share for a dimension."""
    _require_revenue(sales)
    if dimension not in ALLOWED_DIMENSIONS | {"age_band"}:
        raise ValueError(f"Unsupported dimension '{dimension}'")

    result = (
        sales.groupby(dimension, as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units=("units", "sum"),
            average_age=("customer_age", "mean"),
        )
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    result["average_order_value"] = result["revenue"] / result["orders"]
    result["revenue_share_pct"] = (result["revenue"] / result["revenue"].sum() * 100).round(1)
    result["average_age"] = result["average_age"].round(1)
    return result


def cross_tab_revenue(sales: pd.DataFrame, rows: str, columns: str) -> pd.DataFrame:
    """Revenue pivot table, e.g. region x product."""
    _require_revenue(sales)
    return sales.pivot_table(
        index=rows, columns=columns, values="revenue", aggfunc="sum", fill_value=0
    )


def statistical_summary(sales: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics (mean, median, std, quartiles) for numeric business fields."""
    _require_revenue(sales)
    fields = ["revenue", "units", "unit_price", "customer_age"]
    summary = sales[fields].agg(["count", "mean", "median", "std", "min", "max"]).T
    summary["q1"] = sales[fields].quantile(0.25)
    summary["q3"] = sales[fields].quantile(0.75)
    return summary.round(2)


def key_insights(sales: pd.DataFrame) -> list[str]:
    """Deterministic, fact-checked headline insights used by the dashboard."""
    months = period_performance(sales, "month")
    products = dimension_performance(sales, "product")
    regions = dimension_performance(sales, "region")
    segments = dimension_performance(sales, "customer_segment")
    ages = dimension_performance(sales, "age_band")

    best_month = months.loc[months["revenue"].idxmax()]
    worst_month = months.loc[months["revenue"].idxmin()]
    latest = months.iloc[-1]
    top_product, low_product = products.iloc[0], products.iloc[-1]
    top_region, low_region = regions.iloc[0], regions.iloc[-1]
    top_segment = segments.iloc[0]
    top_age = ages.iloc[0]

    insights = [
        (
            f"{best_month['month']} was the strongest month (INR {best_month['revenue']:,.0f}); "
            f"{worst_month['month']} was the weakest (INR {worst_month['revenue']:,.0f})."
        ),
        (
            f"Latest month {latest['month']} changed {latest['growth_pct']:+.1f}% versus "
            "the previous month."
        ),
        (
            f"{top_product['product']} leads products with {top_product['revenue_share_pct']}% of "
            f"revenue; {low_product['product']} contributes only "
            f"{low_product['revenue_share_pct']}%."
        ),
        (
            f"{top_region['region']} is the top region (INR {top_region['revenue']:,.0f}, "
            f"{top_region['revenue_share_pct']}%); {low_region['region']} trails at "
            f"{low_region['revenue_share_pct']}%."
        ),
        (
            f"{top_segment['customer_segment']} customers generate the most revenue "
            f"({top_segment['revenue_share_pct']}%)."
        ),
        (
            f"Customers aged {top_age['age_band']} generate the most revenue "
            f"({top_age['revenue_share_pct']}%, {top_age['orders']} orders)."
        ),
    ]
    return insights


def _require_columns(sales: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(sales.columns)
    if missing:
        raise ValueError(
            "Sales data must be prepared before analysis; missing: " + ", ".join(sorted(missing))
        )


def _require_revenue(sales: pd.DataFrame) -> None:
    if "revenue" not in sales.columns:
        raise ValueError("Sales data must be prepared before calculating revenue metrics")
