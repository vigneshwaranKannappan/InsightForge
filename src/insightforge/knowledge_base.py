"""Turn trusted pandas statistics into a retrievable knowledge base.

Each document is a small, self-contained "fact sheet" about one topic (a time period,
a product, a region, a demographic group, ...). The LLM never calculates numbers
itself: it only explains the facts stored here.
"""

import pandas as pd
from langchain_core.documents import Document

from insightforge.analytics import (
    calculate_kpis,
    cross_tab_revenue,
    dimension_performance,
    key_insights,
    period_performance,
    statistical_summary,
)

DIMENSION_LABELS = {
    "product": "Product",
    "category": "Category",
    "region": "Region",
    "customer_segment": "Customer segment",
    "age_band": "Age band",
}

DIMENSION_KEYWORDS = {
    "product": ["product", "products", "item", "items", "best-selling", "seller", "sku"],
    "category": ["category", "categories", "type"],
    "region": ["region", "regions", "regional", "area", "geography", "location", "where"],
    "customer_segment": ["segment", "segments", "segmentation", "customer", "customers", "b2b"],
    "age_band": [
        "age",
        "ages",
        "group",
        "groups",
        "band",
        "bands",
        "demographic",
        "demographics",
        "young",
        "old",
        "older",
        "younger",
    ],
}


def _inr(value: float) -> str:
    return f"INR {value:,.0f}"


def _dimension_documents(sales: pd.DataFrame, dimension: str) -> list[Document]:
    table = dimension_performance(sales, dimension)
    label = DIMENSION_LABELS[dimension]
    keywords = DIMENSION_KEYWORDS[dimension]

    lines = [f"{label} performance ranked by revenue:"]
    for rank, row in enumerate(table.itertuples(index=False), start=1):
        lines.append(
            f"{rank}. {getattr(row, dimension)}: revenue {_inr(row.revenue)} "
            f"({row.revenue_share_pct}% share), {row.orders} orders, {row.units} units, "
            f"average order value {_inr(row.average_order_value)}, "
            f"average customer age {row.average_age}."
        )
    top, bottom = table.iloc[0], table.iloc[-1]
    lines.append(
        f"Top {label.lower()}: {top[dimension]}. Lowest {label.lower()}: {bottom[dimension]}."
    )

    documents = [
        Document(
            page_content="\n".join(lines),
            metadata={"topic": f"{dimension}_overview", "keywords": keywords},
        )
    ]

    for row in table.itertuples(index=False):
        name = str(getattr(row, dimension))
        subset = sales[sales[dimension] == name]
        detail = [
            (
                f"{label} '{name}' details: revenue {_inr(row.revenue)} "
                f"({row.revenue_share_pct}% of total), {row.orders} orders, {row.units} units, "
                f"average order value {_inr(row.average_order_value)}, "
                f"average customer age {row.average_age}."
            )
        ]
        for other in ("product", "region", "customer_segment"):
            if other == dimension:
                continue
            breakdown = subset.groupby(other)["revenue"].sum().sort_values(ascending=False)
            parts = ", ".join(f"{k} {_inr(v)}" for k, v in breakdown.items())
            detail.append(f"Revenue by {DIMENSION_LABELS[other].lower()}: {parts}.")
        monthly = subset.groupby("month")["revenue"].sum()
        detail.append(
            "Monthly revenue: " + ", ".join(f"{k} {_inr(v)}" for k, v in monthly.items()) + "."
        )
        documents.append(
            Document(
                page_content="\n".join(detail),
                metadata={
                    "topic": f"{dimension}:{name}",
                    "keywords": [*keywords, *name.lower().split()],
                },
            )
        )
    return documents


def build_knowledge_base(sales: pd.DataFrame) -> list[Document]:
    """Build retrievable fact documents from prepared sales data."""
    kpis = calculate_kpis(sales)
    start, end = sales["date"].min().date(), sales["date"].max().date()

    documents = [
        Document(
            page_content=(
                f"Business overview for {start} to {end}: total revenue "
                f"{_inr(kpis['total_revenue'])}, {kpis['total_orders']} orders, "
                f"{kpis['total_units']} units sold. Average order value "
                f"{_inr(kpis['average_order_value'])}, median order value "
                f"{_inr(kpis['median_order_value'])}, order revenue standard deviation "
                f"{_inr(kpis['revenue_std_dev'])}."
            ),
            metadata={
                "topic": "overview",
                "keywords": [
                    "total",
                    "overall",
                    "revenue",
                    "sales",
                    "kpi",
                    "kpis",
                    "summary",
                    "overview",
                    "orders",
                    "performance",
                    "business",
                    "aov",
                    "average",
                ],
            },
        ),
        Document(
            page_content="Key insights:\n" + "\n".join(f"- {i}" for i in key_insights(sales)),
            metadata={
                "topic": "insights",
                "keywords": [
                    "insight",
                    "insights",
                    "recommend",
                    "recommendation",
                    "recommendations",
                    "improve",
                    "strategy",
                    "should",
                    "summary",
                    "highlights",
                    "key",
                    "action",
                ],
            },
        ),
    ]

    stats = statistical_summary(sales)
    stat_lines = ["Statistical measures (count, mean, median, standard deviation, min, max):"]
    for field, row in stats.iterrows():
        stat_lines.append(
            f"- {field}: count {row['count']:.0f}, mean {row['mean']:,.2f}, "
            f"median {row['median']:,.2f}, std dev {row['std']:,.2f}, "
            f"min {row['min']:,.2f}, max {row['max']:,.2f}, "
            f"Q1 {row['q1']:,.2f}, Q3 {row['q3']:,.2f}"
        )
    documents.append(
        Document(
            page_content="\n".join(stat_lines),
            metadata={
                "topic": "statistics",
                "keywords": [
                    "median",
                    "mean",
                    "std",
                    "standard",
                    "deviation",
                    "statistics",
                    "statistical",
                    "distribution",
                    "variance",
                    "spread",
                    "quartile",
                    "min",
                    "max",
                    "average",
                ],
            },
        ),
    )

    for period, words in (
        ("month", ["month", "monthly", "trend", "trends", "time", "growth", "over", "mom"]),
        ("quarter", ["quarter", "quarterly", "q1", "q2", "trend", "time", "growth"]),
    ):
        table = period_performance(sales, period)
        lines = [f"Sales performance by {period} (chronological):"]
        for row in table.itertuples(index=False):
            growth = "n/a" if pd.isna(row.growth_pct) else f"{row.growth_pct:+.1f}%"
            lines.append(
                f"- {getattr(row, period)}: revenue {_inr(row.revenue)}, {row.orders} orders, "
                f"{row.units} units, growth vs previous {period} {growth}"
            )
        best = table.loc[table["revenue"].idxmax()]
        worst = table.loc[table["revenue"].idxmin()]
        lines.append(
            f"Best {period}: {best[period]} ({_inr(best['revenue'])}). "
            f"Weakest {period}: {worst[period]} ({_inr(worst['revenue'])})."
        )
        month_names = (
            [m.lower() for m in pd.to_datetime(table["month"]).dt.month_name()]
            if period == "month"
            else []
        )
        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={"topic": f"{period}_trend", "keywords": [*words, *month_names]},
            )
        )

    for dimension in DIMENSION_LABELS:
        documents.extend(_dimension_documents(sales, dimension))

    matrix = cross_tab_revenue(sales, "region", "product")
    matrix_lines = ["Revenue by region and product (INR):"]
    for region, row in matrix.iterrows():
        sold = ", ".join(f"{product} {value:,.0f}" for product, value in row.items() if value)
        matrix_lines.append(f"- {region}: {sold}")
    documents.append(
        Document(
            page_content="\n".join(matrix_lines),
            metadata={
                "topic": "region_product_matrix",
                "keywords": ["region", "product", "matrix", "combination", "mix", "where", "which"],
            },
        )
    )
    return documents
