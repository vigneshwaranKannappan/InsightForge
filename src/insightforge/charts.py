"""Plotly figures shared by the Streamlit app and notebooks."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from insightforge.analytics import cross_tab_revenue, dimension_performance, period_performance

TEMPLATE = "plotly_white"
INR_AXIS = {"tickprefix": "₹", "tickformat": ",.0f"}


def sales_trend_figure(sales: pd.DataFrame) -> go.Figure:
    monthly = period_performance(sales, "month")
    fig = px.line(
        monthly,
        x="month",
        y="revenue",
        markers=True,
        hover_data={"orders": True, "units": True, "growth_pct": True},
        title="Sales trend over time (monthly revenue)",
        template=TEMPLATE,
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue", yaxis=INR_AXIS)
    return fig


def product_performance_figure(sales: pd.DataFrame) -> go.Figure:
    products = dimension_performance(sales, "product").sort_values("revenue")
    category = sales.groupby("product")["category"].first()
    products["category"] = products["product"].map(category)
    fig = px.bar(
        products,
        x="revenue",
        y="product",
        color="category",
        orientation="h",
        text="revenue_share_pct",
        hover_data={"units": True, "orders": True},
        title="Product performance comparison",
        template=TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title="Revenue", yaxis_title="", xaxis=INR_AXIS)
    return fig


def regional_figure(sales: pd.DataFrame) -> go.Figure:
    regions = dimension_performance(sales, "region")
    fig = px.bar(
        regions,
        x="region",
        y="revenue",
        text="revenue_share_pct",
        hover_data={"orders": True, "units": True, "average_order_value": ":,.0f"},
        title="Regional analysis",
        template=TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title="Region", yaxis_title="Revenue", yaxis=INR_AXIS)
    return fig


def region_product_heatmap(sales: pd.DataFrame) -> go.Figure:
    matrix = cross_tab_revenue(sales, "region", "product")
    fig = px.imshow(
        matrix,
        text_auto=",.0f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="Revenue by region and product (INR)",
        template=TEMPLATE,
    )
    fig.update_layout(xaxis_title="Product", yaxis_title="Region")
    return fig


def segment_figure(sales: pd.DataFrame) -> go.Figure:
    segments = dimension_performance(sales, "customer_segment")
    fig = px.pie(
        segments,
        names="customer_segment",
        values="revenue",
        hole=0.5,
        title="Revenue by customer segment",
        template=TEMPLATE,
    )
    fig.update_traces(textinfo="label+percent")
    return fig


def demographics_figure(sales: pd.DataFrame) -> go.Figure:
    ages = (
        sales.groupby(["age_band", "customer_segment"], as_index=False)["revenue"]
        .sum()
        .sort_values("age_band")
    )
    fig = px.bar(
        ages,
        x="age_band",
        y="revenue",
        color="customer_segment",
        title="Customer demographics: revenue by age band and segment",
        template=TEMPLATE,
    )
    fig.update_layout(xaxis_title="Age band", yaxis_title="Revenue", yaxis=INR_AXIS)
    return fig


def age_distribution_figure(sales: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        sales,
        x="customer_age",
        color="customer_segment",
        nbins=10,
        title="Customer age distribution",
        template=TEMPLATE,
    )
    fig.update_layout(xaxis_title="Customer age", yaxis_title="Orders")
    return fig
