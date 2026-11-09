import pandas as pd
import pytest

from insightforge.analytics import calculate_kpis, prepare_sales, revenue_by


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["A", "B", "C"],
            "date": ["2026-01-01", "2026-01-10", "2026-02-01"],
            "product": ["Laptop", "Mouse", "Laptop"],
            "category": ["Computers", "Accessories", "Computers"],
            "region": ["South", "North", "South"],
            "customer_age": [25, 40, 32],
            "customer_segment": ["Consumer", "Corporate", "Consumer"],
            "units": [1, 2, 1],
            "unit_price": [1000, 100, 1000],
        }
    )


def test_prepare_sales_calculates_revenue_without_mutating_input(
    sample_sales: pd.DataFrame,
) -> None:
    prepared = prepare_sales(sample_sales)

    assert "revenue" not in sample_sales.columns
    assert prepared["revenue"].tolist() == [1000, 200, 1000]
    assert prepared["month"].tolist() == ["2026-01", "2026-01", "2026-02"]


def test_calculate_kpis_returns_expected_values(sample_sales: pd.DataFrame) -> None:
    kpis = calculate_kpis(prepare_sales(sample_sales))

    assert kpis["total_revenue"] == 2200
    assert kpis["total_orders"] == 3
    assert kpis["total_units"] == 4
    assert kpis["average_order_value"] == pytest.approx(733.33, abs=0.01)
    assert kpis["median_order_value"] == 1000


def test_revenue_by_region_is_sorted_highest_first(sample_sales: pd.DataFrame) -> None:
    result = revenue_by(prepare_sales(sample_sales), "region")

    assert result.to_dict("records") == [
        {"region": "South", "revenue": 2000},
        {"region": "North", "revenue": 200},
    ]


def test_revenue_by_rejects_unknown_dimension(sample_sales: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported dimension"):
        revenue_by(prepare_sales(sample_sales), "salesperson")


def test_metrics_require_prepared_data(sample_sales: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must be prepared"):
        calculate_kpis(sample_sales)
