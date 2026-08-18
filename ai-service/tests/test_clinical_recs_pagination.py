from src.api.router_clinical_recs import _paginate_recommendations


RECOMMENDATIONS = [
    {
        "id": "1",
        "title": "Артериальная гипертензия",
        "mkb_code": "I10",
        "keywords": "давление",
    },
    {
        "id": "2",
        "title": "Сахарный диабет",
        "mkb_code": "E11",
        "keywords": "глюкоза инсулин",
    },
    {
        "id": "3",
        "title": "Сердечная недостаточность",
        "mkb_code": "I50",
        "keywords": "сердце",
    },
]


def test_pagination_returns_requested_slice_and_metadata() -> None:
    response = _paginate_recommendations(
        RECOMMENDATIONS,
        query="",
        page=2,
        page_size=2,
    )

    assert response["total"] == 3
    assert response["total_pages"] == 2
    assert response["page"] == 2
    assert [item["id"] for item in response["recommendations"]] == ["3"]


def test_search_is_case_insensitive_across_supported_fields() -> None:
    by_title = _paginate_recommendations(
        RECOMMENDATIONS,
        query="ДИАБЕТ",
        page=1,
        page_size=20,
    )
    by_mkb = _paginate_recommendations(
        RECOMMENDATIONS,
        query="i50",
        page=1,
        page_size=20,
    )
    by_keyword = _paginate_recommendations(
        RECOMMENDATIONS,
        query="ДАВЛЕНИЕ",
        page=1,
        page_size=20,
    )

    assert [item["id"] for item in by_title["recommendations"]] == ["2"]
    assert [item["id"] for item in by_mkb["recommendations"]] == ["3"]
    assert [item["id"] for item in by_keyword["recommendations"]] == ["1"]
