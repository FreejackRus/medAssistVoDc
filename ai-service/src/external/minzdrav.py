from __future__ import annotations

import logging
import httpx

log = logging.getLogger(__name__)

API_URL = "https://apicr.minzdrav.gov.ru/api.ashx?op=GetJsonClinrecsFilterV2"


async def fetch_recommendations() -> list[dict]:
    """Fetch clinical recommendations from Minzdrav API."""
    payload = {
        "filters": [
            {
                "fieldName": "status",
                "filterType": 1,
                "filterValueType": 2,
                "value1": 0,
                "value2": "",
                "values": [],
            }
        ],
        "sortOption": {"fieldName": "publishdate", "sortType": 2},
        "pageSize": 999999,
        "currentPage": 1,
        "useANDoperator": True,
        "columns": [],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            API_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; MedAssistant/1.0)",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    recs = []
    for item in data.get("Data", []):
        mkbs = item.get("Mkbs") or []
        recs.append({
            "id": item.get("CodeVersion", item.get("Id", "")),
            "title": item.get("Name", ""),
            "description": item.get("Description", ""),
            "annotation": item.get("Annotation", ""),
            "publishdate": item.get("PublishDateStr", ""),
            "status": item.get("Status", 0),
            "organization": "Минздрав РФ",
            "mkb_code": mkbs[0].get("MkbCode", "") if mkbs else item.get("Code", ""),
            "category": item.get("Category", ""),
            "age_group": item.get("AgeCategoryStr", ""),
            "keywords": item.get("Keywords", ""),
            "version": item.get("Version", ""),
            "code_version": item.get("CodeVersion", ""),
        })
    return recs
