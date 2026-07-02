"""
SHL Catalog Fetcher — downloads assessment data from the SHL JSON API.

Usage:
    python -m scraper.scrape

Output:
    data/processed/assessments.json
"""
import json
import os
import httpx


CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"


def fetch_catalog(url: str = CATALOG_URL) -> list[dict]:
    """
    Fetch SHL assessment catalog from the JSON API.

    Returns:
        List of raw assessment dicts
    """
    print(f"Fetching catalog from: {url}")
    response = httpx.get(url, timeout=30)
    response.raise_for_status()

    # Use strict=False to handle invalid control characters in the JSON
    assessments = json.loads(response.text, strict=False)
    print(f"Fetched {len(assessments)} assessments")
    return assessments


def clean_assessments(raw: list[dict]) -> list[dict]:
    """
    Clean and normalize assessment data.
    Keeps ALL original fields and adds computed fields on top.
    """
    cleaned = []
    seen = set()

    for item in raw:
        name = item.get("name", "").strip()
        if not name or name in seen:
            continue
        seen.add(name)

        # Start with ALL original fields
        assessment = dict(item)

        # Add computed fields (mapped names for our app)
        keys = item.get("keys", [])
        assessment["test_type"] = ", ".join(keys) if keys else "N/A"
        assessment["url"] = item.get("link", "")
        assessment["remote_testing"] = item.get("remote", "").capitalize()
        assessment["adaptive_testing"] = item.get("adaptive", "").capitalize()

        # Build text for embedding (concatenate all searchable fields)
        parts = [
            name,
            item.get("description", ""),
            assessment["test_type"],
            " ".join(item.get("job_levels", [])),
            item.get("duration", ""),
        ]
        assessment["text_for_embedding"] = " ".join(filter(None, parts))

        cleaned.append(assessment)

    print(f"Cleaned: {len(cleaned)} unique assessments")
    return cleaned


def save_assessments(assessments: list[dict], output_dir: str):
    """Save cleaned assessments to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "assessments.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    raw = fetch_catalog()
    cleaned = clean_assessments(raw)
    save_assessments(cleaned, data_dir)
