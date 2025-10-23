"""
Utility helpers for fetching external resources via the Firecrawl API.

This provides a consistent fallback path for demo environments where direct
requests to NOAA/USGS/FEMA endpoints might fail due to rate limits or network
restrictions. The API key must be supplied through the FIRECRAWL_API_KEY
environment variable.
"""

from __future__ import annotations

import os

import requests

FIRECRAWL_API_URL = os.environ.get("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v1/scrape")


class FirecrawlUnavailable(RuntimeError):
    """Raised when Firecrawl cannot satisfy a data request."""


def fetch_via_firecrawl(url: str, timeout: int = 60, render: bool = False) -> str:
    """
    Retrieve the contents at the provided URL via Firecrawl.

    Parameters
    ----------
    url:
        The fully qualified URL to fetch.
    timeout:
        Request timeout in seconds.
    render:
        Whether Firecrawl should render the page (for JS-heavy sites). Defaults to
        False because our targets are static text/JSON files.

    Returns
    -------
    str
        The raw textual payload Firecrawl returned.

    Raises
    ------
    FirecrawlUnavailable
        If the API key is missing or the upstream call fails.
    """

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise FirecrawlUnavailable("FIRECRAWL_API_KEY is not set.")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"url": url, "render": render}

    try:
        resp = requests.post(FIRECRAWL_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise FirecrawlUnavailable(f"Firecrawl request failed: {exc}") from exc

    data = resp.json()

    # Firecrawl responses may include multiple keys depending on the selected mode.
    text_candidates = [
        data.get("data", {}).get("content"),
        data.get("data", {}).get("markdown"),
        data.get("data", {}).get("raw_document"),
        data.get("content"),
        data.get("markdown"),
    ]
    for candidate in text_candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    raise FirecrawlUnavailable("Firecrawl response did not contain textual content.")


def fetch_json_via_firecrawl(url: str, timeout: int = 60) -> dict:
    """
    Fetch a JSON resource via Firecrawl and parse it.

    Firecrawl may return markdown-wrapped JSON; we attempt to strip fencing and
    decode the payload. When parsing fails, a FirecrawlUnavailable error is raised.
    """
    import json

    text = fetch_via_firecrawl(url, timeout=timeout)
    text = text.strip()

    # Remove markdown fences or extra annotations if present.
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise FirecrawlUnavailable(f"Unable to parse JSON from Firecrawl response for {url}: {exc}") from exc


def fetch_lines_via_firecrawl(url: str, timeout: int = 60) -> list[str]:
    """
    Fetch a text resource via Firecrawl and return the lines for easier parsing.
    """
    text = fetch_via_firecrawl(url, timeout=timeout)
    return text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
