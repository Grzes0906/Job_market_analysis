"""
Backend module for the Polish IT Job Market Analysis project.
Fetches job postings from the Adzuna API, cleans the data, and deduplicates records.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base API endpoint for the Polish job market
ADZUNA_BASE_URL: str = "https://api.adzuna.com/v1/api/jobs/pl/search/1"

# Technologies to search for
SEARCH_TERMS: list[str] = [
    "Python",
    "Java",
    "JavaScript",
    "Data",
    "DevOps",
    "React",
    "SQL",
    "Machine Learning",
    "AWS",
    "Go",
]

# Max results per API request
RESULTS_PER_PAGE: int = 50

# Delay between requests to avoid rate limits
REQUEST_DELAY_SECONDS: float = 1.0

# Salary thresholds for outlier removal (annual PLN)
SALARY_MIN_THRESHOLD: float = 30_000.0
SALARY_MAX_THRESHOLD: float = 1_000_000.0


def _load_credentials() -> tuple[str, str]:
    """Load Adzuna API credentials from the .env file."""
    load_dotenv()

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise ValueError("Missing ADZUNA_APP_ID or ADZUNA_APP_KEY in .env file.")

    return app_id, app_key


def fetch_remote_jobs() -> pd.DataFrame:
    """Fetch job postings for all search terms and return a deduplicated DataFrame."""
    app_id, app_key = _load_credentials()
    all_records: list[dict[str, Any]] = []

    for index, term in enumerate(SEARCH_TERMS):
        term_records = _fetch_term_jobs(term, app_id, app_key)
        all_records.extend(term_records)
        logger.info(
            "Term '%s' (%d/%d): %d postings collected — running total: %d.",
            term, index + 1, len(SEARCH_TERMS), len(term_records), len(all_records),
        )

        if index < len(SEARCH_TERMS) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not all_records:
        raise ValueError("No job records were collected. Check API limits or network.")

    df = pd.DataFrame(all_records)

    # Force numeric types for salary columns
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")

    # Remove duplicates based on company and position
    pre_dedup_count = len(df)
    df = df.drop_duplicates(subset=["company", "position"]).reset_index(drop=True)

    logger.info(
        "Deduplication: %d → %d rows (removed %d duplicates).",
        pre_dedup_count, len(df), pre_dedup_count - len(df),
    )

    return df


def _fetch_term_jobs(term: str, app_id: str, app_key: str) -> list[dict[str, Any]]:
    """Fetch and parse job postings for a single search term."""
    params: dict[str, Any] = {
        "app_id": app_id,
        "app_key": app_key,
        "what": term,
        "results_per_page": RESULTS_PER_PAGE,
        "content-type": "application/json",
    }

    logger.info("GET %s  [what=%s]", ADZUNA_BASE_URL, term)

    try:
        response = requests.get(ADZUNA_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Network or HTTP error for term '%s' (%s).", term, exc)
        return []

    payload: Any = response.json()

    if not isinstance(payload, dict) or "results" not in payload:
        logger.warning("Unexpected response format for term '%s'.", term)
        return []

    raw_results: Any = payload["results"]

    if not isinstance(raw_results, list):
        return []

    return _parse_job_entries(raw_results)


def _parse_job_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract specific fields from raw API job entries."""
    records: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        company = (entry.get("company") or {}).get("display_name", "")
        location = (entry.get("location") or {}).get("display_name", "")

        records.append({
            "company": company,
            "position": entry.get("title", ""),
            "location": location,
            "salary_min": _to_float(entry.get("salary_min")),
            "salary_max": _to_float(entry.get("salary_max")),
        })

    return records


def clean_and_analyze_salaries(df: pd.DataFrame) -> pd.DataFrame:
    """Clean salary data, calculate average salary, and remove outliers."""
    if not {"salary_min", "salary_max"}.issubset(df.columns):
        raise ValueError("Missing required salary columns.")

    clean = df.copy()

    clean["salary_min"] = pd.to_numeric(clean["salary_min"], errors="coerce").replace(0, np.nan)
    clean["salary_max"] = pd.to_numeric(clean["salary_max"], errors="coerce").replace(0, np.nan)

    # Drop rows where both min and max salaries are missing
    both_missing_mask = clean["salary_min"].isna() & clean["salary_max"].isna()
    clean = clean[~both_missing_mask].copy()

    # Calculate average salary using vectorization
    salary_matrix = np.column_stack([
        clean["salary_min"].to_numpy(dtype=float, na_value=np.nan),
        clean["salary_max"].to_numpy(dtype=float, na_value=np.nan),
    ])
    clean["salary_avg"] = np.nanmean(salary_matrix, axis=1)

    # Filter out unrealistic salaries
    valid_range_mask = clean["salary_avg"].between(SALARY_MIN_THRESHOLD, SALARY_MAX_THRESHOLD, inclusive="both")
    clean = clean[valid_range_mask].reset_index(drop=True)

    return clean


def _to_float(value: Any) -> float | None:
    """Convert value to float, return None if conversion fails."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    """Run an end-to-end smoke test."""
    print("=" * 62)
    print("  Polish IT Job Market — Adzuna Backend Smoke Test")
    print(f"  Querying {len(SEARCH_TERMS)} terms: {', '.join(SEARCH_TERMS)}")
    print("=" * 62)

    try:
        raw_df = fetch_remote_jobs()
    except ValueError as exc:
        logger.error("Fatal fetch error: %s", exc)
        sys.exit(1)

    salary_coverage = raw_df[["salary_min", "salary_max"]].notna().any(axis=1)
    print(f"\n[1] Raw fetch shape: {raw_df.shape}")
    print(f"    Jobs with salary data: {salary_coverage.sum()} / {len(raw_df)}")

    try:
        clean_df = clean_and_analyze_salaries(raw_df)
    except ValueError as exc:
        logger.error("Cleaning failed: %s", exc)
        sys.exit(1)

    print(f"\n[2] Cleaned shape: {clean_df.shape}")

    salary_cols = ["salary_min", "salary_max", "salary_avg"]
    print(f"\n[3] Salary stats [PLN, annual] (n={len(clean_df)}):")
    print(clean_df[salary_cols].describe().round(2).to_string())
    print("\nDone. ✓")


if __name__ == "__main__":
    main()