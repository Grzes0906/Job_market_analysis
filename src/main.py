"""
src/main.py
-----------
Core backend module for the Global Remote IT Job Market Analysis project.

Aggregates remote IT job postings from the RemoteOK public API across a
curated list of technology tags (e.g. "python", "aws", "react") to build a
statistically significant dataset.  Each tag is queried individually via
``/api?tags=<tag>``, results are merged into one collection, and exact
duplicates (jobs that matched multiple tags) are removed before the DataFrame
is returned.  A configurable polite delay between requests prevents
rate-limiting.

Pipeline
--------
1. ``fetch_remote_jobs()``           – multi-tag ingestion → deduplicated DataFrame
2. ``clean_and_analyze_salaries()``  – salary cleaning, outlier removal, avg column

Usage (standalone smoke-test):
    python src/main.py
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# RemoteOK rejects requests without a meaningful User-Agent (HTTP 403).
_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "RemoteJobsResearch/1.0 "
        "(Academic data-analysis project; "
        "github.com/your-org/remote-jobs-analysis)"
    )
}

# Public RemoteOK API base endpoint (no auth required).
REMOTEOK_API_URL: str = "https://remoteok.com/api"

# Technology tags used to query the API individually.
# Each tag maps to  GET /api?tags=<tag>.  Broaden or narrow this list to
# control dataset size vs. crawl time.
TECH_TAGS: list[str] = [
    "python",
    "javascript",
    "data",
    "aws",
    "react",
    "node",
    "devops",
    "go",
    "sql",
    "machine-learning",
]

# Polite delay (seconds) between consecutive tag requests to avoid 429s.
REQUEST_DELAY_SECONDS: float = 2.0

# Salary boundaries used for outlier removal (USD, annual).
SALARY_MIN_THRESHOLD: float = 1_000.0      # Below this → fake / placeholder
SALARY_MAX_THRESHOLD: float = 5_000_000.0  # Above this → clearly erroneous


# ---------------------------------------------------------------------------
# 1. Data ingestion
# ---------------------------------------------------------------------------

def fetch_remote_jobs() -> pd.DataFrame:
    """Aggregate remote IT job postings across multiple technology tags.

    Iterates over :data:`TECH_TAGS`, issuing one GET request per tag to
    ``/api?tags=<tag>``.  A :data:`REQUEST_DELAY_SECONDS` sleep is inserted
    between requests to respect the server's rate limits.  Tags that return
    HTTP errors or malformed payloads are skipped with a warning so a single
    failed tag cannot abort the entire crawl.

    After all tags have been queried, the collected records are merged into
    one list and deduplicated on ``(company, position)`` to eliminate jobs
    that appeared under multiple tags.

    Returns
    -------
    pd.DataFrame
        Raw (uncleaned) DataFrame with one row per unique job posting and the
        following columns:

        * ``company``    – Hiring company name.
        * ``position``   – Job title / role.
        * ``tags``       – Comma-separated technology / skill tags.
        * ``location``   – Advertised location string (often "Worldwide").
        * ``salary_min`` – Lower bound of advertised salary range (float).
        * ``salary_max`` – Upper bound of advertised salary range (float).

    Raises
    ------
    ValueError
        If no records were collected across *all* tag queries (total failure).
    """
    all_records: list[dict[str, Any]] = []

    for index, tag in enumerate(TECH_TAGS):
        tag_records = _fetch_tag_jobs(tag)
        all_records.extend(tag_records)
        logger.info(
            "Tag '%s' (%d/%d): %d postings collected — running total: %d.",
            tag,
            index + 1,
            len(TECH_TAGS),
            len(tag_records),
            len(all_records),
        )

        # Pause between requests — skip the delay after the final tag.
        if index < len(TECH_TAGS) - 1:
            logger.debug(
                "Sleeping %.1fs before next request.", REQUEST_DELAY_SECONDS
            )
            time.sleep(REQUEST_DELAY_SECONDS)

    if not all_records:
        raise ValueError(
            "No job records were collected across all tag queries. "
            "Check network connectivity and RemoteOK API availability."
        )

    df = pd.DataFrame(all_records)

    # Ensure salary columns are always numeric even when all values are NaN.
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")

    # Remove jobs that matched multiple tags — keep first occurrence.
    pre_dedup_count: int = len(df)
    df = (
        df
        .drop_duplicates(subset=["company", "position"])
        .reset_index(drop=True)
    )
    logger.info(
        "Deduplication on (company, position): %d → %d rows "
        "(removed %d duplicates).",
        pre_dedup_count,
        len(df),
        pre_dedup_count - len(df),
    )

    logger.info(
        "Final raw DataFrame: %d rows × %d columns.", df.shape[0], df.shape[1]
    )
    return df


# ---------------------------------------------------------------------------
# 1a. Private ingestion helpers
# ---------------------------------------------------------------------------

def _fetch_tag_jobs(tag: str) -> list[dict[str, Any]]:
    """Fetch and parse job postings for a single RemoteOK tag.

    Performs one HTTP GET to ``/api?tags=<tag>`` and delegates JSON parsing
    to :func:`_parse_job_entries`.  Any network or HTTP error is caught,
    logged as a warning, and an empty list is returned — allowing the caller
    to continue with remaining tags.

    Parameters
    ----------
    tag : str
        Technology tag to query (e.g. ``"python"``, ``"devops"``).

    Returns
    -------
    list[dict[str, Any]]
        Parsed job records for this tag, or an empty list on failure.
    """
    url = f"{REMOTEOK_API_URL}?tags={tag}"
    logger.info("GET %s", url)

    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning(
            "HTTP error for tag '%s' (%s) — skipping tag.", tag, exc
        )
        return []
    except requests.RequestException as exc:
        logger.warning(
            "Network error for tag '%s' (%s) — skipping tag.", tag, exc
        )
        return []

    raw: Any = response.json()

    if not isinstance(raw, list) or len(raw) == 0:
        logger.warning(
            "Unexpected response format for tag '%s' "
            "(got %s, expected non-empty list) — skipping tag.",
            tag,
            type(raw).__name__,
        )
        return []

    # The first element is a metadata / legal notice dict (not a job).
    # Guard: only skip it when it truly lacks the 'company' field so the
    # logic stays correct if RemoteOK ever normalises this behaviour.
    job_entries: list[dict[str, Any]] = (
        raw[1:] if "company" not in raw[0] else raw
    )

    return _parse_job_entries(job_entries)


def _parse_job_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise a list of raw job-posting dicts into flat record dicts.

    Extracts the six target fields from each entry, coerces salary values to
    float via :func:`_to_float`, and joins the ``tags`` list into a single
    comma-separated string.  Non-dict items are silently skipped.

    Parameters
    ----------
    entries : list[dict[str, Any]]
        Raw job-posting dicts as returned by the RemoteOK API (metadata
        element already removed).

    Returns
    -------
    list[dict[str, Any]]
        List of flat record dicts ready for ``pd.DataFrame(records)``.
    """
    records: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue  # Skip unexpected non-dict items defensively.

        # Tags arrive as a list; join to a comma-separated string for storage.
        raw_tags: list[str] | None = entry.get("tags")
        tags_str: str = (
            ", ".join(raw_tags) if isinstance(raw_tags, list) else ""
        )

        records.append(
            {
                "company": entry.get("company", ""),
                "position": entry.get("position", ""),
                "tags": tags_str,
                "location": entry.get("location", ""),
                "salary_min": _to_float(entry.get("salary_min")),
                "salary_max": _to_float(entry.get("salary_max")),
            }
        )

    return records


# ---------------------------------------------------------------------------
# 2. Cleaning & analysis
# ---------------------------------------------------------------------------

def clean_and_analyze_salaries(df: pd.DataFrame) -> pd.DataFrame:
    """Clean salary data and derive a vectorised average salary column.

    Steps applied (in order):

    1. **Drop salary-less rows** – remove any row where *both* ``salary_min``
       and ``salary_max`` are NaN, empty, or zero (i.e. no usable salary
       information whatsoever).
    2. **Compute ``salary_avg``** – element-wise mean of ``salary_min`` and
       ``salary_max`` using NumPy/Pandas vectorised operations.  When only one
       bound is present the available value is used as the estimate.
    3. **Remove outliers** – discard rows whose ``salary_avg`` falls outside
       the plausible annual USD range
       [``SALARY_MIN_THRESHOLD``, ``SALARY_MAX_THRESHOLD``].

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by :func:`fetch_remote_jobs`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the additional ``salary_avg`` column,
        reset integer index, and all salary values guaranteed to be
        finite floats within the plausible range.

    Raises
    ------
    ValueError
        If *df* is missing one or both of the required salary columns.
    """
    required_cols = {"salary_min", "salary_max"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required column(s): {missing}"
        )

    logger.info("Starting salary cleaning. Input rows: %d", len(df))

    # Work on an explicit copy to avoid mutating the caller's DataFrame.
    clean: pd.DataFrame = df.copy()

    # --- Step 1: coerce to numeric (already done in fetch, but be defensive)
    clean["salary_min"] = pd.to_numeric(clean["salary_min"], errors="coerce")
    clean["salary_max"] = pd.to_numeric(clean["salary_max"], errors="coerce")

    # Replace explicit zeros with NaN so they count as "missing".
    clean["salary_min"] = clean["salary_min"].replace(0, np.nan)
    clean["salary_max"] = clean["salary_max"].replace(0, np.nan)

    # --- Step 2: drop rows where BOTH salary bounds are absent / zero.
    both_missing_mask: pd.Series = (
        clean["salary_min"].isna() & clean["salary_max"].isna()
    )
    clean = clean[~both_missing_mask].copy()
    logger.info(
        "Rows after dropping salary-less entries: %d "
        "(removed %d rows).",
        len(clean),
        both_missing_mask.sum(),
    )

    # --- Step 3: vectorised salary_avg calculation.
    #   np.nanmean-equivalent via stacking → handles single-sided ranges.
    salary_matrix: np.ndarray = np.column_stack(
        [
            clean["salary_min"].to_numpy(dtype=float, na_value=np.nan),
            clean["salary_max"].to_numpy(dtype=float, na_value=np.nan),
        ]
    )
    clean["salary_avg"] = np.nanmean(salary_matrix, axis=1)

    # --- Step 4: outlier removal based on salary_avg.
    pre_outlier_count: int = len(clean)
    valid_range_mask: pd.Series = clean["salary_avg"].between(
        SALARY_MIN_THRESHOLD, SALARY_MAX_THRESHOLD, inclusive="both"
    )
    clean = clean[valid_range_mask].reset_index(drop=True)

    removed_outliers: int = pre_outlier_count - len(clean)
    logger.info(
        "Rows after outlier removal: %d (removed %d outlier rows).",
        len(clean),
        removed_outliers,
    )

    return clean


# ---------------------------------------------------------------------------
# 3. Private helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    """Safely coerce an arbitrary API value to float or None.

    Parameters
    ----------
    value : Any
        Raw value from the JSON payload (may be str, int, float, None, …).

    Returns
    -------
    float | None
        Parsed float, or ``None`` when conversion is not possible.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 4. Smoke-test entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Remote IT Job Market — Backend Smoke Test")
    print(f"  Querying {len(TECH_TAGS)} tags: {', '.join(TECH_TAGS)}")
    print("=" * 60)

    # -- Fetch --
    try:
        raw_df = fetch_remote_jobs()
    except ValueError as exc:
        logger.error("Fatal fetch error: %s", exc)
        sys.exit(1)

    print(f"\n[1] Raw aggregated fetch — shape: {raw_df.shape}")
    print(raw_df[["company", "position", "tags", "salary_min",
                   "salary_max"]].head(5).to_string(index=False))

    salary_coverage = raw_df[["salary_min", "salary_max"]].notna().any(axis=1)
    print(
        f"\n    Jobs with salary data: "
        f"{salary_coverage.sum()} / {len(raw_df)} "
        f"({salary_coverage.mean():.1%})"
    )

    # -- Clean & analyse --
    try:
        clean_df = clean_and_analyze_salaries(raw_df)
    except ValueError as exc:
        logger.error("Cleaning failed: %s", exc)
        sys.exit(1)

    print(f"\n[2] After cleaning — shape: {clean_df.shape}")
    print(clean_df[["company", "position", "salary_min",
                     "salary_max", "salary_avg"]].head(5).to_string(index=False))

    # -- Descriptive statistics --
    salary_cols = ["salary_min", "salary_max", "salary_avg"]
    print(f"\n[3] Salary descriptive statistics (n={len(clean_df)}):")
    print(clean_df[salary_cols].describe().round(2).to_string())
    print("\nDone. ✓")


if __name__ == "__main__":
    main()