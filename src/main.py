"""
src/main.py
-----------
Core backend module for the Global Remote IT Job Market Analysis project.

Fetches live job postings from the RemoteOK public API, normalises the raw
JSON payload into a tidy Pandas DataFrame, and applies salary-focused
cleaning / outlier removal so the data is ready for downstream plotting or
statistical analysis.

Usage (standalone smoke-test):
    python src/main.py
"""

from __future__ import annotations

import logging
import sys
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

# Public RemoteOK API endpoint (no auth required).
REMOTEOK_API_URL: str = "https://remoteok.com/api"

# Salary boundaries used for outlier removal (USD, annual).
SALARY_MIN_THRESHOLD: float = 1_000.0      # Below this → fake / placeholder
SALARY_MAX_THRESHOLD: float = 5_000_000.0  # Above this → clearly erroneous


# ---------------------------------------------------------------------------
# 1. Data ingestion
# ---------------------------------------------------------------------------

def fetch_remote_jobs() -> pd.DataFrame:
    """Fetch current remote IT job postings from the RemoteOK public API.

    The API returns a JSON array whose **first element is always a metadata /
    legal notice object** rather than a real job posting.  That element is
    detected and stripped before the DataFrame is constructed.

    Returns
    -------
    pd.DataFrame
        Raw (uncleaned) DataFrame with one row per job posting and the
        following columns:

        * ``company``    – Hiring company name.
        * ``position``   – Job title / role.
        * ``tags``       – Comma-separated technology / skill tags.
        * ``location``   – Advertised location string (often "Worldwide").
        * ``salary_min`` – Lower bound of advertised salary range (float).
        * ``salary_max`` – Upper bound of advertised salary range (float).

    Raises
    ------
    requests.HTTPError
        If the RemoteOK server returns a non-2xx HTTP status code.
    requests.ConnectionError
        If the network request cannot be completed.
    ValueError
        If the API response is not a non-empty JSON list.
    """
    logger.info("Sending GET request to %s", REMOTEOK_API_URL)

    response = requests.get(
        REMOTEOK_API_URL,
        headers=_REQUEST_HEADERS,
        timeout=30,
    )

    # Surface HTTP errors (403, 429, 5xx, …) as Python exceptions.
    response.raise_for_status()

    raw: Any = response.json()

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(
            f"Unexpected API response format – expected a non-empty list, "
            f"got {type(raw).__name__}."
        )

    # The first element is a metadata / legal notice dict (not a job).
    # Guard: only skip it when it truly lacks a typical job field such as
    # 'company', so the code remains correct if RemoteOK ever fixes this.
    job_entries: list[dict[str, Any]] = (
        raw[1:] if "company" not in raw[0] else raw
    )

    logger.info("Received %d job postings from API.", len(job_entries))

    records: list[dict[str, Any]] = []
    for entry in job_entries:
        if not isinstance(entry, dict):
            continue  # Skip any unexpected non-dict items defensively.

        # Tags arrive as a list; join into a single comma-separated string.
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

    df = pd.DataFrame(records)

    # Ensure salary columns are always numeric even when all values are NaN.
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")

    logger.info(
        "DataFrame built: %d rows × %d columns.", df.shape[0], df.shape[1]
    )
    return df


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
        Raw value from the JSON payload (may take values of str, int, float, None, …).

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

if __name__ == "__main__":
    print("=" * 60)
    print("  Remote IT Job Market — Backend Smoke Test")
    print("=" * 60)

    # -- Fetch --
    try:
        raw_df = fetch_remote_jobs()
    except requests.HTTPError as exc:
        logger.error("HTTP error while fetching jobs: %s", exc)
        sys.exit(1)
    except requests.ConnectionError as exc:
        logger.error("Network error – check your internet connection: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Unexpected API response: %s", exc)
        sys.exit(1)

    print(f"\n[1] Raw fetch — shape: {raw_df.shape}")
    print(raw_df.head(3).to_string(index=False))

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