"""
src/main.py
-----------
Core backend module for the Polish IT Job Market Analysis project.

Aggregates Polish IT job postings from the Adzuna Jobs API across a curated
list of technology search terms (e.g. "Python", "Java", "Data") to build a
statistically significant dataset.  Each term is queried individually against
the Polish market endpoint (``/jobs/pl/search/1``), results are merged and
deduplicated on ``(company, position)`` to eliminate cross-term overlaps.

Credentials are read securely from environment variables via ``python-dotenv``
(``ADZUNA_APP_ID`` and ``ADZUNA_APP_KEY``).  Salary thresholds are calibrated
for annual PLN figures.

Pipeline
--------
0. ``_load_credentials()``           – dotenv → (app_id, app_key) or ValueError
1. ``fetch_remote_jobs()``           – multi-term ingestion → deduplicated DataFrame
2. ``clean_and_analyze_salaries()``  – salary cleaning, outlier removal, avg column

Usage (standalone smoke-test):
    python src/main.py
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

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Adzuna Polish-market search endpoint (page 1; we collect all terms in one
# pass rather than paginating, which keeps the crawl lightweight for the
# academic use-case).
ADZUNA_BASE_URL: str = "https://api.adzuna.com/v1/api/jobs/pl/search/1"

# Technology search terms used to query the API individually.
# Each term maps to GET /jobs/pl/search/1?what=<term>&results_per_page=50.
# Broaden or narrow this list to control dataset size vs. crawl time.
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

# Adzuna's documented maximum is 50 results per page.
RESULTS_PER_PAGE: int = 50

# Polite delay (seconds) between consecutive term requests to avoid 429s.
REQUEST_DELAY_SECONDS: float = 1.0

# Salary outlier thresholds calibrated for annual gross PLN figures.
#   30 000 PLN/yr  ≈ minimum wage territory (filters placeholders / test data)
#   1 000 000 PLN/yr ≈ hard ceiling for realistic IT compensation in Poland
SALARY_MIN_THRESHOLD: float = 30_000.0
SALARY_MAX_THRESHOLD: float = 1_000_000.0


# ---------------------------------------------------------------------------
# 0. Credential management
# ---------------------------------------------------------------------------

def _load_credentials() -> tuple[str, str]:
    """Load Adzuna API credentials from the environment via python-dotenv.

    Reads the ``.env`` file in the current working directory (if present) and
    exposes its contents as environment variables before attempting to read
    ``ADZUNA_APP_ID`` and ``ADZUNA_APP_KEY``.

    Returns
    -------
    tuple[str, str]
        A ``(app_id, app_key)`` pair, both guaranteed to be non-empty strings.

    Raises
    ------
    ValueError
        If either variable is absent or empty.
    """
    load_dotenv()

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        missing = []
        if not app_id:
            missing.append("ADZUNA_APP_ID")
        if not app_key:
            missing.append("ADZUNA_APP_KEY")

        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "Create a .env file in the project root with the following content:\n\n"
            "    ADZUNA_APP_ID=your_app_id_here\n"
            "    ADZUNA_APP_KEY=your_app_key_here\n\n"
            "Obtain credentials at https://developer.adzuna.com/"
        )

    # PyCharm is now 100% sure these are strings, no 'ignore' needed
    return app_id, app_key


# ---------------------------------------------------------------------------
# 1. Data ingestion
# ---------------------------------------------------------------------------

def fetch_remote_jobs() -> pd.DataFrame:
    """Aggregate Polish IT job postings across multiple technology search terms.

    Loads Adzuna credentials via :func:`_load_credentials`, then iterates over
    :data:`SEARCH_TERMS`, issuing one GET request per term to the Adzuna Polish
    market endpoint.  A :data:`REQUEST_DELAY_SECONDS` sleep is inserted between
    requests to respect rate limits.  Terms that return HTTP errors or malformed
    payloads are skipped with a warning so a single failed term cannot abort the
    entire crawl.

    After all terms have been queried, collected records are merged and
    deduplicated on ``(company, position)`` to eliminate jobs that matched
    multiple search terms.

    Returns
    -------
    pd.DataFrame
        Raw (uncleaned) DataFrame with one row per unique job posting and the
        following columns:

        * ``company``    – Hiring company display name.
        * ``position``   – Job title / role.
        * ``location``   – City or region display name.
        * ``salary_min`` – Lower bound of advertised salary range (float, PLN).
        * ``salary_max`` – Upper bound of advertised salary range (float, PLN).

    Raises
    ------
    ValueError
        If credentials are missing (propagated from :func:`_load_credentials`)
        or if no records were collected across all term queries (total failure).
    """
    app_id, app_key = _load_credentials()

    all_records: list[dict[str, Any]] = []

    for index, term in enumerate(SEARCH_TERMS):
        term_records = _fetch_term_jobs(term, app_id, app_key)
        all_records.extend(term_records)
        logger.info(
            "Term '%s' (%d/%d): %d postings collected — running total: %d.",
            term,
            index + 1,
            len(SEARCH_TERMS),
            len(term_records),
            len(all_records),
        )

        # Pause between requests — skip the delay after the final term.
        if index < len(SEARCH_TERMS) - 1:
            logger.debug(
                "Sleeping %.1fs before next request.", REQUEST_DELAY_SECONDS
            )
            time.sleep(REQUEST_DELAY_SECONDS)

    if not all_records:
        raise ValueError(
            "No job records were collected across all search term queries.\n"
            "Possible causes: network outage, invalid credentials, or the "
            "Adzuna API returned no results for any of the configured terms.\n"
            f"Configured terms: {SEARCH_TERMS}"
        )

    df = pd.DataFrame(all_records)

    # Ensure salary columns are always numeric even when all values are NaN.
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")

    # Remove jobs that matched multiple search terms — keep first occurrence.
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

def _fetch_term_jobs(
    term: str,
    app_id: str,
    app_key: str,
) -> list[dict[str, Any]]:
    """Fetch and parse job postings for a single Adzuna search term.

    Performs one authenticated HTTP GET to the Adzuna Polish market endpoint
    with the provided search term and delegates JSON normalisation to
    :func:`_parse_job_entries`.  Any network or HTTP error is caught, logged
    as a warning, and an empty list is returned — allowing the caller to
    continue with remaining terms.

    Parameters
    ----------
    term : str
        Free-text technology search term (e.g. ``"Python"``, ``"DevOps"``).
        Passed as the ``what`` query parameter.
    app_id : str
        Adzuna application ID (from ``ADZUNA_APP_ID`` env var).
    app_key : str
        Adzuna application key (from ``ADZUNA_APP_KEY`` env var).

    Returns
    -------
    list[dict[str, Any]]
        Parsed and normalised job records for this term, or an empty list on
        any failure.
    """
    params: dict[str, Any] = {
        "app_id": app_id,
        "app_key": app_key,
        "what": term,
        "results_per_page": RESULTS_PER_PAGE,
        "content-type": "application/json",
    }

    logger.info("GET %s  [what=%s]", ADZUNA_BASE_URL, term)

    try:
        response = requests.get(
            ADZUNA_BASE_URL,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning(
            "HTTP error for term '%s' (%s) — skipping term.", term, exc
        )
        return []
    except requests.RequestException as exc:
        logger.warning(
            "Network error for term '%s' (%s) — skipping term.", term, exc
        )
        return []

    payload: Any = response.json()

    # Adzuna wraps results in a top-level dict: {"results": [...], "count": N}
    if not isinstance(payload, dict) or "results" not in payload:
        logger.warning(
            "Unexpected response format for term '%s' "
            "(expected dict with 'results' key, got %s) — skipping term.",
            term,
            type(payload).__name__,
        )
        return []

    raw_results: Any = payload["results"]

    if not isinstance(raw_results, list):
        logger.warning(
            "Field 'results' for term '%s' is not a list (got %s) "
            "— skipping term.",
            term,
            type(raw_results).__name__,
        )
        return []

    logger.debug("Term '%s': API returned %d raw result(s).", term, len(raw_results))
    return _parse_job_entries(raw_results)


def _parse_job_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise a list of raw Adzuna job-posting dicts into flat record dicts.

    Extracts the five target fields from each entry.  Nested objects
    (``company``, ``location``) are safely traversed with chained ``.get()``
    calls so that a missing sub-key never raises a ``KeyError``.  Salary
    values are coerced to float via :func:`_to_float`.  Non-dict items are
    silently skipped.

    Parameters
    ----------
    entries : list[dict[str, Any]]
        Raw job-posting dicts as returned under the ``"results"`` key of an
        Adzuna API response.

    Returns
    -------
    list[dict[str, Any]]
        List of flat record dicts ready for ``pd.DataFrame(records)``.

    Notes
    -----
    Adzuna nested field mapping:

    +--------------------------+------------------+
    | API path                 | DataFrame column |
    +==========================+==================+
    | ``company.display_name`` | ``company``      |
    | ``title``                | ``position``     |
    | ``location.display_name``| ``location``     |
    | ``salary_min``           | ``salary_min``   |
    | ``salary_max``           | ``salary_max``   |
    +--------------------------+------------------+
    """
    records: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue  # Skip unexpected non-dict items defensively.

        # Safely navigate nested objects; default to empty string if absent.
        company: str = (
            entry.get("company") or {}
        ).get("display_name", "")

        location: str = (
            entry.get("location") or {}
        ).get("display_name", "")

        records.append(
            {
                "company": company,
                "position": entry.get("title", ""),
                "location": location,
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
       the plausible annual PLN range
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
        finite floats within the plausible PLN range.

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
    """Run an end-to-end smoke test: fetch → clean → describe.

    Intended for direct script execution only (``python src/main.py``).
    Exits with code 1 on any unrecoverable error so the failure is visible
    in CI pipelines and shell scripts.
    """
    print("=" * 62)
    print("  Polish IT Job Market — Adzuna Backend Smoke Test")
    print(f"  Querying {len(SEARCH_TERMS)} terms: {', '.join(SEARCH_TERMS)}")
    print("=" * 62)

    # -- Fetch --
    try:
        raw_df = fetch_remote_jobs()
    except ValueError as exc:
        logger.error("Fatal fetch error:\n%s", exc)
        sys.exit(1)

    print(f"\n[1] Raw aggregated fetch — shape: {raw_df.shape}")
    print(
        raw_df[["company", "position", "location", "salary_min", "salary_max"]]
        .head(5)
        .to_string(index=False)
    )

    salary_coverage: pd.Series = (
        raw_df[["salary_min", "salary_max"]].notna().any(axis=1)
    )
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
    print(
        clean_df[["company", "position", "salary_min", "salary_max", "salary_avg"]]
        .head(5)
        .to_string(index=False)
    )

    # -- Descriptive statistics --
    salary_cols: list[str] = ["salary_min", "salary_max", "salary_avg"]
    print(f"\n[3] Salary descriptive statistics  [PLN, annual]  (n={len(clean_df)}):")
    print(clean_df[salary_cols].describe().round(2).to_string())
    print("\nDone. ✓")


if __name__ == "__main__":
    main()