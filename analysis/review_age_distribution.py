from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    mannwhitneyu,
    spearmanr,
)


# ============================================================
# CONFIG
# ============================================================

RELEVANT_DIR = Path("./relevant_reviews")
NEWEST_DIR = Path("./newest_reviews")

OUTPUT_DIR = Path("./review_age_analysis")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_EXCEL = (
    OUTPUT_DIR /
    "review_age_relevant_vs_newest.xlsx"
)


# ============================================================
# BALANCED BENCHMARK
# ============================================================

BENCHMARK_N = 1700


TOP_K_VALUES = [
    50,
    100,
    250,
    500,
    1000,
    1700,
]


# ============================================================
# COLUMN NAMES
# ============================================================

REVIEW_DATE_CANDIDATES = [
    "review_date_iso",
    "review_date",
]

COLLECTION_DATE_CANDIDATES = [
    "collected_at_utc",
    "collected_at",
]

ORDER_COL = "scrape_order"


# ============================================================
# FALLBACK COLLECTION DATES
#
# Used ONLY when collected_at_utc is completely missing.
# ============================================================

FALLBACK_COLLECTION_DATES = {

    "Newest_instagram_google_play_reviews.xlsx":
        "2026-08-23 23:00:00+00:00"
}


# ============================================================
# EXCEL-SAFE HELPER
# ============================================================

def make_excel_safe(df):
    """
    Remove timezone information from any datetime values
    before writing the DataFrame to Excel.
    """

    df = df.copy()

    for col in df.columns:

        # ----------------------------------------------------
        # Proper timezone-aware datetime column
        # ----------------------------------------------------

        if isinstance(
            df[col].dtype,
            pd.DatetimeTZDtype
        ):

            df[col] = (
                df[col]
                .dt
                .tz_localize(None)
            )

        # ----------------------------------------------------
        # Object column that may contain Timestamp/datetime
        # ----------------------------------------------------

        elif df[col].dtype == "object":

            def clean_value(value):

                if isinstance(
                    value,
                    pd.Timestamp
                ):

                    if value.tzinfo is not None:

                        return (
                            value
                            .tz_localize(None)
                        )

                    return value

                if isinstance(
                    value,
                    datetime
                ):

                    if value.tzinfo is not None:

                        return value.replace(
                            tzinfo=None
                        )

                    return value

                return value


            df[col] = (
                df[col]
                .apply(clean_value)
            )

    return df


# ============================================================
# PLATFORM NAME
# ============================================================

def get_platform_from_filename(path):

    name = (
        path
        .stem
        .lower()
        .strip()
    )

    # Remove Newest prefix
    name = re.sub(
        r"^newest[_\-\s]*",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove suffix
    name = re.sub(
        r"_google_play_reviews.*$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = name.strip(
        "_- "
    )

    mapping = {

        "facebook":
            "Facebook",

        "instagram":
            "Instagram",

        "snapchat":
            "Snapchat",

        "tiktok":
            "TikTok",

        "twitter":
            "X",

        "x":
            "X",

        "youtube":
            "YouTube",
    }

    return mapping.get(
        name,
        name.replace(
            "_",
            " "
        ).title()
    )


# ============================================================
# FIND FIRST AVAILABLE COLUMN
# ============================================================

def find_column(
    df,
    candidates,
    description
):

    for column in candidates:

        if column in df.columns:

            return column

    raise ValueError(

        f"Could not find {description}. "
        f"Tried: {candidates}"
    )


# ============================================================
# DATETIME PARSER
#
# Everything is converted to UTC and then timezone information
# is removed. Therefore ALL dates inside the analysis are
# timezone-naive UTC timestamps.
# ============================================================

def parse_datetime_series(series):

    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True
    )

    return (
        parsed
        .dt
        .tz_localize(None)
    )


# ============================================================
# READ ONE REVIEW FILE
# ============================================================

def read_review_file(path):

    print(
        f"\nReading: {path.name}"
    )

    df = pd.read_excel(
        path
    )

    raw_rows = len(df)


    # ========================================================
    # FIND REVIEW DATE COLUMN
    # ========================================================

    review_date_col = (
        find_column(
            df,
            REVIEW_DATE_CANDIDATES,
            "review date column"
        )
    )


    # ========================================================
    # FIND COLLECTION DATE COLUMN
    # ========================================================

    collection_date_col = (
        find_column(
            df,
            COLLECTION_DATE_CANDIDATES,
            "collection date column"
        )
    )


    # ========================================================
    # PARSE REVIEW DATE
    # ========================================================

    df[
        "review_datetime"
    ] = parse_datetime_series(
        df[
            review_date_col
        ]
    )


    # ========================================================
    # PARSE COLLECTION DATE
    # ========================================================

    df[
        "collection_datetime"
    ] = parse_datetime_series(
        df[
            collection_date_col
        ]
    )


    fallback_used = False
    fallback_value_used = None


    # ========================================================
    # FALLBACK COLLECTION DATE
    #
    # For example:
    # Instagram Newest has valid review_date_iso values but
    # collected_at_utc is NaN for every row.
    # ========================================================

    if (
        df[
            "collection_datetime"
        ]
        .isna()
        .all()
    ):

        fallback_value = (
            FALLBACK_COLLECTION_DATES
            .get(
                path.name
            )
        )


        if fallback_value is None:

            raise ValueError(

                f"{path.name}: "
                f"'{collection_date_col}' contains "
                f"no valid collection timestamps and "
                f"no fallback date is configured."
            )


        # ----------------------------------------------------
        # Parse fallback as UTC
        # ----------------------------------------------------

        fallback_timestamp = (
            pd.to_datetime(
                fallback_value,
                utc=True
            )
        )


        # ----------------------------------------------------
        # Convert scalar Timestamp to timezone-naive UTC
        #
        # This makes it compatible with review_datetime.
        # ----------------------------------------------------

        fallback_timestamp = (
            fallback_timestamp
            .tz_localize(None)
        )


        # ----------------------------------------------------
        # Fill every row
        # ----------------------------------------------------

        df[
            "collection_datetime"
        ] = fallback_timestamp


        fallback_used = True
        fallback_value_used = (
            fallback_timestamp
        )


        print(
            f"  INFO: collection timestamp "
            f"missing for all rows."
        )

        print(
            f"  Using fallback collection date: "
            f"{fallback_timestamp}"
        )


    # ========================================================
    # FILL PARTIALLY MISSING COLLECTION DATES
    #
    # If a future file has only some collection timestamps
    # missing, use a configured fallback where available.
    # ========================================================

    elif (
        df[
            "collection_datetime"
        ]
        .isna()
        .any()
    ):

        fallback_value = (
            FALLBACK_COLLECTION_DATES
            .get(
                path.name
            )
        )

        if fallback_value is not None:

            fallback_timestamp = (
                pd.to_datetime(
                    fallback_value,
                    utc=True
                )
                .tz_localize(None)
            )

            missing_count = int(
                df[
                    "collection_datetime"
                ]
                .isna()
                .sum()
            )

            df[
                "collection_datetime"
            ] = (
                df[
                    "collection_datetime"
                ]
                .fillna(
                    fallback_timestamp
                )
            )

            fallback_used = True
            fallback_value_used = (
                fallback_timestamp
            )

            print(
                f"  INFO: filled "
                f"{missing_count:,} missing "
                f"collection timestamps "
                f"using fallback."
            )


    # ========================================================
    # CALCULATE REVIEW AGE
    #
    # Age = collection datetime - review datetime
    # ========================================================

    age_timedelta = (

        df[
            "collection_datetime"
        ]

        -

        df[
            "review_datetime"
        ]
    )


    df[
        "review_age_days"
    ] = (

        age_timedelta
        .dt
        .total_seconds()

        /

        86400
    )


    # ========================================================
    # REMOVE IMPOSSIBLE NEGATIVE AGES
    # ========================================================

    impossible_age = (
        df[
            "review_age_days"
        ] < 0
    )


    impossible_count = int(
        impossible_age.sum()
    )


    if impossible_count > 0:

        print(
            f"  WARNING: "
            f"{impossible_count:,} reviews "
            f"have negative calculated age."
        )


    df.loc[
        impossible_age,
        "review_age_days"
    ] = np.nan


    # ========================================================
    # SCRAPE ORDER
    # ========================================================

    if ORDER_COL in df.columns:

        df[
            ORDER_COL
        ] = pd.to_numeric(
            df[
                ORDER_COL
            ],
            errors="coerce"
        )


        df = df.sort_values(
            ORDER_COL,
            na_position="last"
        )


    else:

        print(
            f"  WARNING: "
            f"{path.name} has no "
            f"'{ORDER_COL}'. "
            f"Using row order."
        )


        df[
            ORDER_COL
        ] = (

            np.arange(
                len(df)
            )

            +

            1
        )


    # ========================================================
    # KEEP VALID AGE ROWS
    # ========================================================

    valid_df = df[
        df[
            "review_age_days"
        ].notna()
    ].copy()


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    diagnostics = {

        "file":
            path.name,

        "raw_rows":
            raw_rows,

        "review_date_column":
            review_date_col,

        "collection_date_column":
            collection_date_col,

        "fallback_used":
            fallback_used,

        "fallback_collection_date":
            fallback_value_used,

        "valid_age_rows":
            len(valid_df),

        "invalid_or_missing_age_rows":
            (
                raw_rows
                -
                len(valid_df)
            ),

        "negative_age_rows":
            impossible_count,

        "earliest_review":
            valid_df[
                "review_datetime"
            ].min(),

        "latest_review":
            valid_df[
                "review_datetime"
            ].max(),

        "earliest_collection":
            valid_df[
                "collection_datetime"
            ].min(),

        "latest_collection":
            valid_df[
                "collection_datetime"
            ].max(),
    }


    print(
        f"  Raw rows       : "
        f"{raw_rows:,}"
    )

    print(
        f"  Valid age rows : "
        f"{len(valid_df):,}"
    )


    return (

        valid_df.reset_index(
            drop=True
        ),

        diagnostics
    )


# ============================================================
# AGE SUMMARY
# ============================================================

def summarize_age(
    platform,
    ordering,
    df
):

    age = (
        df[
            "review_age_days"
        ]
        .dropna()
    )


    n = len(age)


    if n == 0:

        return {

            "platform":
                platform,

            "ordering":
                ordering,

            "n":
                0,
        }


    return {

        "platform":
            platform,

        "ordering":
            ordering,

        "n":
            n,


        # ====================================================
        # CENTRAL TENDENCY
        # ====================================================

        "mean_age_days":
            age.mean(),

        "median_age_days":
            age.median(),


        # ====================================================
        # QUANTILES
        # ====================================================

        "p25_age_days":
            age.quantile(
                0.25
            ),

        "p75_age_days":
            age.quantile(
                0.75
            ),

        "p90_age_days":
            age.quantile(
                0.90
            ),

        "max_age_days":
            age.max(),


        # ====================================================
        # <= 7 DAYS
        # ====================================================

        "within_7_days_n":
            int(
                (
                    age <= 7
                ).sum()
            ),

        "within_7_days_percent":
            (
                age <= 7
            ).mean() * 100,


        # ====================================================
        # <= 30 DAYS
        # ====================================================

        "within_30_days_n":
            int(
                (
                    age <= 30
                ).sum()
            ),

        "within_30_days_percent":
            (
                age <= 30
            ).mean() * 100,


        # ====================================================
        # <= 90 DAYS
        # ====================================================

        "within_90_days_n":
            int(
                (
                    age <= 90
                ).sum()
            ),

        "within_90_days_percent":
            (
                age <= 90
            ).mean() * 100,


        # ====================================================
        # <= 180 DAYS
        # ====================================================

        "within_180_days_n":
            int(
                (
                    age <= 180
                ).sum()
            ),

        "within_180_days_percent":
            (
                age <= 180
            ).mean() * 100,


        # ====================================================
        # OLDER
        # ====================================================

        "older_than_180_days_percent":
            (
                age > 180
            ).mean() * 100,

        "older_than_365_days_percent":
            (
                age > 365
            ).mean() * 100,
    }


# ============================================================
# SAFE MANN-WHITNEY TEST
# ============================================================

def safe_mannwhitney(
    a,
    b
):

    a = (
        pd.to_numeric(
            a,
            errors="coerce"
        )
        .dropna()
    )


    b = (
        pd.to_numeric(
            b,
            errors="coerce"
        )
        .dropna()
    )


    if (
        len(a) < 2
        or
        len(b) < 2
    ):

        return (
            np.nan,
            np.nan
        )


    result = mannwhitneyu(
        a,
        b,
        alternative="two-sided"
    )


    return (
        result.statistic,
        result.pvalue
    )


# ============================================================
# SAFE SPEARMAN CORRELATION
# ============================================================

def safe_spearman(
    x,
    y
):

    temp = pd.DataFrame({

        "x":
            x,

        "y":
            y,

    }).dropna()


    if len(temp) < 3:

        return (
            np.nan,
            np.nan
        )


    if (
        temp["x"].nunique() < 2
        or
        temp["y"].nunique() < 2
    ):

        return (
            np.nan,
            np.nan
        )


    result = spearmanr(
        temp["x"],
        temp["y"]
    )


    return (
        result.statistic,
        result.pvalue
    )


# ============================================================
# DISCOVER FILES
# ============================================================

relevant_files = {

    get_platform_from_filename(
        path
    ):
        path

    for path in (
        RELEVANT_DIR.glob(
            "*.xlsx"
        )
    )
}


newest_files = {

    get_platform_from_filename(
        path
    ):
        path

    for path in (
        NEWEST_DIR.glob(
            "*.xlsx"
        )
    )
}


platforms = sorted(

    set(
        relevant_files.keys()
    )

    &

    set(
        newest_files.keys()
    )
)


# ============================================================
# HEADER
# ============================================================

print(
    "=" * 120
)

print(
    "REVIEW AGE DISTRIBUTION: "
    "RELEVANT VS NEWEST"
)

print(
    "=" * 120
)


print(
    f"\nBalanced benchmark: "
    f"{BENCHMARK_N:,} reviews "
    f"per ordering per platform"
)


print(
    "\nMatched platforms:"
)


for platform in platforms:

    print(
        f"  {platform}"
    )


if not platforms:

    raise RuntimeError(
        "No matching platforms found."
    )


# ============================================================
# LOAD ALL DATASETS
# ============================================================

datasets = {}

diagnostic_rows = []


for platform in platforms:

    relevant, rel_diag = (
        read_review_file(
            relevant_files[
                platform
            ]
        )
    )


    newest, new_diag = (
        read_review_file(
            newest_files[
                platform
            ]
        )
    )


    datasets[
        platform
    ] = {

        "Relevant":
            relevant,

        "Newest":
            newest,
    }


    diagnostic_rows.append({

        "platform":
            platform,

        "ordering":
            "Relevant",

        **rel_diag
    })


    diagnostic_rows.append({

        "platform":
            platform,

        "ordering":
            "Newest",

        **new_diag
    })


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


# ============================================================
# ELIGIBILITY
# ============================================================

eligibility_rows = []

eligible_platforms = []


for platform in platforms:

    relevant_n = len(
        datasets[
            platform
        ]["Relevant"]
    )


    newest_n = len(
        datasets[
            platform
        ]["Newest"]
    )


    if (
        relevant_n >= BENCHMARK_N
        and
        newest_n >= BENCHMARK_N
    ):

        status = "ELIGIBLE"

        eligible_platforms.append(
            platform
        )

    else:

        status = "EXCLUDED"


    eligibility_rows.append({

        "platform":
            platform,

        "relevant_valid_age_n":
            relevant_n,

        "newest_valid_age_n":
            newest_n,

        "required_n":
            BENCHMARK_N,

        "status":
            status,
    })


eligibility_df = pd.DataFrame(
    eligibility_rows
)


# ============================================================
# PRINT DIAGNOSTICS
# ============================================================

print(
    "\n"
)

print(
    "=" * 120
)

print(
    "DATE DIAGNOSTICS"
)

print(
    "=" * 120
)


print(
    diagnostics_df.to_string(
        index=False
    )
)


print(
    "\n"
)

print(
    "=" * 120
)

print(
    "ELIGIBILITY"
)

print(
    "=" * 120
)


print(
    eligibility_df.to_string(
        index=False
    )
)


if not eligible_platforms:

    raise RuntimeError(

        "No platform has enough "
        "valid review-age observations."
    )


# ============================================================
# BALANCED 1700 ANALYSIS
# ============================================================

summary_rows = []

comparison_rows = []

balanced_frames = []


for platform in eligible_platforms:

    relevant = (

        datasets[
            platform
        ]["Relevant"]

        .head(
            BENCHMARK_N
        )

        .copy()
    )


    newest = (

        datasets[
            platform
        ]["Newest"]

        .head(
            BENCHMARK_N
        )

        .copy()
    )


    relevant[
        "platform"
    ] = platform


    relevant[
        "ordering"
    ] = "Relevant"


    newest[
        "platform"
    ] = platform


    newest[
        "ordering"
    ] = "Newest"


    balanced_frames.extend(
        [
            relevant,
            newest
        ]
    )


    rel_summary = summarize_age(
        platform,
        "Relevant",
        relevant
    )


    new_summary = summarize_age(
        platform,
        "Newest",
        newest
    )


    summary_rows.extend(
        [
            rel_summary,
            new_summary
        ]
    )


    # ========================================================
    # AGE COMPARISON
    # ========================================================

    rel_median = (
        rel_summary[
            "median_age_days"
        ]
    )


    new_median = (
        new_summary[
            "median_age_days"
        ]
    )


    median_age_gap = (
        rel_median
        -
        new_median
    )


    median_age_ratio = (

        rel_median
        /
        new_median

        if new_median > 0

        else np.nan
    )


    age_u, age_p = (
        safe_mannwhitney(

            relevant[
                "review_age_days"
            ],

            newest[
                "review_age_days"
            ]
        )
    )


    comparison_rows.append({

        "platform":
            platform,

        "n_each":
            BENCHMARK_N,


        # ====================================================
        # AGE
        # ====================================================

        "relevant_mean_age_days":
            rel_summary[
                "mean_age_days"
            ],

        "newest_mean_age_days":
            new_summary[
                "mean_age_days"
            ],

        "relevant_median_age_days":
            rel_median,

        "newest_median_age_days":
            new_median,

        "median_age_gap_days":
            median_age_gap,

        "median_age_ratio":
            median_age_ratio,


        # ====================================================
        # <= 7 DAYS
        # ====================================================

        "relevant_within_7_days_percent":
            rel_summary[
                "within_7_days_percent"
            ],

        "newest_within_7_days_percent":
            new_summary[
                "within_7_days_percent"
            ],


        # ====================================================
        # <= 30 DAYS
        # ====================================================

        "relevant_within_30_days_percent":
            rel_summary[
                "within_30_days_percent"
            ],

        "newest_within_30_days_percent":
            new_summary[
                "within_30_days_percent"
            ],


        # ====================================================
        # <= 90 DAYS
        # ====================================================

        "relevant_within_90_days_percent":
            rel_summary[
                "within_90_days_percent"
            ],

        "newest_within_90_days_percent":
            new_summary[
                "within_90_days_percent"
            ],


        # ====================================================
        # <= 180 DAYS
        # ====================================================

        "relevant_within_180_days_percent":
            rel_summary[
                "within_180_days_percent"
            ],

        "newest_within_180_days_percent":
            new_summary[
                "within_180_days_percent"
            ],


        # ====================================================
        # > 365 DAYS
        # ====================================================

        "relevant_over_365_days_percent":
            rel_summary[
                "older_than_365_days_percent"
            ],

        "newest_over_365_days_percent":
            new_summary[
                "older_than_365_days_percent"
            ],


        # ====================================================
        # STATISTICAL TEST
        # ====================================================

        "age_mannwhitney_u":
            age_u,

        "age_p_value":
            age_p,
    })


summary_df = pd.DataFrame(
    summary_rows
)


comparison_df = pd.DataFrame(
    comparison_rows
)


balanced_df = pd.concat(
    balanced_frames,
    ignore_index=True
)


# ============================================================
# PRINT PLATFORM AGE COMPARISON
# ============================================================

print(
    "\n"
)

print(
    "=" * 170
)

print(
    "BALANCED REVIEW-AGE COMPARISON"
)

print(
    "=" * 170
)


display_columns = [

    "platform",

    "relevant_median_age_days",

    "newest_median_age_days",

    "median_age_gap_days",

    "median_age_ratio",

    "relevant_within_7_days_percent",

    "newest_within_7_days_percent",

    "relevant_within_30_days_percent",

    "newest_within_30_days_percent",

    "relevant_over_365_days_percent",

    "newest_over_365_days_percent",

    "age_p_value",
]


print(

    comparison_df[
        display_columns
    ].to_string(

        index=False,

        formatters={

            "relevant_median_age_days":
                lambda x:
                f"{x:.1f}",

            "newest_median_age_days":
                lambda x:
                f"{x:.1f}",

            "median_age_gap_days":
                lambda x:
                f"{x:+.1f}",

            "median_age_ratio":
                lambda x:
                f"{x:.2f}x",

            "relevant_within_7_days_percent":
                lambda x:
                f"{x:.2f}%",

            "newest_within_7_days_percent":
                lambda x:
                f"{x:.2f}%",

            "relevant_within_30_days_percent":
                lambda x:
                f"{x:.2f}%",

            "newest_within_30_days_percent":
                lambda x:
                f"{x:.2f}%",

            "relevant_over_365_days_percent":
                lambda x:
                f"{x:.2f}%",

            "newest_over_365_days_percent":
                lambda x:
                f"{x:.2f}%",

            "age_p_value":
                lambda x:
                f"{x:.3e}",
        }
    )
)


# ============================================================
# POOLED BALANCED RESULT
# ============================================================

relevant_pool = (

    balanced_df[
        balanced_df[
            "ordering"
        ] == "Relevant"
    ]
)


newest_pool = (

    balanced_df[
        balanced_df[
            "ordering"
        ] == "Newest"
    ]
)


rel_age = (
    relevant_pool[
        "review_age_days"
    ]
)


new_age = (
    newest_pool[
        "review_age_days"
    ]
)


pooled_u, pooled_p = (
    safe_mannwhitney(
        rel_age,
        new_age
    )
)


pooled_df = pd.DataFrame(
    [
        {

            "platforms":
                len(
                    eligible_platforms
                ),

            "reviews_per_ordering":
                len(
                    relevant_pool
                ),

            "total_reviews":
                len(
                    balanced_df
                ),

            "relevant_mean_age_days":
                rel_age.mean(),

            "newest_mean_age_days":
                new_age.mean(),

            "relevant_median_age_days":
                rel_age.median(),

            "newest_median_age_days":
                new_age.median(),

            "median_age_gap_days":
                (
                    rel_age.median()
                    -
                    new_age.median()
                ),

            "median_age_ratio":
                (
                    rel_age.median()
                    /
                    new_age.median()

                    if new_age.median() > 0

                    else np.nan
                ),

            "relevant_within_7_days_percent":
                (
                    rel_age <= 7
                ).mean() * 100,

            "newest_within_7_days_percent":
                (
                    new_age <= 7
                ).mean() * 100,

            "relevant_within_30_days_percent":
                (
                    rel_age <= 30
                ).mean() * 100,

            "newest_within_30_days_percent":
                (
                    new_age <= 30
                ).mean() * 100,

            "relevant_within_90_days_percent":
                (
                    rel_age <= 90
                ).mean() * 100,

            "newest_within_90_days_percent":
                (
                    new_age <= 90
                ).mean() * 100,

            "relevant_over_365_days_percent":
                (
                    rel_age > 365
                ).mean() * 100,

            "newest_over_365_days_percent":
                (
                    new_age > 365
                ).mean() * 100,

            "age_mannwhitney_u":
                pooled_u,

            "age_p_value":
                pooled_p,
        }
    ]
)


# ============================================================
# PRINT POOLED RESULT
# ============================================================

print(
    "\n"
)

print(
    "=" * 130
)

print(
    "POOLED BALANCED REVIEW-AGE RESULT"
)

print(
    "=" * 130
)


print(
    pooled_df.to_string(
        index=False
    )
)


# ============================================================
# TOP-K AGE GRADIENT
# ============================================================

topk_rows = []


for platform in eligible_platforms:

    for ordering in [
        "Relevant",
        "Newest"
    ]:

        full = (

            datasets[
                platform
            ][ordering]

            .head(
                BENCHMARK_N
            )

            .copy()
        )


        for k in TOP_K_VALUES:

            sample = (

                full
                .head(k)
                .copy()
            )


            age = (
                sample[
                    "review_age_days"
                ]
            )


            topk_rows.append({

                "platform":
                    platform,

                "ordering":
                    ordering,

                "top_k":
                    k,

                "mean_age_days":
                    age.mean(),

                "median_age_days":
                    age.median(),

                "within_7_days_percent":
                    (
                        age <= 7
                    ).mean() * 100,

                "within_30_days_percent":
                    (
                        age <= 30
                    ).mean() * 100,

                "over_365_days_percent":
                    (
                        age > 365
                    ).mean() * 100,
            })


topk_df = pd.DataFrame(
    topk_rows
)


# ============================================================
# POOLED TOP-K AGE
# ============================================================

pooled_topk_rows = []


for k in TOP_K_VALUES:

    rel_frames = []
    new_frames = []


    for platform in eligible_platforms:

        rel_frames.append(

            datasets[
                platform
            ]["Relevant"]

            .head(k)
        )


        new_frames.append(

            datasets[
                platform
            ]["Newest"]

            .head(k)
        )


    rel = pd.concat(
        rel_frames,
        ignore_index=True
    )


    new = pd.concat(
        new_frames,
        ignore_index=True
    )


    pooled_topk_rows.append({

        "top_k_per_platform":
            k,

        "relevant_median_age_days":
            rel[
                "review_age_days"
            ].median(),

        "newest_median_age_days":
            new[
                "review_age_days"
            ].median(),

        "median_age_gap_days":
            (
                rel[
                    "review_age_days"
                ].median()
                -
                new[
                    "review_age_days"
                ].median()
            ),

        "relevant_mean_age_days":
            rel[
                "review_age_days"
            ].mean(),

        "newest_mean_age_days":
            new[
                "review_age_days"
            ].mean(),

        "relevant_within_30_days_percent":
            (
                rel[
                    "review_age_days"
                ] <= 30
            ).mean() * 100,

        "newest_within_30_days_percent":
            (
                new[
                    "review_age_days"
                ] <= 30
            ).mean() * 100,
    })


pooled_topk_df = pd.DataFrame(
    pooled_topk_rows
)


# ============================================================
# PRINT TOP-K
# ============================================================

print(
    "\n"
)

print(
    "=" * 130
)

print(
    "POOLED TOP-K REVIEW-AGE GRADIENT"
)

print(
    "=" * 130
)


print(

    pooled_topk_df.to_string(

        index=False,

        formatters={

            "relevant_median_age_days":
                lambda x:
                f"{x:.1f}",

            "newest_median_age_days":
                lambda x:
                f"{x:.1f}",

            "median_age_gap_days":
                lambda x:
                f"{x:+.1f}",

            "relevant_mean_age_days":
                lambda x:
                f"{x:.1f}",

            "newest_mean_age_days":
                lambda x:
                f"{x:.1f}",

            "relevant_within_30_days_percent":
                lambda x:
                f"{x:.2f}%",

            "newest_within_30_days_percent":
                lambda x:
                f"{x:.2f}%",
        }
    )
)


# ============================================================
# SCRAPE ORDER VS REVIEW AGE
# ============================================================

correlation_rows = []


for platform in eligible_platforms:

    for ordering in [
        "Relevant",
        "Newest"
    ]:

        df = (

            datasets[
                platform
            ][ordering]

            .head(
                BENCHMARK_N
            )

            .copy()
        )


        rho, p = safe_spearman(

            df[
                ORDER_COL
            ],

            df[
                "review_age_days"
            ]
        )


        correlation_rows.append({

            "platform":
                platform,

            "ordering":
                ordering,

            "n":
                len(df),

            "rank_vs_age_rho":
                rho,

            "rank_vs_age_p_value":
                p,
        })


correlation_df = pd.DataFrame(
    correlation_rows
)


# ============================================================
# PRINT CORRELATIONS
# ============================================================

print(
    "\n"
)

print(
    "=" * 120
)

print(
    "SCRAPE ORDER VS REVIEW AGE"
)

print(
    "=" * 120
)


print(

    correlation_df.to_string(

        index=False,

        formatters={

            "rank_vs_age_rho":
                lambda x:
                f"{x:+.3f}",

            "rank_vs_age_p_value":
                lambda x:
                f"{x:.3e}",
        }
    )
)


# ============================================================
# CHART 1
# MEDIAN AGE BY PLATFORM
# ============================================================

chart_df = (
    comparison_df
    .set_index(
        "platform"
    )
)


ax = chart_df[
    [
        "relevant_median_age_days",
        "newest_median_age_days",
    ]
].plot(
    kind="bar",
    figsize=(11, 6)
)


ax.set_title(
    "Median Review Age: Relevant vs Newest"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Median Review Age (Days)"
)

ax.legend(
    [
        "Relevant",
        "Newest",
    ]
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR /
    "median_review_age_by_platform.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 2
# REVIEWS WITHIN 30 DAYS
# ============================================================

ax = chart_df[
    [
        "relevant_within_30_days_percent",
        "newest_within_30_days_percent",
    ]
].plot(
    kind="bar",
    figsize=(11, 6)
)


ax.set_title(
    "Reviews Posted Within Previous 30 Days"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Reviews (%)"
)

ax.legend(
    [
        "Relevant",
        "Newest",
    ]
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR /
    "within_30_days_by_platform.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 3
# POOLED TOP-K AGE
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    pooled_topk_df[
        "top_k_per_platform"
    ],
    pooled_topk_df[
        "relevant_median_age_days"
    ],
    marker="o",
    label="Relevant"
)


plt.plot(
    pooled_topk_df[
        "top_k_per_platform"
    ],
    pooled_topk_df[
        "newest_median_age_days"
    ],
    marker="o",
    label="Newest"
)


plt.xlabel(
    "Top-k Reviews per Platform"
)

plt.ylabel(
    "Median Review Age (Days)"
)

plt.title(
    "Review Age by Ranking Depth"
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR /
    "pooled_review_age_topk.png",
    dpi=300
)

plt.close()


# ============================================================
# MAKE EVERYTHING EXCEL-SAFE
# ============================================================

diagnostics_df = (
    make_excel_safe(
        diagnostics_df
    )
)

eligibility_df = (
    make_excel_safe(
        eligibility_df
    )
)

comparison_df = (
    make_excel_safe(
        comparison_df
    )
)

summary_df = (
    make_excel_safe(
        summary_df
    )
)

pooled_df = (
    make_excel_safe(
        pooled_df
    )
)

topk_df = (
    make_excel_safe(
        topk_df
    )
)

pooled_topk_df = (
    make_excel_safe(
        pooled_topk_df
    )
)

correlation_df = (
    make_excel_safe(
        correlation_df
    )
)

balanced_df = (
    make_excel_safe(
        balanced_df
    )
)


# ============================================================
# SAVE EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:


    diagnostics_df.to_excel(
        writer,
        sheet_name="Date Diagnostics",
        index=False
    )


    eligibility_df.to_excel(
        writer,
        sheet_name="Eligibility",
        index=False
    )


    comparison_df.to_excel(
        writer,
        sheet_name="Platform Age Comparison",
        index=False
    )


    summary_df.to_excel(
        writer,
        sheet_name="Ordering Age Summary",
        index=False
    )


    pooled_df.to_excel(
        writer,
        sheet_name="Pooled Age Result",
        index=False
    )


    topk_df.to_excel(
        writer,
        sheet_name="Platform TopK Age",
        index=False
    )


    pooled_topk_df.to_excel(
        writer,
        sheet_name="Pooled TopK Age",
        index=False
    )


    correlation_df.to_excel(
        writer,
        sheet_name="Rank Age Correlation",
        index=False
    )


    balanced_df.to_excel(
        writer,
        sheet_name="Balanced Reviews",
        index=False
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
)

print(
    "=" * 120
)

print(
    "REVIEW-AGE ANALYSIS COMPLETE"
)

print(
    "=" * 120
)


print(
    f"\nPlatforms analyzed: "
    f"{len(eligible_platforms)}"
)


print(
    f"\nTotal balanced reviews: "
    f"{len(balanced_df):,}"
)


print(
    f"\nExcel output:\n"
    f"{OUTPUT_EXCEL.resolve()}"
)


print(
    f"\nCharts:\n"
    f"{OUTPUT_DIR.resolve()}"
)