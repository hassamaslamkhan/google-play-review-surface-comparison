from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    mannwhitneyu,
    norm,
)


# ============================================================
# CONFIG
# ============================================================

RELEVANT_DIR = Path("./relevant_reviews")
NEWEST_DIR = Path("./newest_reviews")

OUTPUT_DIR = Path("./common_date_window_analysis")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_EXCEL = (
    OUTPUT_DIR /
    "common_date_window_robustness.xlsx"
)

MAX_SAMPLE_PER_ORDERING = 1700

RATING_COL = "rating"
TEXT_COL = "review_text"
HELPFUL_COL = "helpful_count"
ORDER_COL = "scrape_order"

DATE_CANDIDATES = [
    "review_date_iso",
    "review_date",
]

NEGATIVE_THRESHOLD = 3


# ============================================================
# PLATFORM NAME
# ============================================================

def get_platform_from_filename(path):

    name = path.stem.lower().strip()

    name = re.sub(
        r"^newest[_\-\s]*",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(
        r"_google_play_reviews.*$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = name.strip("_- ")

    mapping = {
        "facebook": "Facebook",
        "instagram": "Instagram",
        "snapchat": "Snapchat",
        "tiktok": "TikTok",
        "twitter": "X",
        "x": "X",
        "youtube": "YouTube",
    }

    return mapping.get(
        name,
        name.replace("_", " ").title()
    )


# ============================================================
# RATING PARSER
# ============================================================

def parse_rating(value):

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        )
    ):
        value = float(value)

        if 1 <= value <= 5:
            return value

        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    try:
        rating = float(text)

        if 1 <= rating <= 5:
            return rating

    except Exception:
        pass

    patterns = [
        r"rated\s+([1-5])",
        r"([1-5])\s*stars?",
        r"^([1-5])(?:\.0)?$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            rating = float(
                match.group(1)
            )

            if 1 <= rating <= 5:
                return rating

    return np.nan


# ============================================================
# DATE PARSER
# ============================================================

def find_and_parse_date(df, filename):

    date_col = None

    for candidate in DATE_CANDIDATES:

        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:

        raise ValueError(
            f"{filename}: none of these date columns exist: "
            f"{DATE_CANDIDATES}"
        )

    parsed = pd.to_datetime(
        df[date_col],
        errors="coerce",
        utc=True
    )

    # Convert to timezone-naive normalized calendar date.
    parsed = (
        parsed
        .dt.tz_convert(None)
        .dt.normalize()
    )

    return date_col, parsed


# ============================================================
# READ FILE
# ============================================================

def read_review_file(path):

    df = pd.read_excel(path)

    raw_n = len(df)

    required = [
        RATING_COL,
        TEXT_COL,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{path.name}: missing columns {missing}"
        )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    df[RATING_COL] = (
        df[RATING_COL]
        .apply(parse_rating)
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_source, parsed_date = (
        find_and_parse_date(
            df,
            path.name
        )
    )

    df["analysis_date"] = parsed_date

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["review_words"] = (
        df[TEXT_COL]
        .str.split()
        .str.len()
    )

    # --------------------------------------------------------
    # Helpful count
    # --------------------------------------------------------

    if HELPFUL_COL in df.columns:

        df[HELPFUL_COL] = (
            pd.to_numeric(
                df[HELPFUL_COL],
                errors="coerce"
            )
            .fillna(0)
        )

    else:

        df[HELPFUL_COL] = 0

    # --------------------------------------------------------
    # Scrape order
    # --------------------------------------------------------

    if ORDER_COL in df.columns:

        df[ORDER_COL] = (
            pd.to_numeric(
                df[ORDER_COL],
                errors="coerce"
            )
        )

        df = df.sort_values(
            ORDER_COL,
            na_position="last"
        )

    else:

        print(
            f"WARNING: {path.name} has no "
            f"{ORDER_COL}; row order will be used."
        )

        df[ORDER_COL] = (
            np.arange(len(df)) + 1
        )

    # --------------------------------------------------------
    # Valid data
    # --------------------------------------------------------

    valid_rating = (
        df[RATING_COL].notna()
    )

    valid_date = (
        df["analysis_date"].notna()
    )

    df = df[
        valid_rating &
        valid_date
    ].copy()

    df["is_negative"] = (
        df[RATING_COL] <
        NEGATIVE_THRESHOLD
    )

    diagnostics = {

        "raw_rows":
            raw_n,

        "valid_rating_and_date_rows":
            len(df),

        "date_source":
            date_source,

        "earliest_date":
            df["analysis_date"].min(),

        "latest_date":
            df["analysis_date"].max(),
    }

    return (
        df.reset_index(drop=True),
        diagnostics
    )


# ============================================================
# TWO-PROPORTION Z TEST
# ============================================================

def two_proportion_z_test(
    success_1,
    n_1,
    success_2,
    n_2
):

    if (
        n_1 == 0 or
        n_2 == 0
    ):

        return (
            np.nan,
            np.nan
        )

    p1 = success_1 / n_1
    p2 = success_2 / n_2

    pooled = (
        success_1 +
        success_2
    ) / (
        n_1 +
        n_2
    )

    se = math.sqrt(
        pooled *
        (1 - pooled) *
        (
            1 / n_1 +
            1 / n_2
        )
    )

    if se == 0:

        return (
            np.nan,
            np.nan
        )

    z = (
        p1 - p2
    ) / se

    p = (
        2 *
        norm.sf(
            abs(z)
        )
    )

    return z, p


# ============================================================
# MANN-WHITNEY
# ============================================================

def safe_mannwhitney(
    a,
    b
):

    a = pd.to_numeric(
        a,
        errors="coerce"
    ).dropna()

    b = pd.to_numeric(
        b,
        errors="coerce"
    ).dropna()

    if (
        len(a) < 2 or
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
# SAMPLE SUMMARY
# ============================================================

def summarize(
    platform,
    ordering,
    df
):

    n = len(df)

    neg_n = int(
        df["is_negative"].sum()
    )

    return {

        "platform":
            platform,

        "ordering":
            ordering,

        "n":
            n,

        "mean_rating":
            df[RATING_COL].mean(),

        "median_rating":
            df[RATING_COL].median(),

        "negative_n":
            neg_n,

        "negative_percent":
            (
                neg_n / n * 100
                if n > 0
                else np.nan
            ),

        "mean_words":
            df["review_words"].mean(),

        "median_words":
            df["review_words"].median(),

        "mean_helpful":
            df[HELPFUL_COL].mean(),

        "median_helpful":
            df[HELPFUL_COL].median(),

        "p90_helpful":
            df[HELPFUL_COL].quantile(
                0.90
            ),

        "p95_helpful":
            df[HELPFUL_COL].quantile(
                0.95
            ),

        "total_helpful":
            df[HELPFUL_COL].sum(),
    }


# ============================================================
# DISCOVER FILES
# ============================================================

relevant_files = {

    get_platform_from_filename(path):
        path

    for path in RELEVANT_DIR.glob(
        "*.xlsx"
    )
}


newest_files = {

    get_platform_from_filename(path):
        path

    for path in NEWEST_DIR.glob(
        "*.xlsx"
    )
}


platforms = sorted(
    set(relevant_files.keys())
    &
    set(newest_files.keys())
)


print("=" * 120)
print(
    "COMMON-DATE-WINDOW ROBUSTNESS ANALYSIS"
)
print("=" * 120)

print(
    "\nMatched platforms:"
)

for platform in platforms:

    print(
        f"  {platform}"
    )


if not platforms:

    raise RuntimeError(
        "No matched platforms found."
    )


# ============================================================
# LOAD DATA
# ============================================================

datasets = {}

diagnostic_rows = []


for platform in platforms:

    rel, rel_diag = (
        read_review_file(
            relevant_files[
                platform
            ]
        )
    )

    new, new_diag = (
        read_review_file(
            newest_files[
                platform
            ]
        )
    )

    datasets[platform] = {

        "Relevant":
            rel,

        "Newest":
            new,
    }

    diagnostic_rows.append({

        "platform":
            platform,

        "ordering":
            "Relevant",

        "file":
            relevant_files[
                platform
            ].name,

        **rel_diag
    })

    diagnostic_rows.append({

        "platform":
            platform,

        "ordering":
            "Newest",

        "file":
            newest_files[
                platform
            ].name,

        **new_diag
    })


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


# ============================================================
# DETERMINE COMMON DATE WINDOWS
# ============================================================

window_rows = []

common_data = {}


for platform in platforms:

    rel = datasets[
        platform
    ]["Relevant"]

    new = datasets[
        platform
    ]["Newest"]

    rel_start = (
        rel[
            "analysis_date"
        ].min()
    )

    rel_end = (
        rel[
            "analysis_date"
        ].max()
    )

    new_start = (
        new[
            "analysis_date"
        ].min()
    )

    new_end = (
        new[
            "analysis_date"
        ].max()
    )

    common_start = max(
        rel_start,
        new_start
    )

    common_end = min(
        rel_end,
        new_end
    )


    if common_start > common_end:

        print(
            f"\nNO COMMON DATE WINDOW: "
            f"{platform}"
        )

        continue


    rel_common = rel[
        (
            rel[
                "analysis_date"
            ] >= common_start
        )
        &
        (
            rel[
                "analysis_date"
            ] <= common_end
        )
    ].copy()


    new_common = new[
        (
            new[
                "analysis_date"
            ] >= common_start
        )
        &
        (
            new[
                "analysis_date"
            ] <= common_end
        )
    ].copy()


    pair_n = min(
        len(rel_common),
        len(new_common),
        MAX_SAMPLE_PER_ORDERING
    )


    common_data[
        platform
    ] = {

        "Relevant":
            rel_common,

        "Newest":
            new_common,

        "pair_n":
            pair_n,

        "common_start":
            common_start,

        "common_end":
            common_end,
    }


    window_rows.append({

        "platform":
            platform,

        "relevant_start":
            rel_start,

        "relevant_end":
            rel_end,

        "newest_start":
            new_start,

        "newest_end":
            new_end,

        "common_start":
            common_start,

        "common_end":
            common_end,

        "common_window_days":
            (
                common_end -
                common_start
            ).days + 1,

        "relevant_in_window":
            len(
                rel_common
            ),

        "newest_in_window":
            len(
                new_common
            ),

        "equal_pair_n":
            pair_n,
    })


window_df = pd.DataFrame(
    window_rows
)


print("\n")
print("=" * 140)
print("COMMON DATE WINDOWS")
print("=" * 140)

print(
    window_df.to_string(
        index=False
    )
)


# ============================================================
# PLATFORM-EQUAL PAIR ANALYSIS
# ============================================================

comparison_rows = []

sample_frames = []


for platform in common_data:

    data = common_data[
        platform
    ]

    n = data[
        "pair_n"
    ]

    if n < 20:

        print(
            f"\nSkipping {platform}: "
            f"common equal sample is only {n}"
        )

        continue


    relevant = (
        data[
            "Relevant"
        ]
        .head(n)
        .copy()
    )


    newest = (
        data[
            "Newest"
        ]
        .head(n)
        .copy()
    )


    relevant[
        "platform"
    ] = platform

    newest[
        "platform"
    ] = platform

    relevant[
        "ordering"
    ] = "Relevant"

    newest[
        "ordering"
    ] = "Newest"


    sample_frames.extend(
        [
            relevant,
            newest
        ]
    )


    rel_summary = summarize(
        platform,
        "Relevant",
        relevant
    )

    new_summary = summarize(
        platform,
        "Newest",
        newest
    )


    rel_neg_rate = (
        relevant[
            "is_negative"
        ].mean()
    )

    new_neg_rate = (
        newest[
            "is_negative"
        ].mean()
    )


    ner = (
        rel_neg_rate /
        new_neg_rate

        if new_neg_rate > 0

        else np.nan
    )


    neg_z, neg_p = (
        two_proportion_z_test(

            int(
                relevant[
                    "is_negative"
                ].sum()
            ),

            len(
                relevant
            ),

            int(
                newest[
                    "is_negative"
                ].sum()
            ),

            len(
                newest
            )
        )
    )


    rating_u, rating_p = (
        safe_mannwhitney(
            relevant[
                RATING_COL
            ],
            newest[
                RATING_COL
            ]
        )
    )


    words_u, words_p = (
        safe_mannwhitney(
            relevant[
                "review_words"
            ],
            newest[
                "review_words"
            ]
        )
    )


    helpful_u, helpful_p = (
        safe_mannwhitney(
            relevant[
                HELPFUL_COL
            ],
            newest[
                HELPFUL_COL
            ]
        )
    )


    comparison_rows.append({

        "platform":
            platform,

        "common_start":
            data[
                "common_start"
            ],

        "common_end":
            data[
                "common_end"
            ],

        "common_window_days":
            (
                data[
                    "common_end"
                ]
                -
                data[
                    "common_start"
                ]
            ).days + 1,

        "n_each":
            n,

        # ----------------------------------------
        # Rating
        # ----------------------------------------

        "relevant_mean_rating":
            rel_summary[
                "mean_rating"
            ],

        "newest_mean_rating":
            new_summary[
                "mean_rating"
            ],

        "rating_gap_newest_minus_relevant":
            (
                new_summary[
                    "mean_rating"
                ]
                -
                rel_summary[
                    "mean_rating"
                ]
            ),

        # ----------------------------------------
        # Negative
        # ----------------------------------------

        "relevant_negative_percent":
            rel_summary[
                "negative_percent"
            ],

        "newest_negative_percent":
            new_summary[
                "negative_percent"
            ],

        "negative_percentage_point_gap":
            (
                rel_summary[
                    "negative_percent"
                ]
                -
                new_summary[
                    "negative_percent"
                ]
            ),

        "negative_exposure_ratio":
            ner,

        "negative_z":
            neg_z,

        "negative_p_value":
            neg_p,

        # ----------------------------------------
        # Length
        # ----------------------------------------

        "relevant_mean_words":
            rel_summary[
                "mean_words"
            ],

        "newest_mean_words":
            new_summary[
                "mean_words"
            ],

        "review_length_ratio":
            (
                rel_summary[
                    "mean_words"
                ]
                /
                new_summary[
                    "mean_words"
                ]

                if new_summary[
                    "mean_words"
                ] > 0

                else np.nan
            ),

        "relevant_median_words":
            rel_summary[
                "median_words"
            ],

        "newest_median_words":
            new_summary[
                "median_words"
            ],

        # ----------------------------------------
        # Helpful
        # ----------------------------------------

        "relevant_mean_helpful":
            rel_summary[
                "mean_helpful"
            ],

        "newest_mean_helpful":
            new_summary[
                "mean_helpful"
            ],

        "relevant_median_helpful":
            rel_summary[
                "median_helpful"
            ],

        "newest_median_helpful":
            new_summary[
                "median_helpful"
            ],

        # ----------------------------------------
        # Statistics
        # ----------------------------------------

        "rating_mannwhitney_u":
            rating_u,

        "rating_p_value":
            rating_p,

        "words_mannwhitney_u":
            words_u,

        "words_p_value":
            words_p,

        "helpful_mannwhitney_u":
            helpful_u,

        "helpful_p_value":
            helpful_p,
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# PRINT PLATFORM RESULTS
# ============================================================

print("\n")
print("=" * 165)
print(
    "COMMON-DATE RELEVANT VS NEWEST COMPARISON"
)
print("=" * 165)


display_columns = [

    "platform",

    "common_start",

    "common_end",

    "n_each",

    "relevant_mean_rating",

    "newest_mean_rating",

    "rating_gap_newest_minus_relevant",

    "relevant_negative_percent",

    "newest_negative_percent",

    "negative_exposure_ratio",

    "relevant_mean_words",

    "newest_mean_words",

    "review_length_ratio",

    "relevant_median_helpful",

    "newest_median_helpful",

    "negative_p_value",
]


print(

    comparison_df[
        display_columns
    ].to_string(

        index=False,

        formatters={

            "relevant_mean_rating":
                lambda x:
                f"{x:.2f}",

            "newest_mean_rating":
                lambda x:
                f"{x:.2f}",

            "rating_gap_newest_minus_relevant":
                lambda x:
                f"{x:+.2f}",

            "relevant_negative_percent":
                lambda x:
                f"{x:.2f}%",

            "newest_negative_percent":
                lambda x:
                f"{x:.2f}%",

            "negative_exposure_ratio":
                lambda x:
                f"{x:.2f}x",

            "relevant_mean_words":
                lambda x:
                f"{x:.1f}",

            "newest_mean_words":
                lambda x:
                f"{x:.1f}",

            "review_length_ratio":
                lambda x:
                f"{x:.2f}x",

            "relevant_median_helpful":
                lambda x:
                f"{x:.1f}",

            "newest_median_helpful":
                lambda x:
                f"{x:.1f}",

            "negative_p_value":
                lambda x:
                f"{x:.3e}",
        }
    )
)


# ============================================================
# MICRO POOLED RESULT
#
# Uses all equalized platform pairs.
# Larger platform pairs therefore contribute more.
# ============================================================

sample_df = pd.concat(
    sample_frames,
    ignore_index=True
)


rel_pool = sample_df[
    sample_df[
        "ordering"
    ] == "Relevant"
]


new_pool = sample_df[
    sample_df[
        "ordering"
    ] == "Newest"
]


rel_neg = (
    rel_pool[
        "is_negative"
    ].mean()
)

new_neg = (
    new_pool[
        "is_negative"
    ].mean()
)


micro_ner = (
    rel_neg /
    new_neg

    if new_neg > 0
    else np.nan
)


micro_df = pd.DataFrame(
    [
        {

            "platforms":
                comparison_df[
                    "platform"
                ].nunique(),

            "relevant_n":
                len(
                    rel_pool
                ),

            "newest_n":
                len(
                    new_pool
                ),

            "relevant_mean_rating":
                rel_pool[
                    RATING_COL
                ].mean(),

            "newest_mean_rating":
                new_pool[
                    RATING_COL
                ].mean(),

            "rating_gap":
                (
                    new_pool[
                        RATING_COL
                    ].mean()
                    -
                    rel_pool[
                        RATING_COL
                    ].mean()
                ),

            "relevant_negative_percent":
                rel_neg * 100,

            "newest_negative_percent":
                new_neg * 100,

            "negative_pp_gap":
                (
                    rel_neg -
                    new_neg
                ) * 100,

            "negative_exposure_ratio":
                micro_ner,

            "relevant_mean_words":
                rel_pool[
                    "review_words"
                ].mean(),

            "newest_mean_words":
                new_pool[
                    "review_words"
                ].mean(),

            "review_length_ratio":
                (
                    rel_pool[
                        "review_words"
                    ].mean()
                    /
                    new_pool[
                        "review_words"
                    ].mean()
                ),

            "relevant_median_helpful":
                rel_pool[
                    HELPFUL_COL
                ].median(),

            "newest_median_helpful":
                new_pool[
                    HELPFUL_COL
                ].median(),
        }
    ]
)


print("\n")
print("=" * 120)
print("MICRO-POOLED COMMON-DATE RESULT")
print("=" * 120)

print(
    micro_df.to_string(
        index=False
    )
)


# ============================================================
# MACRO / EQUAL-PLATFORM RESULT
#
# Each platform contributes equal weight.
# ============================================================

macro_df = pd.DataFrame(
    [
        {

            "platforms":
                len(
                    comparison_df
                ),

            "mean_relevant_rating":
                comparison_df[
                    "relevant_mean_rating"
                ].mean(),

            "mean_newest_rating":
                comparison_df[
                    "newest_mean_rating"
                ].mean(),

            "mean_rating_gap":
                comparison_df[
                    "rating_gap_newest_minus_relevant"
                ].mean(),

            "mean_relevant_negative_percent":
                comparison_df[
                    "relevant_negative_percent"
                ].mean(),

            "mean_newest_negative_percent":
                comparison_df[
                    "newest_negative_percent"
                ].mean(),

            "mean_negative_pp_gap":
                comparison_df[
                    "negative_percentage_point_gap"
                ].mean(),

            "mean_platform_ner":
                comparison_df[
                    "negative_exposure_ratio"
                ].mean(),

            "mean_relevant_words":
                comparison_df[
                    "relevant_mean_words"
                ].mean(),

            "mean_newest_words":
                comparison_df[
                    "newest_mean_words"
                ].mean(),

            "mean_length_ratio":
                comparison_df[
                    "review_length_ratio"
                ].mean(),
        }
    ]
)


print("\n")
print("=" * 120)
print("MACRO / EQUAL-PLATFORM COMMON-DATE RESULT")
print("=" * 120)

print(
    macro_df.to_string(
        index=False
    )
)


# ============================================================
# GLOBAL-BALANCED SENSITIVITY
#
# Same N for every platform AND ordering.
# ============================================================

global_n = int(
    comparison_df[
        "n_each"
    ].min()
)


global_frames = []


for platform in comparison_df[
    "platform"
]:

    data = common_data[
        platform
    ]

    rel = (
        data[
            "Relevant"
        ]
        .head(global_n)
        .copy()
    )

    new = (
        data[
            "Newest"
        ]
        .head(global_n)
        .copy()
    )

    rel[
        "ordering"
    ] = "Relevant"

    new[
        "ordering"
    ] = "Newest"

    rel[
        "platform"
    ] = platform

    new[
        "platform"
    ] = platform

    global_frames.extend(
        [
            rel,
            new
        ]
    )


global_df = pd.concat(
    global_frames,
    ignore_index=True
)


global_rel = global_df[
    global_df[
        "ordering"
    ] == "Relevant"
]


global_new = global_df[
    global_df[
        "ordering"
    ] == "Newest"
]


g_rel_neg = (
    global_rel[
        "is_negative"
    ].mean()
)

g_new_neg = (
    global_new[
        "is_negative"
    ].mean()
)


global_balanced_df = pd.DataFrame(
    [
        {

            "global_n_each_platform_ordering":
                global_n,

            "total_reviews":
                len(
                    global_df
                ),

            "relevant_mean_rating":
                global_rel[
                    RATING_COL
                ].mean(),

            "newest_mean_rating":
                global_new[
                    RATING_COL
                ].mean(),

            "rating_gap":
                (
                    global_new[
                        RATING_COL
                    ].mean()
                    -
                    global_rel[
                        RATING_COL
                    ].mean()
                ),

            "relevant_negative_percent":
                g_rel_neg * 100,

            "newest_negative_percent":
                g_new_neg * 100,

            "negative_pp_gap":
                (
                    g_rel_neg -
                    g_new_neg
                ) * 100,

            "negative_exposure_ratio":
                (
                    g_rel_neg /
                    g_new_neg

                    if g_new_neg > 0
                    else np.nan
                ),

            "relevant_mean_words":
                global_rel[
                    "review_words"
                ].mean(),

            "newest_mean_words":
                global_new[
                    "review_words"
                ].mean(),

            "review_length_ratio":
                (
                    global_rel[
                        "review_words"
                    ].mean()
                    /
                    global_new[
                        "review_words"
                    ].mean()
                ),

            "relevant_median_helpful":
                global_rel[
                    HELPFUL_COL
                ].median(),

            "newest_median_helpful":
                global_new[
                    HELPFUL_COL
                ].median(),
        }
    ]
)


print("\n")
print("=" * 120)
print(
    "GLOBAL-BALANCED COMMON-DATE SENSITIVITY"
)
print("=" * 120)

print(
    global_balanced_df.to_string(
        index=False
    )
)


# ============================================================
# CHART 1
# NEGATIVE PERCENTAGE
# ============================================================

chart_df = (
    comparison_df
    .set_index(
        "platform"
    )
)


ax = chart_df[
    [
        "relevant_negative_percent",
        "newest_negative_percent",
    ]
].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title(
    "Negative Review Share Within Common Date Windows"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Negative Reviews (%)"
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
    "common_date_negative_share.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 2
# RATING
# ============================================================

ax = chart_df[
    [
        "relevant_mean_rating",
        "newest_mean_rating",
    ]
].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title(
    "Mean Rating Within Common Date Windows"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Mean Star Rating"
)

ax.set_ylim(
    1,
    5
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
    "common_date_mean_rating.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 3
# REVIEW LENGTH
# ============================================================

ax = chart_df[
    [
        "relevant_mean_words",
        "newest_mean_words",
    ]
].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title(
    "Review Length Within Common Date Windows"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Mean Words"
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
    "common_date_review_length.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE
# ============================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    diagnostics_df.to_excel(
        writer,
        sheet_name="File Diagnostics",
        index=False
    )

    window_df.to_excel(
        writer,
        sheet_name="Common Date Windows",
        index=False
    )

    comparison_df.to_excel(
        writer,
        sheet_name="Platform Comparison",
        index=False
    )

    micro_df.to_excel(
        writer,
        sheet_name="Micro Pooled",
        index=False
    )

    macro_df.to_excel(
        writer,
        sheet_name="Macro Equal Platform",
        index=False
    )

    global_balanced_df.to_excel(
        writer,
        sheet_name="Global Balanced",
        index=False
    )

    sample_df.to_excel(
        writer,
        sheet_name="Common Date Samples",
        index=False
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 120)
print(
    "COMMON-DATE ROBUSTNESS ANALYSIS COMPLETE"
)
print("=" * 120)

print(
    f"\nExcel:\n"
    f"{OUTPUT_EXCEL.resolve()}"
)

print(
    f"\nCharts:\n"
    f"{OUTPUT_DIR.resolve()}"
)