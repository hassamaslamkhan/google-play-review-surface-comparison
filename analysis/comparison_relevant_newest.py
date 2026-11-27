from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIG
# ============================================================

RELEVANT_DIR = Path("./relevant_reviews")
NEWEST_DIR = Path("./newest_reviews")

OUTPUT_DIR = Path("./relevant_vs_newest_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EXCEL = (
    OUTPUT_DIR /
    "relevant_vs_newest_fixed_1700.xlsx"
)

# Fixed balanced benchmark
BENCHMARK_N = 1700

RATING_COL = "rating"
TEXT_COL = "review_text"
HELPFUL_COL = "helpful_count"
ORDER_COL = "scrape_order"

NEGATIVE_THRESHOLD = 3


# ============================================================
# PLATFORM NAME FROM FILENAME
# ============================================================

def get_platform_from_filename(path):

    name = path.stem.lower().strip()

    # Remove newest prefix
    name = re.sub(
        r"^newest[_\-\s]*",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove suffixes
    name = re.sub(
        r"_google_play_reviews.*$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(
        r"_reviews.*$",
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
        "x": "X",
        "twitter": "X",
        "youtube": "YouTube",
    }

    return mapping.get(
        name,
        name.replace("_", " ").title()
    )


# ============================================================
# ROBUST RATING PARSER
# ============================================================

def parse_rating(value):
    """
    Handles:
        1
        1.0
        "1"
        "1 star"
        "Rated 1 stars out of five stars"
        "Rated 5 stars out of 5"
    """

    if pd.isna(value):
        return np.nan

    # Already numeric
    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):

        rating = float(value)

        if 1 <= rating <= 5:
            return rating

        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    # Try direct numeric conversion
    try:

        rating = float(text)

        if 1 <= rating <= 5:
            return rating

    except Exception:
        pass

    # Search for rating 1-5 in text
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
# READ REVIEW FILE
# ============================================================

def read_review_file(path):

    raw = pd.read_excel(path)

    raw_rows = len(raw)

    if RATING_COL not in raw.columns:

        raise ValueError(
            f"Missing '{RATING_COL}' column"
        )

    if TEXT_COL not in raw.columns:

        raise ValueError(
            f"Missing '{TEXT_COL}' column"
        )


    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    raw["rating_original"] = raw[
        RATING_COL
    ]

    raw[RATING_COL] = raw[
        RATING_COL
    ].apply(parse_rating)


    valid_rating_count = (
        raw[RATING_COL]
        .notna()
        .sum()
    )


    # --------------------------------------------------------
    # Review text
    # --------------------------------------------------------

    raw[TEXT_COL] = (
        raw[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    raw["review_words"] = (
        raw[TEXT_COL]
        .str.split()
        .str.len()
    )


    # --------------------------------------------------------
    # Helpful
    # --------------------------------------------------------

    if HELPFUL_COL in raw.columns:

        raw[HELPFUL_COL] = (
            pd.to_numeric(
                raw[HELPFUL_COL],
                errors="coerce"
            )
            .fillna(0)
        )

    else:

        raw[HELPFUL_COL] = 0


    # --------------------------------------------------------
    # Scrape order
    # --------------------------------------------------------

    if ORDER_COL in raw.columns:

        raw[ORDER_COL] = pd.to_numeric(
            raw[ORDER_COL],
            errors="coerce"
        )

        raw = raw.sort_values(
            ORDER_COL,
            na_position="last"
        )


    # --------------------------------------------------------
    # Keep valid ratings
    # --------------------------------------------------------

    df = raw[
        raw[RATING_COL].notna()
    ].copy()


    # Negative = rating 1 or 2
    df["is_negative"] = (
        df[RATING_COL] <
        NEGATIVE_THRESHOLD
    )


    diagnostics = {
        "raw_rows": raw_rows,
        "valid_rating_rows":
            int(valid_rating_count),
        "invalid_rating_rows":
            int(
                raw_rows -
                valid_rating_count
            ),
    }


    return (
        df.reset_index(drop=True),
        diagnostics
    )


# ============================================================
# SUMMARIZE ONE SAMPLE
# ============================================================

def summarize_sample(
    platform,
    ordering,
    df
):

    n = len(df)

    negative_count = int(
        df["is_negative"].sum()
    )

    negative_percent = (
        negative_count / n * 100
        if n
        else np.nan
    )


    return {

        "platform":
            platform,

        "ordering":
            ordering,

        "n":
            n,

        # Rating
        "mean_rating":
            df[RATING_COL].mean(),

        "median_rating":
            df[RATING_COL].median(),

        "negative_reviews":
            negative_count,

        "negative_percent":
            negative_percent,

        # Length
        "mean_words":
            df["review_words"].mean(),

        "median_words":
            df["review_words"].median(),

        "p75_words":
            df["review_words"].quantile(
                0.75
            ),

        "p90_words":
            df["review_words"].quantile(
                0.90
            ),

        # Helpful
        "total_helpful":
            df[HELPFUL_COL].sum(),

        "mean_helpful":
            df[HELPFUL_COL].mean(),

        "median_helpful":
            df[HELPFUL_COL].median(),

        "p75_helpful":
            df[HELPFUL_COL].quantile(
                0.75
            ),

        "p90_helpful":
            df[HELPFUL_COL].quantile(
                0.90
            ),

        "p95_helpful":
            df[HELPFUL_COL].quantile(
                0.95
            ),

        "p99_helpful":
            df[HELPFUL_COL].quantile(
                0.99
            ),

        "helpful_gt_100":
            int(
                (
                    df[HELPFUL_COL]
                    > 100
                ).sum()
            ),

        "helpful_gt_1000":
            int(
                (
                    df[HELPFUL_COL]
                    > 1000
                ).sum()
            ),
    }


# ============================================================
# MANN-WHITNEY HELPER
# ============================================================

def mann_whitney_safe(
    series_a,
    series_b
):

    a = (
        pd.to_numeric(
            series_a,
            errors="coerce"
        )
        .dropna()
    )

    b = (
        pd.to_numeric(
            series_b,
            errors="coerce"
        )
        .dropna()
    )

    if len(a) < 2 or len(b) < 2:

        return np.nan, np.nan

    try:

        result = mannwhitneyu(
            a,
            b,
            alternative="two-sided"
        )

        return (
            result.statistic,
            result.pvalue
        )

    except Exception:

        return np.nan, np.nan


# ============================================================
# FIND FILES
# ============================================================

relevant_files = {

    get_platform_from_filename(path):
        path

    for path in (
        RELEVANT_DIR.glob("*.xlsx")
    )
}


newest_files = {

    get_platform_from_filename(path):
        path

    for path in (
        NEWEST_DIR.glob("*.xlsx")
    )
}


all_platforms = sorted(
    set(relevant_files.keys())
    |
    set(newest_files.keys())
)


print("=" * 110)
print(
    "RELEVANT VS NEWEST GOOGLE PLAY REVIEW ANALYSIS"
)
print("=" * 110)

print(
    f"\nFixed benchmark: "
    f"{BENCHMARK_N:,} reviews "
    f"per ordering per platform"
)


# ============================================================
# DISPLAY FILES
# ============================================================

print("\nRELEVANT FILES")

for platform, path in sorted(
    relevant_files.items()
):

    print(
        f"  {platform:12s} "
        f"-> {path.name}"
    )


print("\nNEWEST FILES")

for platform, path in sorted(
    newest_files.items()
):

    print(
        f"  {platform:12s} "
        f"-> {path.name}"
    )


# ============================================================
# LOAD FILES
# ============================================================

datasets = {}

diagnostic_rows = []


for platform in all_platforms:

    datasets[
        platform
    ] = {}


    # --------------------------------------------------------
    # Relevant
    # --------------------------------------------------------

    if platform in relevant_files:

        try:

            relevant_df, diag = (
                read_review_file(
                    relevant_files[
                        platform
                    ]
                )
            )

            datasets[
                platform
            ]["Relevant"] = (
                relevant_df
            )

            diagnostic_rows.append({

                "platform":
                    platform,

                "ordering":
                    "Relevant",

                "file":
                    relevant_files[
                        platform
                    ].name,

                **diag
            })

        except Exception as e:

            print(
                f"\nERROR reading "
                f"{platform} Relevant:"
            )

            print(e)


    # --------------------------------------------------------
    # Newest
    # --------------------------------------------------------

    if platform in newest_files:

        try:

            newest_df, diag = (
                read_review_file(
                    newest_files[
                        platform
                    ]
                )
            )

            datasets[
                platform
            ]["Newest"] = newest_df

            diagnostic_rows.append({

                "platform":
                    platform,

                "ordering":
                    "Newest",

                "file":
                    newest_files[
                        platform
                    ].name,

                **diag
            })

        except Exception as e:

            print(
                f"\nERROR reading "
                f"{platform} Newest:"
            )

            print(e)


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


# ============================================================
# ELIGIBILITY CHECK
# ============================================================

eligibility_rows = []

eligible_platforms = []


for platform in all_platforms:

    rel_exists = (
        "Relevant"
        in datasets.get(
            platform,
            {}
        )
    )

    new_exists = (
        "Newest"
        in datasets.get(
            platform,
            {}
        )
    )


    relevant_n = (
        len(
            datasets[
                platform
            ]["Relevant"]
        )
        if rel_exists
        else 0
    )


    newest_n = (
        len(
            datasets[
                platform
            ]["Newest"]
        )
        if new_exists
        else 0
    )


    if not rel_exists:

        status = (
            "EXCLUDED - "
            "Relevant file missing"
        )

    elif not new_exists:

        status = (
            "EXCLUDED - "
            "Newest file missing"
        )

    elif relevant_n < BENCHMARK_N:

        status = (
            "EXCLUDED - "
            f"Relevant has only "
            f"{relevant_n:,} valid reviews"
        )

    elif newest_n < BENCHMARK_N:

        status = (
            "EXCLUDED - "
            f"Newest has only "
            f"{newest_n:,} valid reviews"
        )

    else:

        status = "ELIGIBLE"

        eligible_platforms.append(
            platform
        )


    eligibility_rows.append({

        "platform":
            platform,

        "relevant_valid_n":
            relevant_n,

        "newest_valid_n":
            newest_n,

        "required_n_each":
            BENCHMARK_N,

        "status":
            status
    })


eligibility_df = pd.DataFrame(
    eligibility_rows
)


print("\n")
print("=" * 110)
print("FILE / RATING DIAGNOSTICS")
print("=" * 110)

if not diagnostics_df.empty:

    print(
        diagnostics_df.to_string(
            index=False
        )
    )


print("\n")
print("=" * 110)
print("PLATFORM ELIGIBILITY")
print("=" * 110)

print(
    eligibility_df.to_string(
        index=False
    )
)


print("\nEligible platforms:")

for platform in eligible_platforms:

    print(
        f"  ✓ {platform}"
    )


excluded = eligibility_df[
    eligibility_df[
        "status"
    ] != "ELIGIBLE"
]


if not excluded.empty:

    print("\nExcluded platforms:")

    for _, row in (
        excluded.iterrows()
    ):

        print(
            f"  ✗ {row['platform']}: "
            f"{row['status']}"
        )


if not eligible_platforms:

    raise RuntimeError(
        "\nNo platform currently has "
        f"{BENCHMARK_N:,} valid reviews "
        "in BOTH Relevant and Newest."
    )


# ============================================================
# BUILD BALANCED BENCHMARK
# ============================================================

summary_rows = []
comparison_rows = []
benchmark_frames = []


for platform in eligible_platforms:

    relevant_full = (
        datasets[
            platform
        ]["Relevant"]
    )

    newest_full = (
        datasets[
            platform
        ]["Newest"]
    )


    # --------------------------------------------------------
    # First exactly 1700 from each ordering
    # --------------------------------------------------------

    relevant = (
        relevant_full
        .head(BENCHMARK_N)
        .copy()
    )

    newest = (
        newest_full
        .head(BENCHMARK_N)
        .copy()
    )


    relevant["platform"] = platform
    relevant["ordering"] = "Relevant"

    newest["platform"] = platform
    newest["ordering"] = "Newest"


    benchmark_frames.append(
        relevant
    )

    benchmark_frames.append(
        newest
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    rel_summary = summarize_sample(
        platform,
        "Relevant",
        relevant
    )

    new_summary = summarize_sample(
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
    # NEGATIVE EXPOSURE RATIO
    # ========================================================

    relevant_negative_rate = (
        relevant[
            "is_negative"
        ].mean()
    )

    newest_negative_rate = (
        newest[
            "is_negative"
        ].mean()
    )


    negative_exposure_ratio = (

        relevant_negative_rate
        /
        newest_negative_rate

        if newest_negative_rate > 0
        else np.nan
    )


    # ========================================================
    # RATING GAP
    #
    # Positive = Newest rating is higher
    # than Relevant rating.
    # ========================================================

    rating_gap = (
        newest[RATING_COL].mean()
        -
        relevant[RATING_COL].mean()
    )


    # ========================================================
    # NEGATIVE PERCENTAGE-POINT GAP
    # ========================================================

    negative_pp_gap = (
        relevant_negative_rate
        -
        newest_negative_rate
    ) * 100


    # ========================================================
    # LENGTH RATIOS
    # ========================================================

    relevant_mean_words = (
        relevant[
            "review_words"
        ].mean()
    )

    newest_mean_words = (
        newest[
            "review_words"
        ].mean()
    )


    review_length_ratio = (

        relevant_mean_words
        /
        newest_mean_words

        if newest_mean_words > 0
        else np.nan
    )


    # ========================================================
    # HELPFUL RATIO
    # ========================================================

    relevant_mean_helpful = (
        relevant[
            HELPFUL_COL
        ].mean()
    )

    newest_mean_helpful = (
        newest[
            HELPFUL_COL
        ].mean()
    )


    helpful_mean_ratio = (

        relevant_mean_helpful
        /
        newest_mean_helpful

        if newest_mean_helpful > 0
        else np.nan
    )


    # ========================================================
    # STATISTICAL TESTS
    # ========================================================

    rating_u, rating_p = (
        mann_whitney_safe(
            relevant[RATING_COL],
            newest[RATING_COL]
        )
    )


    length_u, length_p = (
        mann_whitney_safe(
            relevant[
                "review_words"
            ],
            newest[
                "review_words"
            ]
        )
    )


    helpful_u, helpful_p = (
        mann_whitney_safe(
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

        "benchmark_n_each":
            BENCHMARK_N,

        # ----------------------------------------
        # Rating
        # ----------------------------------------

        "relevant_mean_rating":
            relevant[
                RATING_COL
            ].mean(),

        "newest_mean_rating":
            newest[
                RATING_COL
            ].mean(),

        "rating_gap_newest_minus_relevant":
            rating_gap,

        "relevant_median_rating":
            relevant[
                RATING_COL
            ].median(),

        "newest_median_rating":
            newest[
                RATING_COL
            ].median(),


        # ----------------------------------------
        # Negative
        # ----------------------------------------

        "relevant_negative_n":
            int(
                relevant[
                    "is_negative"
                ].sum()
            ),

        "newest_negative_n":
            int(
                newest[
                    "is_negative"
                ].sum()
            ),

        "relevant_negative_percent":
            relevant_negative_rate * 100,

        "newest_negative_percent":
            newest_negative_rate * 100,

        "negative_percentage_point_gap":
            negative_pp_gap,

        "negative_exposure_ratio":
            negative_exposure_ratio,


        # ----------------------------------------
        # Length
        # ----------------------------------------

        "relevant_mean_words":
            relevant_mean_words,

        "newest_mean_words":
            newest_mean_words,

        "relevant_median_words":
            relevant[
                "review_words"
            ].median(),

        "newest_median_words":
            newest[
                "review_words"
            ].median(),

        "review_length_ratio":
            review_length_ratio,


        # ----------------------------------------
        # Helpful
        # ----------------------------------------

        "relevant_total_helpful":
            relevant[
                HELPFUL_COL
            ].sum(),

        "newest_total_helpful":
            newest[
                HELPFUL_COL
            ].sum(),

        "relevant_mean_helpful":
            relevant_mean_helpful,

        "newest_mean_helpful":
            newest_mean_helpful,

        "relevant_median_helpful":
            relevant[
                HELPFUL_COL
            ].median(),

        "newest_median_helpful":
            newest[
                HELPFUL_COL
            ].median(),

        "helpful_mean_ratio":
            helpful_mean_ratio,


        # ----------------------------------------
        # Statistical tests
        # ----------------------------------------

        "rating_mannwhitney_u":
            rating_u,

        "rating_p_value":
            rating_p,

        "length_mannwhitney_u":
            length_u,

        "length_p_value":
            length_p,

        "helpful_mannwhitney_u":
            helpful_u,

        "helpful_p_value":
            helpful_p,
    })


# ============================================================
# CREATE DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

comparison_df = pd.DataFrame(
    comparison_rows
)

benchmark_df = pd.concat(
    benchmark_frames,
    ignore_index=True
)


# ============================================================
# PRINT MAIN COMPARISON
# ============================================================

print("\n")
print("=" * 150)
print(
    "RELEVANT VS NEWEST — "
    "FIXED 1,700 REVIEW COMPARISON"
)
print("=" * 150)


display_columns = [

    "platform",

    "benchmark_n_each",

    "relevant_mean_rating",

    "newest_mean_rating",

    "rating_gap_newest_minus_relevant",

    "relevant_negative_percent",

    "newest_negative_percent",

    "negative_percentage_point_gap",

    "negative_exposure_ratio",

    "relevant_mean_words",

    "newest_mean_words",

    "review_length_ratio",

    "relevant_median_helpful",

    "newest_median_helpful"
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

            "negative_percentage_point_gap":
                lambda x:
                f"{x:+.2f} pp",

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
        }
    )
)


# ============================================================
# POOLED BALANCED RESULT
# ============================================================

relevant_all = (
    benchmark_df[
        benchmark_df[
            "ordering"
        ] == "Relevant"
    ]
)


newest_all = (
    benchmark_df[
        benchmark_df[
            "ordering"
        ] == "Newest"
    ]
)


rel_neg = (
    relevant_all[
        "is_negative"
    ].mean()
)


new_neg = (
    newest_all[
        "is_negative"
    ].mean()
)


pooled_ner = (

    rel_neg / new_neg

    if new_neg > 0
    else np.nan
)


pooled_rating_gap = (

    newest_all[
        RATING_COL
    ].mean()

    -

    relevant_all[
        RATING_COL
    ].mean()
)


pooled_length_ratio = (

    relevant_all[
        "review_words"
    ].mean()

    /

    newest_all[
        "review_words"
    ].mean()

    if newest_all[
        "review_words"
    ].mean() > 0

    else np.nan
)


overall_df = pd.DataFrame(
    [
        {

            "eligible_platforms":
                len(
                    eligible_platforms
                ),

            "platform_list":
                ", ".join(
                    eligible_platforms
                ),

            "benchmark_n_each":
                BENCHMARK_N,

            "relevant_total_reviews":
                len(
                    relevant_all
                ),

            "newest_total_reviews":
                len(
                    newest_all
                ),

            # Rating
            "relevant_mean_rating":
                relevant_all[
                    RATING_COL
                ].mean(),

            "newest_mean_rating":
                newest_all[
                    RATING_COL
                ].mean(),

            "rating_gap_newest_minus_relevant":
                pooled_rating_gap,

            # Negative
            "relevant_negative_percent":
                rel_neg * 100,

            "newest_negative_percent":
                new_neg * 100,

            "negative_percentage_point_gap":
                (
                    rel_neg -
                    new_neg
                ) * 100,

            "negative_exposure_ratio":
                pooled_ner,

            # Length
            "relevant_mean_words":
                relevant_all[
                    "review_words"
                ].mean(),

            "newest_mean_words":
                newest_all[
                    "review_words"
                ].mean(),

            "review_length_ratio":
                pooled_length_ratio,

            "relevant_median_words":
                relevant_all[
                    "review_words"
                ].median(),

            "newest_median_words":
                newest_all[
                    "review_words"
                ].median(),

            # Helpful
            "relevant_mean_helpful":
                relevant_all[
                    HELPFUL_COL
                ].mean(),

            "newest_mean_helpful":
                newest_all[
                    HELPFUL_COL
                ].mean(),

            "relevant_median_helpful":
                relevant_all[
                    HELPFUL_COL
                ].median(),

            "newest_median_helpful":
                newest_all[
                    HELPFUL_COL
                ].median(),
        }
    ]
)


print("\n")
print("=" * 110)
print("POOLED BALANCED RESULT")
print("=" * 110)

print(
    overall_df.to_string(
        index=False
    )
)


# ============================================================
# CHART 1
# NEGATIVE REVIEW SHARE
# ============================================================

negative_chart = (

    summary_df
    .pivot(
        index="platform",
        columns="ordering",
        values="negative_percent"
    )
    .reindex(
        eligible_platforms
    )
)


ax = negative_chart.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Negative Review Share: Relevant vs Newest"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Reviews Below 3 Stars (%)"
)

ax.legend(
    title="Ordering"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "negative_share_relevant_vs_newest.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 2
# MEAN RATING
# ============================================================

rating_chart = (

    summary_df
    .pivot(
        index="platform",
        columns="ordering",
        values="mean_rating"
    )
    .reindex(
        eligible_platforms
    )
)


ax = rating_chart.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Mean Review Rating: Relevant vs Newest"
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
    title="Ordering"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "mean_rating_relevant_vs_newest.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 3
# REVIEW LENGTH
# ============================================================

length_chart = (

    summary_df
    .pivot(
        index="platform",
        columns="ordering",
        values="mean_words"
    )
    .reindex(
        eligible_platforms
    )
)


ax = length_chart.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Review Length: Relevant vs Newest"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Mean Words per Review"
)

ax.legend(
    title="Ordering"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "review_length_relevant_vs_newest.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 4
# NEGATIVE EXPOSURE RATIO
# ============================================================

ner_plot = (
    comparison_df
    .set_index(
        "platform"
    )
    .reindex(
        eligible_platforms
    )
)


ax = ner_plot[
    "negative_exposure_ratio"
].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.axhline(
    1.0,
    linewidth=1
)

ax.set_title(
    "Negative Exposure Ratio: Relevant vs Newest"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Negative Exposure Ratio"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "negative_exposure_ratio.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 5
# RATING GAP
# ============================================================

ax = ner_plot[
    "rating_gap_newest_minus_relevant"
].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "Rating Gap: Newest Minus Relevant"
)

ax.set_xlabel(
    "Platform"
)

ax.set_ylabel(
    "Star Rating Difference"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "rating_gap_newest_minus_relevant.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE EXCEL
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

    eligibility_df.to_excel(
        writer,
        sheet_name="Platform Eligibility",
        index=False
    )

    overall_df.to_excel(
        writer,
        sheet_name="Pooled Comparison",
        index=False
    )

    comparison_df.to_excel(
        writer,
        sheet_name="Platform Comparison",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="Ordering Summary",
        index=False
    )

    benchmark_df.to_excel(
        writer,
        sheet_name="Benchmark Reviews",
        index=False
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)

print(
    f"\nFixed benchmark:\n"
    f"{BENCHMARK_N:,} reviews "
    f"per ordering per platform"
)

print(
    "\nEligible platforms:"
)

print(
    ", ".join(
        eligible_platforms
    )
)

print(
    f"\nTotal balanced reviews analyzed:"
)

print(
    f"{len(benchmark_df):,}"
)


print(
    f"\nExcel output:\n"
    f"{OUTPUT_EXCEL.resolve()}"
)

print(
    f"\nCharts directory:\n"
    f"{OUTPUT_DIR.resolve()}"
)