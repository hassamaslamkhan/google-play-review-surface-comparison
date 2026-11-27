from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

RELEVANT_DIR = Path("./relevant_reviews")
NEWEST_DIR = Path("./newest_reviews")

OUTPUT_DIR = Path("./topk_visibility_gradient_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EXCEL = (
    OUTPUT_DIR /
    "topk_visibility_gradient_results.xlsx"
)

# Top-k cutoffs
TOP_K_VALUES = [
    50,
    100,
    250,
    500,
    1000,
    1700,
]

MAX_K = max(TOP_K_VALUES)

RATING_COL = "rating"
TEXT_COL = "review_text"
HELPFUL_COL = "helpful_count"
ORDER_COL = "scrape_order"

NEGATIVE_THRESHOLD = 3


# ============================================================
# PLATFORM NAME
# ============================================================

def get_platform_from_filename(path):

    name = path.stem.lower().strip()

    # Remove Newest prefix
    name = re.sub(
        r"^newest[_\-\s]*",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove normal suffix
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

        rating = float(value)

        if 1 <= rating <= 5:
            return rating

        return np.nan


    text = str(value).strip()

    if not text:
        return np.nan


    # Direct numeric
    try:

        rating = float(text)

        if 1 <= rating <= 5:
            return rating

    except Exception:
        pass


    # Text ratings
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

    df = pd.read_excel(path)

    if RATING_COL not in df.columns:

        raise ValueError(
            f"{path.name}: missing "
            f"'{RATING_COL}'"
        )

    if TEXT_COL not in df.columns:

        raise ValueError(
            f"{path.name}: missing "
            f"'{TEXT_COL}'"
        )


    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    df[RATING_COL] = (
        df[RATING_COL]
        .apply(parse_rating)
    )


    # --------------------------------------------------------
    # Review text
    # --------------------------------------------------------

    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Word count
    # --------------------------------------------------------

    df["review_words"] = (
        df[TEXT_COL]
        .str.split()
        .str.len()
    )


    # --------------------------------------------------------
    # Helpful
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
            f"{ORDER_COL}. Using row order."
        )

        df[ORDER_COL] = (
            np.arange(len(df)) + 1
        )


    # --------------------------------------------------------
    # Valid ratings only
    # --------------------------------------------------------

    df = df[
        df[RATING_COL].notna()
    ].copy()


    # --------------------------------------------------------
    # Negative
    # --------------------------------------------------------

    df["is_negative"] = (
        df[RATING_COL] <
        NEGATIVE_THRESHOLD
    )


    return (
        df
        .reset_index(drop=True)
    )


# ============================================================
# SAMPLE SUMMARY
# ============================================================

def summarize_sample(
    platform,
    ordering,
    k,
    df
):

    n = len(df)

    negative_n = int(
        df["is_negative"].sum()
    )

    negative_pct = (
        negative_n / n * 100
        if n > 0
        else np.nan
    )


    return {

        "platform":
            platform,

        "ordering":
            ordering,

        "top_k":
            k,

        "n":
            n,

        # Rating
        "mean_rating":
            df[RATING_COL].mean(),

        "median_rating":
            df[RATING_COL].median(),

        "negative_n":
            negative_n,

        "negative_percent":
            negative_pct,

        "one_star_percent":
            (
                (df[RATING_COL] == 1)
                .mean() * 100
            ),

        "two_star_percent":
            (
                (df[RATING_COL] == 2)
                .mean() * 100
            ),

        "four_five_star_percent":
            (
                (df[RATING_COL] >= 4)
                .mean() * 100
            ),

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
    }


# ============================================================
# SAFE SPEARMAN
# ============================================================

def safe_spearman(x, y):

    temp = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()

    if len(temp) < 3:

        return np.nan, np.nan

    if (
        temp["x"].nunique() < 2
        or
        temp["y"].nunique() < 2
    ):

        return np.nan, np.nan


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


common_platforms = sorted(
    set(relevant_files.keys())
    &
    set(newest_files.keys())
)


print("=" * 110)
print("TOP-K REVIEW VISIBILITY GRADIENT ANALYSIS")
print("=" * 110)

print(
    "\nTop-k values:",
    TOP_K_VALUES
)

print(
    "\nMatched platforms:"
)

for platform in common_platforms:

    print(
        f"  {platform}"
    )


if not common_platforms:

    raise RuntimeError(
        "No matched platforms found."
    )


# ============================================================
# LOAD AND VALIDATE
# ============================================================

datasets = {}

eligibility_rows = []


for platform in common_platforms:

    relevant = read_review_file(
        relevant_files[platform]
    )

    newest = read_review_file(
        newest_files[platform]
    )


    datasets[platform] = {

        "Relevant":
            relevant,

        "Newest":
            newest,
    }


    eligible = (
        len(relevant) >= MAX_K
        and
        len(newest) >= MAX_K
    )


    eligibility_rows.append({

        "platform":
            platform,

        "relevant_valid_n":
            len(relevant),

        "newest_valid_n":
            len(newest),

        "required_n":
            MAX_K,

        "status":
            (
                "ELIGIBLE"
                if eligible
                else "EXCLUDED"
            )
    })


eligibility_df = pd.DataFrame(
    eligibility_rows
)


print("\n")
print("=" * 110)
print("ELIGIBILITY")
print("=" * 110)

print(
    eligibility_df.to_string(
        index=False
    )
)


eligible_platforms = (

    eligibility_df[
        eligibility_df[
            "status"
        ] == "ELIGIBLE"
    ][
        "platform"
    ]
    .tolist()
)


if not eligible_platforms:

    raise RuntimeError(
        "No platform has at least "
        f"{MAX_K} reviews in both samples."
    )


# ============================================================
# TOP-K ANALYSIS
# ============================================================

summary_rows = []

comparison_rows = []

pooled_rows = []


for platform in eligible_platforms:

    relevant_full = (
        datasets[
            platform
        ]["Relevant"]
        .head(MAX_K)
        .copy()
    )

    newest_full = (
        datasets[
            platform
        ]["Newest"]
        .head(MAX_K)
        .copy()
    )


    for k in TOP_K_VALUES:

        relevant = (
            relevant_full
            .head(k)
            .copy()
        )

        newest = (
            newest_full
            .head(k)
            .copy()
        )


        # ----------------------------------------------------
        # Summaries
        # ----------------------------------------------------

        relevant_summary = summarize_sample(
            platform,
            "Relevant",
            k,
            relevant
        )

        newest_summary = summarize_sample(
            platform,
            "Newest",
            k,
            newest
        )


        summary_rows.extend(
            [
                relevant_summary,
                newest_summary,
            ]
        )


        # ----------------------------------------------------
        # Exposure metrics
        # ----------------------------------------------------

        rel_negative_rate = (
            relevant[
                "is_negative"
            ].mean()
        )

        new_negative_rate = (
            newest[
                "is_negative"
            ].mean()
        )


        negative_exposure_ratio = (

            rel_negative_rate
            /
            new_negative_rate

            if new_negative_rate > 0
            else np.nan
        )


        # Newest - Relevant
        rating_gap = (
            newest[
                RATING_COL
            ].mean()
            -
            relevant[
                RATING_COL
            ].mean()
        )


        negative_pp_gap = (
            rel_negative_rate
            -
            new_negative_rate
        ) * 100


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


        length_ratio = (

            relevant_mean_words
            /
            newest_mean_words

            if newest_mean_words > 0
            else np.nan
        )


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


        comparison_rows.append({

            "platform":
                platform,

            "top_k":
                k,

            # Rating
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

            # Negative
            "relevant_negative_percent":
                rel_negative_rate * 100,

            "newest_negative_percent":
                new_negative_rate * 100,

            "negative_percentage_point_gap":
                negative_pp_gap,

            "negative_exposure_ratio":
                negative_exposure_ratio,

            # Length
            "relevant_mean_words":
                relevant_mean_words,

            "newest_mean_words":
                newest_mean_words,

            "review_length_ratio":
                length_ratio,

            "relevant_median_words":
                relevant[
                    "review_words"
                ].median(),

            "newest_median_words":
                newest[
                    "review_words"
                ].median(),

            # Helpful
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
        })


# ============================================================
# DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# POOLED TOP-K ANALYSIS
# Equal contribution from every platform
# ============================================================

pooled_rows = []


for k in TOP_K_VALUES:

    relevant_frames = []
    newest_frames = []


    for platform in eligible_platforms:

        rel = (
            datasets[
                platform
            ]["Relevant"]
            .head(k)
            .copy()
        )

        new = (
            datasets[
                platform
            ]["Newest"]
            .head(k)
            .copy()
        )


        rel["platform"] = platform
        new["platform"] = platform


        relevant_frames.append(
            rel
        )

        newest_frames.append(
            new
        )


    relevant_pool = pd.concat(
        relevant_frames,
        ignore_index=True
    )

    newest_pool = pd.concat(
        newest_frames,
        ignore_index=True
    )


    rel_neg = (
        relevant_pool[
            "is_negative"
        ].mean()
    )

    new_neg = (
        newest_pool[
            "is_negative"
        ].mean()
    )


    ner = (

        rel_neg / new_neg

        if new_neg > 0
        else np.nan
    )


    relevant_words = (
        relevant_pool[
            "review_words"
        ].mean()
    )

    newest_words = (
        newest_pool[
            "review_words"
        ].mean()
    )


    length_ratio = (

        relevant_words
        /
        newest_words

        if newest_words > 0
        else np.nan
    )


    pooled_rows.append({

        "top_k_per_platform":
            k,

        "platform_count":
            len(
                eligible_platforms
            ),

        "relevant_n":
            len(
                relevant_pool
            ),

        "newest_n":
            len(
                newest_pool
            ),

        # Rating
        "relevant_mean_rating":
            relevant_pool[
                RATING_COL
            ].mean(),

        "newest_mean_rating":
            newest_pool[
                RATING_COL
            ].mean(),

        "rating_gap_newest_minus_relevant":
            (
                newest_pool[
                    RATING_COL
                ].mean()
                -
                relevant_pool[
                    RATING_COL
                ].mean()
            ),

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
            ner,

        # Length
        "relevant_mean_words":
            relevant_words,

        "newest_mean_words":
            newest_words,

        "review_length_ratio":
            length_ratio,

        "relevant_median_words":
            relevant_pool[
                "review_words"
            ].median(),

        "newest_median_words":
            newest_pool[
                "review_words"
            ].median(),

        # Helpful
        "relevant_mean_helpful":
            relevant_pool[
                HELPFUL_COL
            ].mean(),

        "newest_mean_helpful":
            newest_pool[
                HELPFUL_COL
            ].mean(),

        "relevant_median_helpful":
            relevant_pool[
                HELPFUL_COL
            ].median(),

        "newest_median_helpful":
            newest_pool[
                HELPFUL_COL
            ].median(),
    })


pooled_df = pd.DataFrame(
    pooled_rows
)


# ============================================================
# WITHIN-RELEVANT GRADIENT
#
# Compare Top-k Relevant against Top-1700 Relevant
# ============================================================

relevant_gradient_rows = []


for platform in eligible_platforms:

    full = (
        datasets[
            platform
        ]["Relevant"]
        .head(MAX_K)
        .copy()
    )


    baseline_negative = (
        full[
            "is_negative"
        ].mean()
    )

    baseline_rating = (
        full[
            RATING_COL
        ].mean()
    )

    baseline_words = (
        full[
            "review_words"
        ].mean()
    )

    baseline_helpful = (
        full[
            HELPFUL_COL
        ].median()
    )


    for k in TOP_K_VALUES:

        sample = (
            full
            .head(k)
            .copy()
        )


        current_negative = (
            sample[
                "is_negative"
            ].mean()
        )

        current_rating = (
            sample[
                RATING_COL
            ].mean()
        )

        current_words = (
            sample[
                "review_words"
            ].mean()
        )

        current_helpful = (
            sample[
                HELPFUL_COL
            ].median()
        )


        relevant_gradient_rows.append({

            "platform":
                platform,

            "top_k":
                k,

            "negative_percent":
                current_negative * 100,

            "negative_pp_vs_top1700":
                (
                    current_negative
                    -
                    baseline_negative
                ) * 100,

            "mean_rating":
                current_rating,

            "rating_difference_vs_top1700":
                (
                    current_rating
                    -
                    baseline_rating
                ),

            "mean_words":
                current_words,

            "word_ratio_vs_top1700":
                (
                    current_words
                    /
                    baseline_words

                    if baseline_words > 0
                    else np.nan
                ),

            "median_helpful":
                current_helpful,

            "top1700_median_helpful":
                baseline_helpful,
        })


relevant_gradient_df = pd.DataFrame(
    relevant_gradient_rows
)


# ============================================================
# SPEARMAN RANK CORRELATIONS
#
# Interpretation:
#
# scrape_order vs rating:
#   positive -> ratings rise farther down ranking
#               top is more negative
#
# scrape_order vs word count:
#   negative -> longer reviews near top
#
# scrape_order vs helpful:
#   negative -> more helpful reviews near top
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
            .head(MAX_K)
            .copy()
        )


        # Rating
        rating_rho, rating_p = (
            safe_spearman(
                df[ORDER_COL],
                df[RATING_COL]
            )
        )


        # Word count
        words_rho, words_p = (
            safe_spearman(
                df[ORDER_COL],
                df["review_words"]
            )
        )


        # Helpful
        helpful_rho, helpful_p = (
            safe_spearman(
                df[ORDER_COL],
                np.log1p(
                    df[
                        HELPFUL_COL
                    ]
                )
            )
        )


        correlation_rows.append({

            "platform":
                platform,

            "ordering":
                ordering,

            "n":
                len(df),

            "rank_vs_rating_rho":
                rating_rho,

            "rank_vs_rating_p":
                rating_p,

            "rank_vs_words_rho":
                words_rho,

            "rank_vs_words_p":
                words_p,

            "rank_vs_log_helpful_rho":
                helpful_rho,

            "rank_vs_log_helpful_p":
                helpful_p,
        })


correlation_df = pd.DataFrame(
    correlation_rows
)


# ============================================================
# PRINT PLATFORM TOP-K RESULTS
# ============================================================

print("\n")
print("=" * 155)
print("PLATFORM TOP-K COMPARISON")
print("=" * 155)


display_cols = [

    "platform",
    "top_k",

    "relevant_mean_rating",
    "newest_mean_rating",

    "relevant_negative_percent",
    "newest_negative_percent",

    "negative_exposure_ratio",

    "relevant_mean_words",
    "newest_mean_words",

    "review_length_ratio",

    "relevant_median_helpful",
    "newest_median_helpful",
]


print(
    comparison_df[
        display_cols
    ].to_string(

        index=False,

        formatters={

            "relevant_mean_rating":
                lambda x:
                f"{x:.2f}",

            "newest_mean_rating":
                lambda x:
                f"{x:.2f}",

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
        }
    )
)


# ============================================================
# PRINT POOLED RESULTS
# ============================================================

print("\n")
print("=" * 150)
print("POOLED TOP-K GRADIENT — ALL PLATFORMS")
print("=" * 150)


pooled_display = [

    "top_k_per_platform",

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
]


print(
    pooled_df[
        pooled_display
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
        }
    )
)


# ============================================================
# PRINT RANK CORRELATIONS
# ============================================================

print("\n")
print("=" * 145)
print("SCRAPE-ORDER CORRELATIONS")
print("=" * 145)


print(
    correlation_df.to_string(

        index=False,

        formatters={

            "rank_vs_rating_rho":
                lambda x:
                f"{x:+.3f}",

            "rank_vs_rating_p":
                lambda x:
                f"{x:.3e}",

            "rank_vs_words_rho":
                lambda x:
                f"{x:+.3f}",

            "rank_vs_words_p":
                lambda x:
                f"{x:.3e}",

            "rank_vs_log_helpful_rho":
                lambda x:
                f"{x:+.3f}",

            "rank_vs_log_helpful_p":
                lambda x:
                f"{x:.3e}",
        }
    )
)


# ============================================================
# CHART 1
# POOLED NEGATIVE SHARE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    pooled_df[
        "top_k_per_platform"
    ],
    pooled_df[
        "relevant_negative_percent"
    ],
    marker="o",
    label="Relevant"
)

plt.plot(
    pooled_df[
        "top_k_per_platform"
    ],
    pooled_df[
        "newest_negative_percent"
    ],
    marker="o",
    label="Newest"
)

plt.xlabel(
    "Top-k Reviews per Platform"
)

plt.ylabel(
    "Negative Reviews (%)"
)

plt.title(
    "Negative Review Share by Review Depth"
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "pooled_negative_share_topk.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 2
# NEGATIVE EXPOSURE RATIO
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    pooled_df[
        "top_k_per_platform"
    ],
    pooled_df[
        "negative_exposure_ratio"
    ],
    marker="o"
)

plt.axhline(
    y=1,
    linewidth=1
)

plt.xlabel(
    "Top-k Reviews per Platform"
)

plt.ylabel(
    "Negative Exposure Ratio"
)

plt.title(
    "Negative Exposure Ratio by Review Depth"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "pooled_negative_exposure_ratio_topk.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 3
# MEAN RATING
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    pooled_df[
        "top_k_per_platform"
    ],
    pooled_df[
        "relevant_mean_rating"
    ],
    marker="o",
    label="Relevant"
)

plt.plot(
    pooled_df[
        "top_k_per_platform"
    ],
    pooled_df[
        "newest_mean_rating"
    ],
    marker="o",
    label="Newest"
)

plt.xlabel(
    "Top-k Reviews per Platform"
)

plt.ylabel(
    "Mean Star Rating"
)

plt.title(
    "Mean Rating by Review Depth"
)

plt.ylim(
    1,
    5
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "pooled_mean_rating_topk.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 4
# REVIEW LENGTH
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    pooled_df[
        "top_k_per_platform"
    ],
    pooled_df[
        "relevant_mean_words"
    ],
    marker="o",
    label="Relevant"
)

plt.plot(
    pooled_df[
        "top_k_per_platform"
    ],
    pooled_df[
        "newest_mean_words"
    ],
    marker="o",
    label="Newest"
)

plt.xlabel(
    "Top-k Reviews per Platform"
)

plt.ylabel(
    "Mean Words per Review"
)

plt.title(
    "Review Length by Review Depth"
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "pooled_review_length_topk.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 5
# PLATFORM-SPECIFIC NER
# ============================================================

plt.figure(
    figsize=(10, 7)
)


for platform in eligible_platforms:

    temp = (
        comparison_df[
            comparison_df[
                "platform"
            ] == platform
        ]
        .sort_values(
            "top_k"
        )
    )

    plt.plot(
        temp[
            "top_k"
        ],
        temp[
            "negative_exposure_ratio"
        ],
        marker="o",
        label=platform
    )


plt.axhline(
    y=1,
    linewidth=1
)

plt.xlabel(
    "Top-k Reviews"
)

plt.ylabel(
    "Negative Exposure Ratio"
)

plt.title(
    "Negative Exposure Ratio by Platform and Review Depth"
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "platform_negative_exposure_ratio_topk.png",
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


    eligibility_df.to_excel(
        writer,
        sheet_name="Eligibility",
        index=False
    )


    comparison_df.to_excel(
        writer,
        sheet_name="Platform TopK Comparison",
        index=False
    )


    pooled_df.to_excel(
        writer,
        sheet_name="Pooled TopK",
        index=False
    )


    summary_df.to_excel(
        writer,
        sheet_name="Ordering TopK Summary",
        index=False
    )


    relevant_gradient_df.to_excel(
        writer,
        sheet_name="Relevant Gradient",
        index=False
    )


    correlation_df.to_excel(
        writer,
        sheet_name="Rank Correlations",
        index=False
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 110)
print("TOP-K ANALYSIS COMPLETE")
print("=" * 110)

print(
    f"\nPlatforms analyzed: "
    f"{len(eligible_platforms)}"
)

print(
    "\nTop-k values:"
)

print(
    TOP_K_VALUES
)

print(
    f"\nExcel output:\n"
    f"{OUTPUT_EXCEL.resolve()}"
)

print(
    f"\nCharts:\n"
    f"{OUTPUT_DIR.resolve()}"
)