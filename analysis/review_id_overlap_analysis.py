from pathlib import Path
import re

import numpy as np
import pandas as pd

from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

RELEVANT_DIR = Path("./relevant_reviews")
NEWEST_DIR = Path("./newest_reviews")

OUTPUT_DIR = Path("./review_overlap_analysis")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_EXCEL = (
    OUTPUT_DIR /
    "review_id_overlap_results.xlsx"
)

BENCHMARK_N = 1700

TOP_K_VALUES = [
    50,
    100,
    250,
    500,
    1000,
    1700,
]

REVIEW_ID_COL = "review_id"
ORDER_COL = "scrape_order"

RATING_COL = "rating"
TEXT_COL = "review_text"
HELPFUL_COL = "helpful_count"
DATE_COL = "review_date_iso"


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
# LOAD FILE
# ============================================================

def read_reviews(path):

    df = pd.read_excel(path)

    raw_n = len(df)

    if REVIEW_ID_COL not in df.columns:
        raise ValueError(
            f"{path.name}: missing '{REVIEW_ID_COL}' column."
        )

    # --------------------------------------------
    # Review ID cleanup
    # --------------------------------------------

    df[REVIEW_ID_COL] = (
        df[REVIEW_ID_COL]
        .astype("string")
        .str.strip()
    )

    invalid_id = (
        df[REVIEW_ID_COL].isna()
        |
        df[REVIEW_ID_COL].eq("")
        |
        df[REVIEW_ID_COL].str.lower().isin(
            ["nan", "none", "<na>"]
        )
    )

    invalid_id_n = int(
        invalid_id.sum()
    )

    df = df[
        ~invalid_id
    ].copy()

    # --------------------------------------------
    # Scrape order
    # --------------------------------------------

    if ORDER_COL in df.columns:

        df[ORDER_COL] = pd.to_numeric(
            df[ORDER_COL],
            errors="coerce"
        )

        df = df.sort_values(
            ORDER_COL,
            na_position="last"
        )

    else:

        df[ORDER_COL] = (
            np.arange(len(df)) + 1
        )

    # --------------------------------------------
    # Duplicate review IDs
    # --------------------------------------------

    duplicate_n = int(
        df.duplicated(
            REVIEW_ID_COL,
            keep=False
        ).sum()
    )

    unique_n = df[
        REVIEW_ID_COL
    ].nunique()

    diagnostics = {
        "file": path.name,
        "raw_rows": raw_n,
        "valid_id_rows": len(df),
        "invalid_id_rows": invalid_id_n,
        "unique_review_ids": unique_n,
        "rows_in_duplicate_ids": duplicate_n,
    }

    return (
        df.reset_index(drop=True),
        diagnostics
    )


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
# OVERLAP FUNCTION
# ============================================================

def calculate_overlap(
    platform,
    relevant,
    newest,
    sample_name
):

    rel_ids = set(
        relevant[
            REVIEW_ID_COL
        ].dropna()
    )

    new_ids = set(
        newest[
            REVIEW_ID_COL
        ].dropna()
    )

    shared = (
        rel_ids & new_ids
    )

    union = (
        rel_ids | new_ids
    )

    rel_n = len(rel_ids)
    new_n = len(new_ids)
    shared_n = len(shared)

    return {
        "platform": platform,
        "sample": sample_name,

        "relevant_unique_ids": rel_n,
        "newest_unique_ids": new_n,

        "shared_review_ids": shared_n,

        "overlap_percent_of_relevant":
            (
                shared_n / rel_n * 100
                if rel_n > 0
                else np.nan
            ),

        "overlap_percent_of_newest":
            (
                shared_n / new_n * 100
                if new_n > 0
                else np.nan
            ),

        "jaccard_percent":
            (
                shared_n / len(union) * 100
                if len(union) > 0
                else np.nan
            ),
    }


# ============================================================
# DISCOVER FILES
# ============================================================

relevant_files = {
    get_platform_from_filename(path): path
    for path in RELEVANT_DIR.glob("*.xlsx")
}

newest_files = {
    get_platform_from_filename(path): path
    for path in NEWEST_DIR.glob("*.xlsx")
}

platforms = sorted(
    set(relevant_files.keys())
    &
    set(newest_files.keys())
)


print("=" * 120)
print("REVIEW-ID OVERLAP ANALYSIS")
print("=" * 120)

print("\nMatched platforms:")

for platform in platforms:
    print(f"  {platform}")

if not platforms:
    raise RuntimeError(
        "No matching platform files found."
    )


# ============================================================
# LOAD
# ============================================================

datasets = {}
diagnostic_rows = []

for platform in platforms:

    relevant, rel_diag = read_reviews(
        relevant_files[platform]
    )

    newest, new_diag = read_reviews(
        newest_files[platform]
    )

    datasets[platform] = {
        "Relevant": relevant,
        "Newest": newest,
    }

    diagnostic_rows.append({
        "platform": platform,
        "ordering": "Relevant",
        **rel_diag
    })

    diagnostic_rows.append({
        "platform": platform,
        "ordering": "Newest",
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

    rel_n = len(
        datasets[
            platform
        ]["Relevant"]
    )

    new_n = len(
        datasets[
            platform
        ]["Newest"]
    )

    if (
        rel_n >= BENCHMARK_N
        and
        new_n >= BENCHMARK_N
    ):

        status = "ELIGIBLE"

        eligible_platforms.append(
            platform
        )

    else:

        status = "EXCLUDED"

    eligibility_rows.append({
        "platform": platform,
        "relevant_valid_n": rel_n,
        "newest_valid_n": new_n,
        "required_n": BENCHMARK_N,
        "status": status,
    })


eligibility_df = pd.DataFrame(
    eligibility_rows
)


print("\n")
print("=" * 120)
print("ID DIAGNOSTICS")
print("=" * 120)

print(
    diagnostics_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 120)
print("ELIGIBILITY")
print("=" * 120)

print(
    eligibility_df.to_string(
        index=False
    )
)


# ============================================================
# FULL-DATA OVERLAP
# ============================================================

full_overlap_rows = []

for platform in platforms:

    result = calculate_overlap(
        platform,
        datasets[platform]["Relevant"],
        datasets[platform]["Newest"],
        "All available reviews"
    )

    full_overlap_rows.append(
        result
    )


full_overlap_df = pd.DataFrame(
    full_overlap_rows
)


print("\n")
print("=" * 145)
print("FULL AVAILABLE REVIEW-ID OVERLAP")
print("=" * 145)

print(
    full_overlap_df.to_string(
        index=False,
        formatters={
            "overlap_percent_of_relevant":
                lambda x: f"{x:.2f}%",

            "overlap_percent_of_newest":
                lambda x: f"{x:.2f}%",

            "jaccard_percent":
                lambda x: f"{x:.2f}%",
        }
    )
)


# ============================================================
# BALANCED 1700 OVERLAP
# ============================================================

balanced_overlap_rows = []

balanced_data = {}


for platform in eligible_platforms:

    relevant = (
        datasets[
            platform
        ]["Relevant"]
        .head(BENCHMARK_N)
        .copy()
    )

    newest = (
        datasets[
            platform
        ]["Newest"]
        .head(BENCHMARK_N)
        .copy()
    )

    balanced_data[platform] = {
        "Relevant": relevant,
        "Newest": newest,
    }

    result = calculate_overlap(
        platform,
        relevant,
        newest,
        f"Top {BENCHMARK_N}"
    )

    balanced_overlap_rows.append(
        result
    )


balanced_overlap_df = pd.DataFrame(
    balanced_overlap_rows
)


print("\n")
print("=" * 145)
print("BALANCED TOP-1700 REVIEW-ID OVERLAP")
print("=" * 145)

print(
    balanced_overlap_df.to_string(
        index=False,
        formatters={
            "overlap_percent_of_relevant":
                lambda x: f"{x:.2f}%",

            "overlap_percent_of_newest":
                lambda x: f"{x:.2f}%",

            "jaccard_percent":
                lambda x: f"{x:.2f}%",
        }
    )
)


# ============================================================
# POOLED BALANCED OVERLAP
#
# Platform-prefixed IDs prevent accidental collisions
# between apps.
# ============================================================

pooled_rel_ids = set()
pooled_new_ids = set()

for platform in eligible_platforms:

    rel_ids = (
        balanced_data[
            platform
        ]["Relevant"][
            REVIEW_ID_COL
        ]
        .dropna()
        .astype(str)
    )

    new_ids = (
        balanced_data[
            platform
        ]["Newest"][
            REVIEW_ID_COL
        ]
        .dropna()
        .astype(str)
    )

    pooled_rel_ids.update(
        f"{platform}::{x}"
        for x in rel_ids
    )

    pooled_new_ids.update(
        f"{platform}::{x}"
        for x in new_ids
    )


pooled_shared = (
    pooled_rel_ids
    &
    pooled_new_ids
)

pooled_union = (
    pooled_rel_ids
    |
    pooled_new_ids
)


pooled_overlap_df = pd.DataFrame([
    {
        "platforms":
            len(eligible_platforms),

        "relevant_unique_ids":
            len(pooled_rel_ids),

        "newest_unique_ids":
            len(pooled_new_ids),

        "shared_review_ids":
            len(pooled_shared),

        "overlap_percent_of_relevant":
            (
                len(pooled_shared)
                /
                len(pooled_rel_ids)
                *
                100
            ),

        "overlap_percent_of_newest":
            (
                len(pooled_shared)
                /
                len(pooled_new_ids)
                *
                100
            ),

        "jaccard_percent":
            (
                len(pooled_shared)
                /
                len(pooled_union)
                *
                100
            ),
    }
])


print("\n")
print("=" * 120)
print("POOLED BALANCED REVIEW-ID OVERLAP")
print("=" * 120)

print(
    pooled_overlap_df.to_string(
        index=False
    )
)


# ============================================================
# TOP-K OVERLAP
# ============================================================

topk_rows = []


for platform in eligible_platforms:

    relevant_full = (
        balanced_data[
            platform
        ]["Relevant"]
    )

    newest_full = (
        balanced_data[
            platform
        ]["Newest"]
    )

    for k in TOP_K_VALUES:

        relevant = (
            relevant_full
            .head(k)
        )

        newest = (
            newest_full
            .head(k)
        )

        result = calculate_overlap(
            platform,
            relevant,
            newest,
            f"Top {k}"
        )

        result[
            "top_k"
        ] = k

        topk_rows.append(
            result
        )


topk_df = pd.DataFrame(
    topk_rows
)


print("\n")
print("=" * 145)
print("TOP-K REVIEW-ID OVERLAP")
print("=" * 145)

display_topk = topk_df[
    [
        "platform",
        "top_k",
        "shared_review_ids",
        "overlap_percent_of_relevant",
        "overlap_percent_of_newest",
        "jaccard_percent",
    ]
]


print(
    display_topk.to_string(
        index=False,
        formatters={
            "overlap_percent_of_relevant":
                lambda x: f"{x:.2f}%",

            "overlap_percent_of_newest":
                lambda x: f"{x:.2f}%",

            "jaccard_percent":
                lambda x: f"{x:.2f}%",
        }
    )
)


# ============================================================
# SHARED REVIEW RANK ANALYSIS
# ============================================================

rank_rows = []

shared_review_rows = []


for platform in eligible_platforms:

    relevant = (
        balanced_data[
            platform
        ]["Relevant"]
        .copy()
    )

    newest = (
        balanced_data[
            platform
        ]["Newest"]
        .copy()
    )

    # One row per ID
    relevant_unique = (
        relevant
        .drop_duplicates(
            REVIEW_ID_COL,
            keep="first"
        )
    )

    newest_unique = (
        newest
        .drop_duplicates(
            REVIEW_ID_COL,
            keep="first"
        )
    )


    rel_columns = [
        REVIEW_ID_COL,
        ORDER_COL,
    ]

    new_columns = [
        REVIEW_ID_COL,
        ORDER_COL,
    ]


    for optional in [
        RATING_COL,
        TEXT_COL,
        HELPFUL_COL,
        DATE_COL,
    ]:

        if optional in relevant_unique.columns:
            rel_columns.append(
                optional
            )

        if optional in newest_unique.columns:
            new_columns.append(
                optional
            )


    rel_temp = (
        relevant_unique[
            rel_columns
        ]
        .copy()
    )

    new_temp = (
        newest_unique[
            new_columns
        ]
        .copy()
    )


    rename_rel = {
        ORDER_COL:
            "relevant_rank",
    }

    rename_new = {
        ORDER_COL:
            "newest_rank",
    }


    for optional in [
        RATING_COL,
        TEXT_COL,
        HELPFUL_COL,
        DATE_COL,
    ]:

        if optional in rel_temp.columns:
            rename_rel[
                optional
            ] = (
                f"relevant_{optional}"
            )

        if optional in new_temp.columns:
            rename_new[
                optional
            ] = (
                f"newest_{optional}"
            )


    rel_temp = rel_temp.rename(
        columns=rename_rel
    )

    new_temp = new_temp.rename(
        columns=rename_new
    )


    shared = rel_temp.merge(
        new_temp,
        on=REVIEW_ID_COL,
        how="inner"
    )


    shared[
        "platform"
    ] = platform


    if len(shared) > 0:

        shared[
            "absolute_rank_difference"
        ] = (
            shared[
                "relevant_rank"
            ]
            -
            shared[
                "newest_rank"
            ]
        ).abs()


        rho, p = safe_spearman(
            shared[
                "relevant_rank"
            ],
            shared[
                "newest_rank"
            ]
        )


        rank_rows.append({
            "platform":
                platform,

            "shared_review_ids":
                len(shared),

            "median_relevant_rank":
                shared[
                    "relevant_rank"
                ].median(),

            "median_newest_rank":
                shared[
                    "newest_rank"
                ].median(),

            "median_absolute_rank_difference":
                shared[
                    "absolute_rank_difference"
                ].median(),

            "mean_absolute_rank_difference":
                shared[
                    "absolute_rank_difference"
                ].mean(),

            "rank_spearman_rho":
                rho,

            "rank_spearman_p":
                p,
        })


        shared_review_rows.append(
            shared
        )

    else:

        rank_rows.append({
            "platform":
                platform,

            "shared_review_ids":
                0,

            "median_relevant_rank":
                np.nan,

            "median_newest_rank":
                np.nan,

            "median_absolute_rank_difference":
                np.nan,

            "mean_absolute_rank_difference":
                np.nan,

            "rank_spearman_rho":
                np.nan,

            "rank_spearman_p":
                np.nan,
        })


rank_df = pd.DataFrame(
    rank_rows
)


if shared_review_rows:

    shared_reviews_df = pd.concat(
        shared_review_rows,
        ignore_index=True
    )

else:

    shared_reviews_df = pd.DataFrame()


print("\n")
print("=" * 140)
print("SHARED REVIEW RANK COMPARISON")
print("=" * 140)

print(
    rank_df.to_string(
        index=False,
        formatters={
            "rank_spearman_rho":
                lambda x:
                f"{x:+.3f}"
                if pd.notna(x)
                else "NaN",

            "rank_spearman_p":
                lambda x:
                f"{x:.3e}"
                if pd.notna(x)
                else "NaN",
        }
    )
)


# ============================================================
# SAVE
# ============================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    diagnostics_df.to_excel(
        writer,
        sheet_name="ID Diagnostics",
        index=False
    )

    eligibility_df.to_excel(
        writer,
        sheet_name="Eligibility",
        index=False
    )

    full_overlap_df.to_excel(
        writer,
        sheet_name="Full Overlap",
        index=False
    )

    balanced_overlap_df.to_excel(
        writer,
        sheet_name="Balanced Overlap",
        index=False
    )

    pooled_overlap_df.to_excel(
        writer,
        sheet_name="Pooled Overlap",
        index=False
    )

    topk_df.to_excel(
        writer,
        sheet_name="TopK Overlap",
        index=False
    )

    rank_df.to_excel(
        writer,
        sheet_name="Shared Rank Comparison",
        index=False
    )

    if not shared_reviews_df.empty:

        shared_reviews_df.to_excel(
            writer,
            sheet_name="Shared Reviews",
            index=False
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 120)
print("REVIEW-ID OVERLAP ANALYSIS COMPLETE")
print("=" * 120)

print(
    f"\nPlatforms analyzed: "
    f"{len(eligible_platforms)}"
)

print(
    f"\nExcel output:\n"
    f"{OUTPUT_EXCEL.resolve()}"
)