from pathlib import Path
import pandas as pd
import numpy as np
import sys

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(".")
TOP_N = 1700

OUTDIR = ROOT / "major_revision_analysis" / "review_length_distribution_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

FILES = {
    ("Facebook", "Relevant"):
        ROOT / "relevant_reviews" / "facebook_google_play_reviews.xlsx",

    ("Instagram", "Relevant"):
        ROOT / "relevant_reviews" / "instagram_google_play_reviews.xlsx",

    ("Snapchat", "Relevant"):
        ROOT / "relevant_reviews" / "snapchat_google_play_reviews.xlsx",

    ("TikTok", "Relevant"):
        ROOT / "relevant_reviews" / "tiktok_google_play_reviews.xlsx",

    ("X", "Relevant"):
        ROOT / "relevant_reviews" / "x_google_play_reviews.xlsx",

    ("YouTube", "Relevant"):
        ROOT / "relevant_reviews" / "youtube_google_play_reviews.xlsx",

    ("Facebook", "Newest"):
        ROOT / "newest_reviews" / "Newest_facebook_google_play_reviews.xlsx",

    ("Instagram", "Newest"):
        ROOT / "newest_reviews" / "Newest_instagram_google_play_reviews.xlsx",

    ("Snapchat", "Newest"):
        ROOT / "newest_reviews" / "Newest_snapchat_google_play_reviews.xlsx",

    ("TikTok", "Newest"):
        ROOT / "newest_reviews" / "Newest_tiktok_google_play_reviews.xlsx",

    ("X", "Newest"):
        ROOT / "newest_reviews" / "Newest_x_google_play_reviews.xlsx",

    ("YouTube", "Newest"):
        ROOT / "newest_reviews" / "Newest_youtube_google_play_reviews.xlsx",
}

# Existing manuscript values.
# These act as a validation gate.
EXPECTED_MEANS = {
    ("Facebook", "Relevant"): 65.7,
    ("Facebook", "Newest"): 9.4,

    ("Instagram", "Relevant"): 70.0,
    ("Instagram", "Newest"): 9.8,

    ("Snapchat", "Relevant"): 54.0,
    ("Snapchat", "Newest"): 12.0,

    ("TikTok", "Relevant"): 58.2,
    ("TikTok", "Newest"): 11.4,

    ("X", "Relevant"): 57.3,
    ("X", "Newest"): 10.0,

    ("YouTube", "Relevant"): 73.2,
    ("YouTube", "Newest"): 14.3,
}

# Allow only normal one-decimal rounding differences.
MEAN_TOLERANCE = 0.11


# ============================================================
# WORD COUNT
# ============================================================

def whitespace_word_count(value):
    """
    Number of whitespace-separated tokens.
    Blank or missing review text is treated as zero words.
    """
    if pd.isna(value):
        return 0

    text = str(value).strip()

    if not text:
        return 0

    return len(text.split())


# ============================================================
# LOAD + VALIDATE
# ============================================================

all_balanced = []
summary_rows = []
validation_failed = False

print("=" * 90)
print("REVIEW LENGTH DISTRIBUTION ANALYSIS V1")
print("=" * 90)
print(f"Python: {sys.version.split()[0]}")
print(f"pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Balanced depth: Top-{TOP_N} per application/order")
print()

for (app, ordering), path in FILES.items():

    print("-" * 90)
    print(f"{app} | {ordering}")
    print(f"File: {path}")

    if not path.exists():
        print("ERROR: FILE NOT FOUND")
        validation_failed = True
        continue

    df = pd.read_excel(path)

    required = {
        "scrape_order",
        "review_id",
        "rating",
        "review_text",
    }

    missing = required - set(df.columns)

    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}")
        validation_failed = True
        continue

    print(f"Raw rows: {len(df):,}")

    # --------------------------------------------------------
    # Validate scrape order
    # --------------------------------------------------------

    df["scrape_order_numeric"] = pd.to_numeric(
        df["scrape_order"],
        errors="coerce"
    )

    bad_order = int(df["scrape_order_numeric"].isna().sum())

    if bad_order:
        print(f"ERROR: {bad_order} invalid scrape_order values")
        validation_failed = True
        continue

    df = df.sort_values(
        "scrape_order_numeric",
        kind="stable"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Diagnose duplicate IDs
    # --------------------------------------------------------

    valid_ids = df["review_id"].dropna().astype(str)

    duplicate_id_rows = int(valid_ids.duplicated().sum())

    print(f"Duplicate review-ID rows: {duplicate_id_rows}")

    if duplicate_id_rows != 0:
        print(
            "ERROR: duplicate review IDs found. "
            "Do not silently deduplicate because that could alter "
            "the previously frozen Top-1700 sample."
        )
        validation_failed = True
        continue

    if len(df) < TOP_N:
        print(
            f"ERROR: only {len(df)} rows available; "
            f"need at least {TOP_N}."
        )
        validation_failed = True
        continue

    # --------------------------------------------------------
    # Frozen Top-1700 selection
    # --------------------------------------------------------

    balanced = df.iloc[:TOP_N].copy()

    # Diagnostics
    blank_text = (
        balanced["review_text"].isna()
        |
        balanced["review_text"].astype(str).str.strip().eq("")
    )

    blank_count = int(blank_text.sum())

    print(f"Selected rows: {len(balanced):,}")
    print(f"Blank/missing review_text in Top-{TOP_N}: {blank_count}")

    # --------------------------------------------------------
    # Exact manuscript definition:
    # whitespace-separated word count
    # --------------------------------------------------------

    balanced["word_count"] = (
        balanced["review_text"]
        .apply(whitespace_word_count)
        .astype(int)
    )

    mean_words = balanced["word_count"].mean()
    median_words = balanced["word_count"].median()

    q1 = balanced["word_count"].quantile(
        0.25,
        interpolation="linear"
    )

    q3 = balanced["word_count"].quantile(
        0.75,
        interpolation="linear"
    )

    iqr = q3 - q1

    minimum = balanced["word_count"].min()
    maximum = balanced["word_count"].max()

    expected = EXPECTED_MEANS[(app, ordering)]
    difference = mean_words - expected

    matches = abs(difference) <= MEAN_TOLERANCE

    print(f"Calculated mean: {mean_words:.4f}")
    print(f"Expected manuscript mean: {expected:.1f}")
    print(f"Difference: {difference:+.4f}")
    print(f"Validation: {'PASS' if matches else 'FAIL'}")

    print(
        f"Median={median_words:.2f}, "
        f"Q1={q1:.2f}, "
        f"Q3={q3:.2f}, "
        f"IQR={iqr:.2f}"
    )

    if not matches:
        validation_failed = True

    summary_rows.append({
        "application": app,
        "ordering": ordering,
        "n": len(balanced),
        "blank_or_missing_text": blank_count,
        "mean_words": mean_words,
        "expected_mean_words": expected,
        "mean_difference": difference,
        "mean_validation_pass": matches,
        "median_words": median_words,
        "q1_words": q1,
        "q3_words": q3,
        "iqr_words": iqr,
        "min_words": minimum,
        "max_words": maximum,
        "source_file": str(path),
    })

    balanced["application"] = app
    balanced["ordering"] = ordering

    all_balanced.append(
        balanced[
            [
                "application",
                "ordering",
                "scrape_order",
                "review_id",
                "rating",
                "word_count",
            ]
        ].copy()
    )


# ============================================================
# VALIDATION GATE
# ============================================================

print()
print("=" * 90)
print("VALIDATION GATE")
print("=" * 90)

summary = pd.DataFrame(summary_rows)

if len(summary) != 12:
    print(f"FAIL: expected 12 application/order datasets, got {len(summary)}")
    validation_failed = True

if validation_failed:
    print()
    print("VALIDATION FAILED.")
    print(
        "Do NOT use median/IQR results in the manuscript yet. "
        "At least one source did not reproduce the previously "
        "reported mean word count."
    )

    summary.to_csv(
        OUTDIR / "review_length_validation_failed.csv",
        index=False
    )

    print(
        "\nDiagnostic file saved to:",
        OUTDIR / "review_length_validation_failed.csv"
    )

    sys.exit(1)

print("PASS: all 12 source files reproduced the manuscript means.")

# ============================================================
# POOLED DESCRIPTIVE DISTRIBUTIONS
# ============================================================

combined = pd.concat(all_balanced, ignore_index=True)

pooled_rows = []

for ordering in ["Relevant", "Newest"]:

    subset = combined.loc[
        combined["ordering"] == ordering,
        "word_count"
    ]

    q1 = subset.quantile(0.25, interpolation="linear")
    q3 = subset.quantile(0.75, interpolation="linear")

    pooled_rows.append({
        "application": "Pooled",
        "ordering": ordering,
        "n": len(subset),
        "blank_or_missing_text": np.nan,
        "mean_words": subset.mean(),
        "expected_mean_words": np.nan,
        "mean_difference": np.nan,
        "mean_validation_pass": True,
        "median_words": subset.median(),
        "q1_words": q1,
        "q3_words": q3,
        "iqr_words": q3 - q1,
        "min_words": subset.min(),
        "max_words": subset.max(),
        "source_file": "Six balanced Top-1700 samples",
    })

pooled = pd.DataFrame(pooled_rows)

final_summary = pd.concat(
    [summary, pooled],
    ignore_index=True
)


# ============================================================
# MANUSCRIPT-FRIENDLY TABLE
# ============================================================

wide_rows = []

apps_order = [
    "Facebook",
    "Instagram",
    "Snapchat",
    "TikTok",
    "X",
    "YouTube",
    "Pooled",
]

for app in apps_order:

    r = final_summary[
        (final_summary["application"] == app)
        &
        (final_summary["ordering"] == "Relevant")
    ].iloc[0]

    n = final_summary[
        (final_summary["application"] == app)
        &
        (final_summary["ordering"] == "Newest")
    ].iloc[0]

    wide_rows.append({
        "application": app,

        "relevant_n": int(r["n"]),
        "relevant_mean": r["mean_words"],
        "relevant_median": r["median_words"],
        "relevant_q1": r["q1_words"],
        "relevant_q3": r["q3_words"],
        "relevant_iqr": r["iqr_words"],

        "newest_n": int(n["n"]),
        "newest_mean": n["mean_words"],
        "newest_median": n["median_words"],
        "newest_q1": n["q1_words"],
        "newest_q3": n["q3_words"],
        "newest_iqr": n["iqr_words"],
    })

wide = pd.DataFrame(wide_rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

summary_path = OUTDIR / "review_length_distribution_long.csv"
wide_path = OUTDIR / "review_length_distribution_manuscript.csv"
balanced_path = OUTDIR / "balanced_review_word_counts.csv"
excel_path = OUTDIR / "review_length_distribution_v1.xlsx"

final_summary.to_csv(summary_path, index=False)
wide.to_csv(wide_path, index=False)
combined.to_csv(balanced_path, index=False)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    final_summary.to_excel(
        writer,
        sheet_name="long_summary",
        index=False
    )

    wide.to_excel(
        writer,
        sheet_name="manuscript_table",
        index=False
    )

    combined.to_excel(
        writer,
        sheet_name="balanced_word_counts",
        index=False
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 90)
print("FINAL REVIEW-LENGTH DISTRIBUTIONS")
print("=" * 90)

for _, row in wide.iterrows():

    print()
    print(row["application"])

    print(
        "  Relevant: "
        f"N={int(row['relevant_n'])}, "
        f"mean={row['relevant_mean']:.2f}, "
        f"median={row['relevant_median']:.2f}, "
        f"Q1={row['relevant_q1']:.2f}, "
        f"Q3={row['relevant_q3']:.2f}, "
        f"IQR={row['relevant_iqr']:.2f}"
    )

    print(
        "  Newest:   "
        f"N={int(row['newest_n'])}, "
        f"mean={row['newest_mean']:.2f}, "
        f"median={row['newest_median']:.2f}, "
        f"Q1={row['newest_q1']:.2f}, "
        f"Q3={row['newest_q3']:.2f}, "
        f"IQR={row['newest_iqr']:.2f}"
    )


print()
print("=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(summary_path)
print(wide_path)
print(balanced_path)
print(excel_path)

print()
print("REVIEW LENGTH ANALYSIS COMPLETE.")