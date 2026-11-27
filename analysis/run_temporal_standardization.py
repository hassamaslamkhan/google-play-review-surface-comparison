from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR_R = PROJECT_DIR / "relevant_reviews"
PROJECT_DIR_N = PROJECT_DIR / "newest_reviews"

OUTPUT_DIR = (
    PROJECT_DIR
    / "major_revision_analysis"
    / "temporal_standardization"
    / "results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 1700

# ============================================================
# EXISTING RAW FILES
# ============================================================

FILES = {
    "Facebook": {
        "Relevant": PROJECT_DIR_R / "facebook_google_play_reviews.xlsx",
        "Newest": PROJECT_DIR_N / "Newest_facebook_google_play_reviews.xlsx",
    },
    "Instagram": {
        "Relevant": PROJECT_DIR_R / "instagram_google_play_reviews.xlsx",
        "Newest": PROJECT_DIR_N / "Newest_instagram_google_play_reviews.xlsx",
    },
    "Snapchat": {
        "Relevant": PROJECT_DIR_R / "snapchat_google_play_reviews.xlsx",
        "Newest": PROJECT_DIR_N / "Newest_snapchat_google_play_reviews.xlsx",
    },
    "TikTok": {
        "Relevant": PROJECT_DIR_R / "tiktok_google_play_reviews.xlsx",
        "Newest": PROJECT_DIR_N / "Newest_tiktok_google_play_reviews.xlsx",
    },
    "X": {
        "Relevant": PROJECT_DIR_R / "x_google_play_reviews.xlsx",
        "Newest": PROJECT_DIR_N / "Newest_x_google_play_reviews.xlsx",
    },
    "YouTube": {
        "Relevant": PROJECT_DIR_R / "youtube_google_play_reviews.xlsx",
        "Newest": PROJECT_DIR_N / "Newest_youtube_google_play_reviews.xlsx",
    },
}

# ============================================================
# COLUMN NAMES USED IN YOUR GOOGLE PLAY FILES
# ============================================================

RATING_CANDIDATES = [
    "rating",
    "score",
    "stars",
    "star_rating",
]

DATE_CANDIDATES = [
    "review_date_iso",
    "review_date",
    "date",
]

TEXT_CANDIDATES = [
    "review_text",
    "content",
    "text",
    "review",
]

ORDER_CANDIDATES = [
    "scrape_order",
    "order",
    "rank",
    "position",
]


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, required=True):
    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find column from {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def check_files():
    print("=" * 80)
    print("CHECKING INPUT FILES")
    print("=" * 80)

    missing = []

    for app, surfaces in FILES.items():
        for surface, path in surfaces.items():
            status = "FOUND" if path.exists() else "MISSING"

            print(
                f"{app:10s} "
                f"{surface:8s} "
                f"{status:7s} "
                f"{path.name}"
            )

            if not path.exists():
                missing.append(path)

    if missing:
        print("\nMissing files:")
        for path in missing:
            print("  ", path)

        raise FileNotFoundError(
            "\nOne or more required files are missing."
        )

    print("\nAll 12 raw files found.\n")


def read_excel(path):
    xls = pd.ExcelFile(path)

    # Your raw files normally use "reviews".
    if "reviews" in xls.sheet_names:
        sheet = "reviews"
    else:
        sheet = xls.sheet_names[0]

    return pd.read_excel(
        path,
        sheet_name=sheet,
    )


def prepare_surface(path, surface):
    df = read_excel(path).copy()

    rating_col = find_column(
        df,
        RATING_CANDIDATES,
    )

    date_col = find_column(
        df,
        DATE_CANDIDATES,
    )

    text_col = find_column(
        df,
        TEXT_CANDIDATES,
    )

    order_col = find_column(
        df,
        ORDER_CANDIDATES,
        required=False,
    )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    df["rating_numeric"] = pd.to_numeric(
        df[rating_col],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Review date
    # --------------------------------------------------------

    df["review_datetime"] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        utc=True,
    )

    df["review_date"] = (
        df["review_datetime"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    # --------------------------------------------------------
    # Text length
    # --------------------------------------------------------

    df["review_text_clean"] = (
        df[text_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["word_count"] = (
        df["review_text_clean"]
        .str.split()
        .str.len()
    )

    # --------------------------------------------------------
    # Negative = 1 or 2 stars
    # --------------------------------------------------------

    df["negative"] = (
        df["rating_numeric"] < 3
    ).astype(int)

    # --------------------------------------------------------
    # Scrape order
    # --------------------------------------------------------

    if order_col is not None:
        df["_scrape_order"] = pd.to_numeric(
            df[order_col],
            errors="coerce",
        )

        df = df.sort_values(
            "_scrape_order",
            kind="stable",
        )

    # --------------------------------------------------------
    # Valid rows
    # --------------------------------------------------------

    df = df[
        df["rating_numeric"].notna()
        & df["review_date"].notna()
    ].copy()

    # Same Top-1700 benchmark as manuscript
    df = df.head(TOP_N).copy()

    df["surface"] = surface

    return df[
        [
            "surface",
            "rating_numeric",
            "negative",
            "review_date",
            "word_count",
        ]
    ]


def weighted_mean(values, weights):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights >= 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    if weights.sum() == 0:
        return np.nan

    return np.average(
        values,
        weights=weights,
    )


def effective_sample_size(weights):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights[
        np.isfinite(weights)
        & (weights > 0)
    ]

    if len(weights) == 0:
        return np.nan

    return (
        weights.sum() ** 2
        / np.sum(weights ** 2)
    )


def ratio(rel_prev, new_prev):
    if (
        pd.isna(rel_prev)
        or pd.isna(new_prev)
        or new_prev == 0
    ):
        return np.nan

    return rel_prev / new_prev


# ============================================================
# RAW ANALYSIS
# ============================================================

def raw_analysis(rel, new):
    rel_prev = rel["negative"].mean()
    new_prev = new["negative"].mean()

    return {
        "rel_prev": rel_prev,
        "new_prev": new_prev,
        "ratio": ratio(
            rel_prev,
            new_prev,
        ),
    }


# ============================================================
# EXACT DATE STANDARDIZATION
# ============================================================

def date_standardization(rel, new):
    """
    Strict common-support analysis.

    1. Find calendar dates represented in BOTH surfaces.
    2. Restrict both surfaces to those dates.
    3. Reweight Relevant so that its date distribution equals
       the Newest date distribution within that common support.
    """

    common_dates = sorted(
        set(rel["review_date"])
        & set(new["review_date"])
    )

    if not common_dates:
        return None

    rel_c = rel[
        rel["review_date"].isin(common_dates)
    ].copy()

    new_c = new[
        new["review_date"].isin(common_dates)
    ].copy()

    # Need enough observations for meaningful analysis.
    if len(rel_c) < 30 or len(new_c) < 30:
        return None

    # --------------------------------------------------------
    # Target = Newest date distribution
    # --------------------------------------------------------

    target = (
        new_c["review_date"]
        .value_counts(normalize=True)
        .rename("target")
    )

    source = (
        rel_c["review_date"]
        .value_counts(normalize=True)
        .rename("source")
    )

    weights = pd.concat(
        [target, source],
        axis=1,
    ).dropna()

    weights["weight"] = (
        weights["target"]
        / weights["source"]
    )

    rel_c = rel_c.merge(
        weights[["weight"]],
        left_on="review_date",
        right_index=True,
        how="inner",
    )

    # Newest defines target distribution.
    new_c["weight"] = 1.0

    rel_prev = weighted_mean(
        rel_c["negative"],
        rel_c["weight"],
    )

    new_prev = weighted_mean(
        new_c["negative"],
        new_c["weight"],
    )

    return {
        "rel_prev": rel_prev,
        "new_prev": new_prev,
        "ratio": ratio(
            rel_prev,
            new_prev,
        ),

        "n_rel": len(rel_c),
        "n_new": len(new_c),

        "coverage_rel": len(rel_c) / len(rel),
        "coverage_new": len(new_c) / len(new),

        "ess_rel": effective_sample_size(
            rel_c["weight"]
        ),

        "common_dates": len(common_dates),

        "date_min": min(common_dates),
        "date_max": max(common_dates),
    }


# ============================================================
# DATE + LENGTH STANDARDIZATION
# ============================================================

def date_length_standardization(rel, new):
    """
    Standardize on BOTH:
      - exact review date
      - review-length quintile

    Only joint strata represented in BOTH surfaces are used.
    """

    common_dates = sorted(
        set(rel["review_date"])
        & set(new["review_date"])
    )

    if not common_dates:
        return None

    rel_c = rel[
        rel["review_date"].isin(common_dates)
    ].copy()

    new_c = new[
        new["review_date"].isin(common_dates)
    ].copy()

    if len(rel_c) < 30 or len(new_c) < 30:
        return None

    # --------------------------------------------------------
    # Create common word-count bins using pooled distribution
    # --------------------------------------------------------

    pooled_words = pd.concat(
        [
            rel_c["word_count"],
            new_c["word_count"],
        ],
        ignore_index=True,
    )

    try:
        _, edges = pd.qcut(
            pooled_words,
            q=5,
            retbins=True,
            duplicates="drop",
        )
    except Exception:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf

    rel_c["length_bin"] = pd.cut(
        rel_c["word_count"],
        bins=edges,
        include_lowest=True,
    )

    new_c["length_bin"] = pd.cut(
        new_c["word_count"],
        bins=edges,
        include_lowest=True,
    )

    # Joint date + length stratum
    rel_c["stratum"] = list(
        zip(
            rel_c["review_date"],
            rel_c["length_bin"],
        )
    )

    new_c["stratum"] = list(
        zip(
            new_c["review_date"],
            new_c["length_bin"],
        )
    )

    common_strata = (
        set(rel_c["stratum"])
        & set(new_c["stratum"])
    )

    if not common_strata:
        return None

    rel_j = rel_c[
        rel_c["stratum"].isin(common_strata)
    ].copy()

    new_j = new_c[
        new_c["stratum"].isin(common_strata)
    ].copy()

    if len(rel_j) < 30 or len(new_j) < 30:
        return None

    # --------------------------------------------------------
    # Target joint distribution from Newest
    # --------------------------------------------------------

    target = (
        new_j["stratum"]
        .value_counts(normalize=True)
        .rename("target")
    )

    source = (
        rel_j["stratum"]
        .value_counts(normalize=True)
        .rename("source")
    )

    weights = pd.concat(
        [target, source],
        axis=1,
    ).dropna()

    weights["weight"] = (
        weights["target"]
        / weights["source"]
    )

    rel_j = rel_j.merge(
        weights[["weight"]],
        left_on="stratum",
        right_index=True,
        how="inner",
    )

    new_j["weight"] = 1.0

    rel_prev = weighted_mean(
        rel_j["negative"],
        rel_j["weight"],
    )

    new_prev = weighted_mean(
        new_j["negative"],
        new_j["weight"],
    )

    return {
        "rel_prev": rel_prev,
        "new_prev": new_prev,
        "ratio": ratio(
            rel_prev,
            new_prev,
        ),

        "n_rel": len(rel_j),
        "n_new": len(new_j),

        "coverage_rel": len(rel_j) / len(rel),
        "coverage_new": len(new_j) / len(new),

        "ess_rel": effective_sample_size(
            rel_j["weight"]
        ),

        "common_strata": len(common_strata),
    }


# ============================================================
# RUN ONE APPLICATION
# ============================================================

def analyze_app(app, paths):
    print("\n" + "=" * 80)
    print(app)
    print("=" * 80)

    rel = prepare_surface(
        paths["Relevant"],
        "Relevant",
    )

    new = prepare_surface(
        paths["Newest"],
        "Newest",
    )

    print(
        f"Relevant N = {len(rel):,}"
    )

    print(
        f"Newest   N = {len(new):,}"
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw = raw_analysis(
        rel,
        new,
    )

    print(
        "\nRAW:"
        f"\n  Relevant negative = {raw['rel_prev']*100:.2f}%"
        f"\n  Newest negative   = {raw['new_prev']*100:.2f}%"
        f"\n  Ratio             = {raw['ratio']:.4f}"
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    ds = date_standardization(
        rel,
        new,
    )

    if ds is None:
        print(
            "\nDATE STANDARDIZATION:"
            "\n  NOT FEASIBLE: insufficient common calendar-date support."
        )
    else:
        print(
            "\nDATE STANDARDIZATION:"
            f"\n  Relevant negative = {ds['rel_prev']*100:.2f}%"
            f"\n  Newest negative   = {ds['new_prev']*100:.2f}%"
            f"\n  Ratio             = {ds['ratio']:.4f}"
            f"\n  Relevant N        = {ds['n_rel']:,}"
            f"\n  Newest N          = {ds['n_new']:,}"
            f"\n  Relevant ESS      = {ds['ess_rel']:.1f}"
            f"\n  Common dates      = {ds['common_dates']}"
            f"\n  Date range        = "
            f"{ds['date_min'].date()} to "
            f"{ds['date_max'].date()}"
        )

    # --------------------------------------------------------
    # DATE + LENGTH
    # --------------------------------------------------------

    dl = date_length_standardization(
        rel,
        new,
    )

    if dl is None:
        print(
            "\nDATE + LENGTH STANDARDIZATION:"
            "\n  NOT FEASIBLE: insufficient joint support."
        )
    else:
        print(
            "\nDATE + LENGTH STANDARDIZATION:"
            f"\n  Relevant negative = {dl['rel_prev']*100:.2f}%"
            f"\n  Newest negative   = {dl['new_prev']*100:.2f}%"
            f"\n  Ratio             = {dl['ratio']:.4f}"
            f"\n  Relevant N        = {dl['n_rel']:,}"
            f"\n  Newest N          = {dl['n_new']:,}"
            f"\n  Relevant ESS      = {dl['ess_rel']:.1f}"
            f"\n  Common strata     = {dl['common_strata']}"
        )

    # --------------------------------------------------------
    # OUTPUT ROW
    # --------------------------------------------------------

    row = {
        "Application": app,

        "Raw Relevant Negative %":
            raw["rel_prev"] * 100,

        "Raw Newest Negative %":
            raw["new_prev"] * 100,

        "Raw Ratio":
            raw["ratio"],
    }

    if ds is not None:
        row.update({
            "Date Relevant Negative %":
                ds["rel_prev"] * 100,

            "Date Newest Negative %":
                ds["new_prev"] * 100,

            "Date-Standardized Ratio":
                ds["ratio"],

            "Date Relevant N":
                ds["n_rel"],

            "Date Newest N":
                ds["n_new"],

            "Date Relevant ESS":
                ds["ess_rel"],

            "Date Relevant Coverage %":
                ds["coverage_rel"] * 100,

            "Date Newest Coverage %":
                ds["coverage_new"] * 100,
        })
    else:
        row["Date-Standardized Ratio"] = np.nan

    if dl is not None:
        row.update({
            "Date+Length Relevant Negative %":
                dl["rel_prev"] * 100,

            "Date+Length Newest Negative %":
                dl["new_prev"] * 100,

            "Date+Length Ratio":
                dl["ratio"],

            "Joint Relevant N":
                dl["n_rel"],

            "Joint Newest N":
                dl["n_new"],

            "Joint Relevant ESS":
                dl["ess_rel"],

            "Joint Relevant Coverage %":
                dl["coverage_rel"] * 100,

            "Joint Newest Coverage %":
                dl["coverage_new"] * 100,
        })
    else:
        row["Date+Length Ratio"] = np.nan

    return row


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    check_files()

    results = []

    for app, paths in FILES.items():
        try:
            result = analyze_app(
                app,
                paths,
            )

            results.append(result)

        except Exception as exc:
            print(
                f"\nERROR processing {app}: {exc}"
            )

            raise

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # FULL AUDIT OUTPUT
    # --------------------------------------------------------

    full_path = (
        OUTPUT_DIR
        / "temporal_textual_standardization_full.csv"
    )

    df.to_csv(
        full_path,
        index=False,
    )

    # --------------------------------------------------------
    # PAPER TABLE
    # --------------------------------------------------------

    paper = df[
        [
            "Application",
            "Raw Ratio",
            "Date-Standardized Ratio",
            "Date+Length Ratio",
        ]
    ].copy()

    def fmt(x):
        if pd.isna(x):
            return "Not feasible"

        return f"{x:.2f}x"

    for col in [
        "Raw Ratio",
        "Date-Standardized Ratio",
        "Date+Length Ratio",
    ]:
        paper[col] = paper[col].apply(fmt)

    paper_path = (
        OUTPUT_DIR
        / "paper_standardization_table.csv"
    )

    paper.to_csv(
        paper_path,
        index=False,
    )

    print("\n\n")
    print("=" * 80)
    print("FINAL PAPER TABLE")
    print("=" * 80)

    print(
        paper.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # LATEX ROWS
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("LATEX ROWS")
    print("=" * 80)

    for _, row in paper.iterrows():

        def latex_fmt(value):
            if value == "Not feasible":
                return r"\textit{Not feasible}"

            return value.replace(
                "x",
                r"$\times$",
            )

        print(
            f"{row['Application']} & "
            f"{latex_fmt(row['Raw Ratio'])} & "
            f"{latex_fmt(row['Date-Standardized Ratio'])} & "
            f"{latex_fmt(row['Date+Length Ratio'])} \\\\"
        )

    print("\nSaved outputs:")
    print(full_path)
    print(paper_path)

    print("\nDONE.")