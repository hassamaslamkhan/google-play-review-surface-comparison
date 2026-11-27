# Google Play Review Surface Comparison

Reproducibility materials for the study:

**Same App, Different Surface: A Descriptive Comparison of Relevant and Newest Google Play Reviews Across Six Social-Media Applications**

The study compares Google Play's `Most relevant` (reported as **Relevant**) and
`Newest` review surfaces for Facebook, Instagram, Snapchat, TikTok, X, and
YouTube using a balanced Top-1,700 benchmark.

## What is included

- archived Google Play review collector snapshot;
- analysis scripts used for the Relevant-vs-Newest comparisons;
- statistical-inference verification code;
- Top-k, review-age, review-length, overlap, common-window, and temporal-standardization analyses;
- derived aggregate outputs supporting manuscript values;
- collection timestamps, app URLs, and software-environment notes.

## What is not included

Raw Google Play review exports are not redistributed here. They contain
user-generated review text and review-level identity fields. See
`docs/data_availability.md`.

## Repository structure

```text
collection/       archived collector snapshot + example collection configuration
analysis/          paper analysis and verification scripts
aggregate_data/    derived aggregate results only
docs/              environment, data availability, and data dictionary
```

## Running the analyses

Run commands from the repository root. If you have review-level source files
that you are permitted to use, place them in:

```text
relevant_reviews/
newest_reviews/
```

using the source filenames shown in `aggregate_data/collection_runs.csv`.

Suggested analysis order:

```bash
python analysis/comparison_relevant_newest.py
python analysis/review_length_distribution_v1.py
python analysis/topk_visibility_gradient.py
python analysis/review_age_distribution.py
python analysis/review_id_overlap_analysis.py
python analysis/common_date_window_robustness.py
python analysis/run_temporal_standardization.py
python analysis/verify_inference_methods.py
```

The analysis scripts reproduce a wider set of diagnostics than are ultimately
reported in the manuscript. Some internal output names retain the historical
term `negative_exposure_ratio`; this quantity is the manuscript's
Negative-Review Prevalence Ratio (NRPR).

## Collection note

See `collection/README.md` before describing the archived collection script.
The supplied collector snapshot preserves extraction, timestamp, deduplication,
scrolling, and stopping logic, but the archived file does not encode a general
programmatic selector for both Google Play ordering modes.

## Data sharing

Only derived aggregate results are included publicly. Raw review text and
review-level identifiers should be shared only where permitted by the source
platform's applicable policies.
