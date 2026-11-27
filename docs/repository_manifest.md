# Repository manifest

## Included from the archived project

### Collection
- `collection/google_play_reviews_selenium_archived.py`

### Analysis
- `analysis/comparison_relevant_newest.py`
- `analysis/review_length_distribution_v1.py`
- `analysis/topk_visibility_gradient.py`
- `analysis/review_age_distribution.py`
- `analysis/review_id_overlap_analysis.py`
- `analysis/common_date_window_robustness.py`
- `analysis/run_temporal_standardization.py`
- `analysis/verify_inference_methods.py`

The temporal-standardization script was changed only to replace the author's
local absolute Windows project path with a repository-relative project path.
The main comparison script was renamed from the archived filename
`comparsion_relevant_newest.py` to `comparison_relevant_newest.py`.

## Intentionally excluded
- all raw files in `relevant_reviews/` and `newest_reviews/`;
- raw or balanced tables containing review text, review IDs, reviewer names, or avatar URLs;
- narrative/LLM analysis folders, because semantic/narrative analysis is not part of the reported paper;
- cross-platform reviewer-name/avatar matching files;
- manuscript drafts and unrelated journal-format documents;
- local/private infrastructure configuration;
- duplicate pipeline ZIPs and unrelated narrative figures.

## Aggregate outputs
Only summary-level CSV files supporting the manuscript are included in
`aggregate_data/`.
