# Aggregate data files

- `collection_runs.csv` — application, ordering, app URL, source filename, and UTC collection timestamp.
- `application_level_main_results.csv` — main Top-1,700 application-specific rating, negative-review, length, and related summaries.
- `pooled_main_results.csv` — pooled descriptive Top-1,700 summary.
- `sample_eligibility.csv` — available valid counts and common-depth eligibility.
- `application_topk_results.csv` — application-specific cumulative Top-k summaries.
- `pooled_topk_results.csv` — pooled cumulative Top-k summaries.
- `review_length_distribution.csv` — manuscript median/Q1/Q3 review-length table.
- `review_length_validation_summary.csv` — distributional validation summary without source file paths.
- `review_age_application_results.csv` — application-specific review-age summaries.
- `review_age_pooled_results.csv` — pooled review-age summary.
- `review_age_pooled_topk.csv` — pooled review-age summaries by Top-k depth.
- `review_id_overlap_application_results.csv` — aggregate application-level ID-overlap counts and Jaccard values; no IDs are included.
- `review_id_overlap_pooled_results.csv` — pooled aggregate overlap summary.
- `common_date_window_feasibility.csv` — common temporal support by application.
- `common_date_window_results.csv` — aggregate results for feasible common-window comparisons.
- `paper_standardization_table.csv` — compact temporal-standardization feasibility/result table.
- `temporal_textual_standardization_full.csv` — aggregate temporal/textual standardization diagnostics.

Some archived analysis scripts use the legacy internal variable name
`negative_exposure_ratio`; in this public aggregate-data export it is renamed
`nrpr` where practical to match the manuscript term
Negative-Review Prevalence Ratio.
