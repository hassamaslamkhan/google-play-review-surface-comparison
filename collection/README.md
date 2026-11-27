# Collection code

`google_play_reviews_selenium_archived.py` is the collector snapshot contained in
the archived project supplied for this repository. It opens the Google Play
review panel, extracts rendered review cards, preserves scrape order, records a
UTC collection timestamp, deduplicates by review ID, scrolls the review panel,
and stops after repeated no-new-ID rounds subject to a maximum-scroll safety
bound.

## Important scope note

This archived snapshot is configured through the `APP_NAME`, `APP_URL`, and
`OUTPUT_FILE` constants. It does **not** contain a generalized programmatic
selector for the `Most relevant` / `Newest` ordering control. Therefore it should
not be represented as a standalone script that fully reconstructs all 12
application-ordering runs without the original run-specific ordering setup.

The analysis repository preserves the exact collection timestamps and app URLs
in `aggregate_data/collection_runs.csv`.

No raw review files, reviewer names, profile-image URLs, or review text are
included in this public repository.
