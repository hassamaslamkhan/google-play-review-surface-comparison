# Data availability and privacy

This repository intentionally does not redistribute raw Google Play review text,
review identifiers, reviewer names, profile-image URLs, or other review-level
identity fields.

The `aggregate_data/` directory contains only derived summary outputs used to
support the manuscript's reported results.

The original project contains raw review exports with user-generated content.
Those files should be shared only where permitted by the source platform's
applicable policies.

The following source-data directories are intentionally excluded from Git:
- `relevant_reviews/`
- `newest_reviews/`

To reproduce the analysis from review-level data, place permitted source exports
in those directories using the filenames expected by the analysis scripts.
