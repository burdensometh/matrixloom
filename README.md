# MatrixLoom

MatrixLoom processes identity-keyed sample tables into fitted numerical features and analysis results. It includes typed ingestion, missing-value handling, frozen categorical encoding, derived features, weighted descriptive statistics, ridge regression, logistic classification, nearest-neighbor regression, clustering, holdout evaluation, and SQLite persistence.

The application is a Python library. It does not contain a dashboard or business-management interface. All examples use fabricated observational records; there is no physical simulation, external data collection, or network dependency.

## Run

Python 3.11 or newer, with no runtime dependencies:

```sh
python3 -m unittest discover -s tests -v
python3 examples/analyze_batch.py /tmp/matrixloom-example
```

The example builds a small batch, fits a pipeline, closes and reopens its model, then computes predictions and evaluation results from the saved state. The output directory should be new when running the example for the first time.

## Data contracts

- `Field.id` is stable identity; `Field.name` is a display label. A field also declares type, role, nullability, and optional measurement unit.
- Samples have unique string identities and finite, nonnegative population weights. Missing observations are `None`, not zero. Tables reject duplicate sample identities.
- `Frame` is an immutable dense numerical matrix. Every column has a `Feature` recording identity, source, operation, and unit. Reordering a compatible feature axis is allowed; changed lineage or units are rejected.
- JSONL uses `_sample_id` and `_weight`. CSV includes the same columns. CSV null is `\N`; a literal string beginning with a backslash receives an extra leading backslash. An empty categorical string is preserved.
- Timestamps require a timezone and are normalized to UTC. Numeric model calculations use floating-point arithmetic. Integer fields preserve ordinary integral input; model feature extraction converts numerical values to floating point.

## Fitting and prediction

`Features` derives row expressions, fits an `Encoder`, then fits a sequential set of numerical transforms. `Predictor` adds a supervised estimator; `Clusterer` adds an unsupervised estimator. Fitted transforms never learn new statistics during prediction. A failed fit leaves the previously fitted predictor state intact.

Supported transform specifications are `impute`, `standardize`, `range`, `winsorize`, `polynomial`, and `select`. These stages preserve sample population. Category encoding is dense, with an explicit width budget and `error` or `zero` behavior for unseen nonmissing categories. A missing category produces unobserved encoded entries until an imputer handles them.

Ridge minimizes weighted squared residuals plus `alpha * sum(beta**2)`, with an optional unpenalized intercept. The implementation uses a small dense normal system with scaled-pivot elimination and rejects singular systems. Logistic fitting uses a weighted penalized binary likelihood, Newton steps, and backtracking. Neighbors use complete-feature Euclidean distances, stable sample-id tie breaking, and optional distance weighting. KMeans requires complete observations and positive support for every output cluster.

Weighted variance with `ddof=1` uses the reliability-weight denominator `sum(w) - sum(w**2)/sum(w)`. Quantiles use the smallest value whose cumulative positive weight reaches the requested fraction. Pairwise covariance reports pair-specific support and may be non-positive-semidefinite when observation patterns differ; it must not be treated automatically as a complete-data covariance matrix.

## Application flow

```python
from matrixloom import Analysis, Field, Schema, Table
from matrixloom.table import Sample

schema = Schema([
    Field("x", "Measurement"),
    Field("y", "Response", role="target"),
])
table = Table(schema, [Sample("a", (1, 3)), Sample("b", (2, 5)), Sample("c", (3, 7))])
analysis = Analysis("/tmp/matrixloom-work")
analysis.ingest("observations", table)
analysis.train("model", "observations", {"fields": ["x"]}, {"kind": "ridge", "alpha": 0}, "y")
predictions = analysis.predict("model", "observations")
```

`Analysis.evaluate` persists an identity-bound evaluation; `classify` provides probability metrics; `compare`, `importance`, and `cluster_quality` consume actual model outputs. Prediction and regression scoring can read input data in batches, while fitted neighbor training data and dense preprocessing states remain in-memory structures once loaded.

Random, stratified, temporal, and k-fold split constructors are available. `holdout` fits only its selected training rows. Bootstrap means assume independent observations. Unconditional permutation importance is descriptive and is not a grouped or causal estimate. Neighbor intervals are empirical summaries and have no coverage guarantee.

## Persistence

`Store` retains immutable dataset and model revisions. A dataset revision freezes schema, order, sample identities, values, weights, and digest. Model bundles bind the fitted preprocessing state and estimator state; optional dataset binding verifies the training digest. Evaluation ids are idempotent only for the same model, dataset revision, population, predictions, and metrics.

Version 1 stores sample rows as JSON arrays in SQLite and exports model bundles as a versioned JSON envelope. Bundle import validates its digest and reconstructs supported numerical states; it does not execute pickles or arbitrary code. Full table loading verifies its digest. Batched and selected-row reads validate structure and population but do not independently recompute the full dataset digest.

### Format version 2: column blocks

New datasets and bundles are written in format version 2. A dataset revision is stored as ordered, checksummed column blocks (`sample_blocks`) rather than one JSON array per sample. Every block records its sample order (`sample_ids`, `first_row`), the identity of every column (`column_ids`), the per-column missing mask (`missing`, `0` observed / `1` missing), the sample weights, and a SHA-256 digest over the block body. The schema, row count, and whole-population digest remain on the revision header.

A fitted model's embedded neighbor training matrix is stored the same way: a `matrixloom-frame-blocks` envelope with the same sample order, feature identities, missing mask, and weights. Digests identify logical content, not encoding, so a bundle's digest, its training digest, its bound dataset digest, and its evaluation digests are unchanged by the encoding. Nothing is refitted.

Reads accept both formats. Version 1 revisions are served from the legacy `samples` table; version 2 revisions are decoded block by block and validated (checksums, feature identity, mask/value agreement, contiguous coverage). `Store.legacy(path)` opens a version 1 inventory read-only, and `Store(path).read_only=True` prevents accidental writes.

Writes are interruption-safe. `save_table` first commits the revision header unpublished, then commits blocks one at a time, then verifies the decoded population against the source digest and publishes in a final transaction. An interrupted write leaves an unpublished draft that readers cannot see (`Store.drafts()` lists them) and a rerun of the same `save_table` resumes it. Bundle payloads are replaced in a single transaction.

### Migrating a version 1 inventory

`migrate_inventory(source, destination=None, block_rows=256)` converts every version 1 dataset revision and bundle in place (or into a copied destination file), never refitting. Each table is decoded, re-encoded as blocks, decoded again, and compared on schema, sample order, weights, and digest before its revision is published and its legacy rows removed in one transaction. Each bundle's training matrix is converted with its logical digest preserved, and each evaluation record is retained as stored.

`verify_inventory(legacy, migrated)` reopens both sides and checks that every table digest and population, every model digest, every prediction over its bound dataset revision, and every historical evaluation record are identical. `migrate_bundle_files(source_dir, destination_dir)` converts exported JSON envelopes. The console command is:

```sh
matrixloom-migrate fixtures/v1/analysis.sqlite /tmp/matrixloom-v2/analysis.sqlite
```

Without a destination the inventory is migrated in place after the original is snapshotted for verification.


The tests include hand-computed matrix solutions, weighted-population checks, identity reordering, missing-data populations, serialization round trips, corruption rejection, fitted-state isolation, and end-to-end reopen/evaluation workflows.
