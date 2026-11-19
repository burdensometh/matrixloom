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

Supported transform specifications are `impute`, `neighbor_impute`, `standardize`, `range`, `winsorize`, `polynomial`, and `select`. These stages preserve sample population. Category encoding is dense, with an explicit width budget and `error` or `zero` behavior for unseen nonmissing categories. A missing category produces unobserved encoded entries until an imputer handles them.

Mean, median, and constant imputation fill each column from a single fitted statistic. `neighbor_impute` instead fills each missing entry from training donors that actually observe that column. The distance between a query row and a donor is the mean standardized squared difference over the input features observed in both rows; standardization scales are learned from the training frame only and never refit during prediction. A `min_common` setting rejects donors sharing fewer than the required number of observed dimensions, distance ties break on stable sample identity, an exact (zero-distance) match restricts aggregation to the matching donors, and a missing column with no eligible donor falls back to the training column mean or a configured constant. Population weights and optional inverse-distance weighting participate in the donor summary. Distances use the original observation mask (an explicit mask can be replayed over an already-filled frame), so imputed values never enter a distance and original observations are never changed. The fitted donor store, scales, fallbacks, and per-entry donor provenance (`fill_sources`) serialize inside the model bundle and survive reopen.

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

`Analysis.evaluate` persists an identity-bound evaluation; `classify` provides probability metrics; `compare`, `importance`, and `cluster_quality` consume actual model outputs. Prediction and regression scoring can read input data in batches, while fitted neighbor training data and dense preprocessing states remain in-memory structures in version 1.

Random, stratified, temporal, and k-fold split constructors are available. `holdout` fits only its selected training rows. Bootstrap means assume independent observations. Unconditional permutation importance is descriptive and is not a grouped or causal estimate. Neighbor intervals are empirical summaries and have no coverage guarantee.

## Persistence

`Store` retains immutable dataset and model revisions. A dataset revision freezes schema, order, sample identities, values, weights, and digest. Model bundles bind the fitted preprocessing state and estimator state; optional dataset binding verifies the training digest. Evaluation ids are idempotent only for the same model, dataset revision, population, predictions, and metrics.

Version 1 stores sample rows as JSON arrays in SQLite and exports model bundles as a versioned JSON envelope. Bundle import validates its digest and reconstructs supported numerical states; it does not execute pickles or arbitrary code. Full table loading verifies its digest. Batched and selected-row reads validate structure and population but do not independently recompute the full dataset digest.

The tests include hand-computed matrix solutions, weighted-population checks, identity reordering, missing-data populations, serialization round trips, corruption rejection, fitted-state isolation, and end-to-end reopen/evaluation workflows.
