# MatrixLoom

MatrixLoom processes identity-keyed sample tables into fitted numerical features and analysis results. It includes typed ingestion, missing-value handling, frozen categorical encoding, derived features, weighted descriptive statistics, ridge regression, logistic classification, nearest-neighbor regression, clustering, holdout evaluation, and SQLite persistence.

The application is a Python library. It does not contain a dashboard or business-management interface. All examples use fabricated observational records; there is no physical simulation, external data collection, or network dependency.

## Run

Python 3.11 or newer, with no runtime dependencies:

```sh
python3 -m unittest discover -s tests -v
python3 examples/analyze_batch.py /tmp/matrixloom-example
python3 examples/feature_graph.py /tmp/matrixloom-graph
```

The example builds a small batch, fits a pipeline, closes and reopens its model, then computes predictions and evaluation results from the saved state. `examples/feature_graph.py` does the same with a branching graph. The output directories should be new when running an example for the first time.

## Data contracts

- `Field.id` is stable identity; `Field.name` is a display label. A field also declares type, role, nullability, and optional measurement unit.
- Samples have unique string identities and finite, nonnegative population weights. Missing observations are `None`, not zero. Tables reject duplicate sample identities.
- `Frame` is an immutable dense numerical matrix. Every column has a `Feature` recording identity, source, operation, and unit. Reordering a compatible feature axis is allowed; changed lineage or units are rejected.- JSONL uses `_sample_id` and `_weight`. CSV includes the same columns. CSV null is `\N`; a literal string beginning with a backslash receives an extra leading backslash. An empty categorical string is preserved.
- Timestamps require a timezone and are normalized to UTC. Numeric model calculations use floating-point arithmetic. Integer fields preserve ordinary integral input; model feature extraction converts numerical values to floating point.

## Fitting and prediction

`Features` derives row expressions, fits an `Encoder`, then fits a sequential set of numerical transforms. `Predictor` adds a supervised estimator; `Clusterer` adds an unsupervised estimator. Fitted transforms never learn new statistics during prediction. A failed fit leaves the previously fitted predictor state intact.

Supported transform specifications are `impute`, `standardize`, `range`, `winsorize`, `polynomial`, and `select`. These stages preserve sample population. Category encoding is dense, with an explicit width budget and `error` or `zero` behavior for unseen nonmissing categories. A missing category produces unobserved encoded entries until an imputer handles them.

### Feature graphs

A sequential plan produces one output axis, so it cannot keep both a standardized branch and a nonlinear interaction branch. `FeatureGraph` expresses that shape explicitly. Each node declares its dependencies by identity:

```python
from matrixloom import FeatureGraph

graph = FeatureGraph(
    nodes=[
        {"id": "raw", "kind": "source", "fields": ["x", "z"], "unknown": "zero"},
        {"id": "numeric", "kind": "transform", "inputs": ["raw"],
         "transform": {"kind": "standardize"}},
        {"id": "expansion", "kind": "transform", "inputs": ["numeric"],
         "transform": {"kind": "polynomial", "degree": 2, "interactions_only": True}},
        {"id": "interaction", "kind": "transform", "inputs": ["expansion"],
         "transform": {"kind": "select", "ids": ["product:" + canonical(["x", "z"])]}},
        {"id": "joined", "kind": "merge", "inputs": ["numeric", "interaction"]},
    ],
    outputs=["joined"],
)
```

Node kinds are `source` (encodes input fields), `transform` (fits one registered stage on its single input), `sequence` (wraps an existing sequential `Features` specification, so current plans can be embedded unchanged), and `merge` (joins branches). A node consumed by several nodes is fitted once and its frozen state is reused by every consumer. `Fit` learns only from the table passed to it; `transform` never re-learns.

Graphs are validated before any node runs. Cycles, absent dependencies, unknown node options, and duplicated node identities are rejected at construction. Merge conflicts are rejected before fitting: merging requires identical sample populations and weights, aligns rows by sample identity rather than position, and refuses a feature identity contributed by two inputs, including categorical sources and deterministic polynomial/selector expansions. Ambiguity is reported instead of silently overwriting a column.

Graphs serialize through `FeatureGraph.state()`/`FeatureGraph.restore()` and can be stored directly as analysis bundles (`Analysis.graph`, applied later with `Analysis.graph_frame`) or used as a predictor/clusterer feature specification by passing `{"kind": "graph", "nodes": [...], "outputs": [...]}`. Sequential specifications remain the default and still round-trip unchanged.

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
