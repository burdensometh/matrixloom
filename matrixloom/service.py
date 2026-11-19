from pathlib import Path

from .diagnostics import associations, collinearity, drift, missing_by_group, outliers
from .errors import SchemaError, StateError
from .evaluation import (holdout, score, learning_curve, permutation_importance,
                         grouped_comparison, targets)
from .frame import numerical_frame
from .metrics import compare, silhouette
from .pipeline import Features, Predictor, Clusterer
from .schema import fingerprint
from .splits import random_split
from .statistics import summarize, covariance_matrix, frequency
from .storage import Store
from .table import Table
from .units import normalize


class Analysis:
    """Application entry points operate on actual samples and fitted numerical states."""

    def __init__(self, home):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.home / "analysis.sqlite")

    def ingest(self, key, table, expected_revision=None, units=None):
        if units:
            table = normalize(table, units)
        return self.store.save_table(key, table, expected_revision)

    def ingest_jsonl(self, key, schema, text, expected_revision=None, units=None):
        return self.ingest(key, Table.from_jsonl(schema, text), expected_revision, units)

    def ingest_csv(self, key, schema, text, expected_revision=None, units=None):
        return self.ingest(key, Table.from_csv(schema, text), expected_revision, units)

    def train(self, model_name, dataset_name, feature_spec, model_spec, target,
              dataset_revision=None, expected_model_revision=None):
        info = self.store.table_info(dataset_name, dataset_revision)
        table = self.store.load_table(dataset_name, info["revision"])
        predictor = Predictor(feature_spec, model_spec, target).fit(table)
        saved = self.store.save_bundle(model_name, predictor, (dataset_name, info["revision"]), expected_model_revision)
        return {**saved, "training_rows": len(table), "features": predictor.features.describe(),
                "training_metrics": score(predictor, table)["metrics"]}

    def cluster(self, model_name, dataset_name, feature_spec, model_spec,
                dataset_revision=None, expected_model_revision=None):
        info = self.store.table_info(dataset_name, dataset_revision)
        table = self.store.load_table(dataset_name, info["revision"])
        clusterer = Clusterer(feature_spec, model_spec).fit(table)
        saved = self.store.save_bundle(model_name, clusterer, (dataset_name, info["revision"]), expected_model_revision)
        return {**saved, "assignments": clusterer.predict(table), "inertia": clusterer.model.inertia}

    def predict(self, model_name, dataset_name, model_revision=None, dataset_revision=None):
        bundle = self.store.load_bundle(model_name, model_revision)
        table = self.store.load_table(dataset_name, dataset_revision)
        return bundle.predict(table)

    def evaluate(self, run_id, model_name, dataset_name, model_revision=None,
                 dataset_revision=None, group_field=None):
        model_info = self.store.bundle_info(model_name, model_revision)
        data_info = self.store.table_info(dataset_name, dataset_revision)
        bundle = self.store.load_bundle(model_name, model_info["revision"])
        if not isinstance(bundle, Predictor):
            raise SchemaError("regression evaluation requires a predictor")
        table = self.store.load_table(dataset_name, data_info["revision"])
        evaluated = score(bundle, table, group_field)
        saved_metrics = {"overall": evaluated["metrics"], "groups": evaluated.get("groups"),
                         "model_digest": evaluated["model_digest"], "data_digest": evaluated["data_digest"]}
        saved = self.store.save_evaluation(run_id, (model_name, model_info["revision"]),
                    (dataset_name, data_info["revision"]), evaluated["predictions"], saved_metrics)
        return {**saved, **evaluated}

    def inspect(self, dataset_name, fields, dataset_revision=None, group_field=None):
        table = self.store.load_table(dataset_name, dataset_revision)
        frame = numerical_frame(table, fields)
        result = {"dataset_digest": table.digest(), "summary": summarize(frame),
                  "missingness": frame.missingness(), "covariance": covariance_matrix(frame, pairwise=True),
                  "associations": associations(frame), "outliers": outliers(frame)}
        if group_field is not None:
            result["missing_by_group"] = missing_by_group(table, group_field, fields)
        return result

    def compare(self, dataset_name, left_name, right_name, dataset_revision=None,
                left_revision=None, right_revision=None):
        table = self.store.load_table(dataset_name, dataset_revision)
        left = self.store.load_bundle(left_name, left_revision)
        right = self.store.load_bundle(right_name, right_revision)
        if not isinstance(left, Predictor) or not isinstance(right, Predictor) or left.target != right.target:
            raise SchemaError("model comparison requires two predictors for the same target")
        return compare(targets(table, left.target), left.predict(table), right.predict(table),
                       dict(zip(table.ids(), table.weights())))

    def compare_grouped(self, comparison_id, dataset_name, left_name, right_name, group_field,
                        metric="mae", repeats=1000, level=.95, seed=0, dataset_revision=None,
                        left_revision=None, right_revision=None):
        """Run and persist a group-level paired bootstrap comparison of two fixed models.

        Cluster-correlated errors make the row-level bootstrap too narrow, so entire
        groups are resampled with replacement and every member of a drawn group is kept.
        Both models share each draw, and the original sample weights are reused.
        """
        data_info = self.store.table_info(dataset_name, dataset_revision)
        left_info = self.store.bundle_info(left_name, left_revision)
        right_info = self.store.bundle_info(right_name, right_revision)
        left = self.store.load_bundle(left_name, left_info["revision"])
        right = self.store.load_bundle(right_name, right_info["revision"])
        table = self.store.load_table(dataset_name, data_info["revision"])
        result = grouped_comparison(table, left, right, group_field, metric, repeats, level, seed)
        # Derive the replay request from the validated result so a stored record always
        # reproduces exactly what was computed, not an unconverted caller argument.
        requests = {"metric": result["metric"], "group_field": group_field,
                    "repeats": result["repeats"], "level": result["level"],
                    "seed": result["seed"], "target": result["target"],
                    "interval_method": result["interval_method"],
                    "interval_convention": result["interval_convention"],
                    "difference_definition": result["difference_definition"],
                    "resampling_unit": result["resampling_unit"], "paired": result["paired"]}
        saved = self.store.save_comparison(comparison_id,
                    (dataset_name, data_info["revision"]),
                    (left_name, left_info["revision"]),
                    (right_name, right_info["revision"]),
                    group_field, metric, requests, result, result["population"])
        return {**result, **saved}

    def replay_comparison(self, comparison_id):
        """Rebuild a stored grouped comparison from its pinned seed and interval settings.

        The recomputation must reproduce the stored digest exactly; otherwise the record
        is treated as corrupt rather than reported as a verified replay.
        """
        stored = self.store.comparison(comparison_id)
        requests = stored["requests"]
        dataset_name, dataset_revision = stored["dataset"]
        left_name, left_revision = stored["left"]
        right_name, right_revision = stored["right"]
        left = self.store.load_bundle(left_name, left_revision)
        right = self.store.load_bundle(right_name, right_revision)
        table = self.store.load_table(dataset_name, dataset_revision)
        recomputed = grouped_comparison(table, left, right, requests["group_field"],
                                        requests["metric"], requests["repeats"],
                                        requests["level"], requests["seed"])
        payload = {"dataset": [dataset_name, dataset_revision],
                   "left": [left_name, left_revision], "right": [right_name, right_revision],
                   "metric": requests["metric"], "group_field": requests["group_field"],
                   "population_digest": stored["population_digest"],
                   "requests": requests, "result": recomputed}
        if fingerprint(payload) != stored["digest"]:
            raise StateError("stored comparison replay does not reproduce its digest")
        return {**recomputed, "comparison_id": comparison_id, "replayed": True,
                "verified": True, "digest": stored["digest"]}

    def holdout(self, dataset_name, feature_spec, model_spec, target, split=None,
                dataset_revision=None, seed=0, test_fraction=.2, group_field=None):
        table = self.store.load_table(dataset_name, dataset_revision)
        split = random_split(table, test_fraction, seed) if split is None else split
        return holdout(table, split, feature_spec, model_spec, target, group_field)

    def learning_curve(self, dataset_name, split, fractions, feature_spec, model_spec, target,
                       dataset_revision=None, seed=0):
        table = self.store.load_table(dataset_name, dataset_revision)
        return learning_curve(table, split, fractions, feature_spec, model_spec, target, seed)

    def importance(self, model_name, dataset_name, fields, repeats=5, seed=0,
                   model_revision=None, dataset_revision=None):
        model = self.store.load_bundle(model_name, model_revision)
        if not isinstance(model, Predictor):
            raise SchemaError("prediction importance requires a predictor")
        table = self.store.load_table(dataset_name, dataset_revision)
        return permutation_importance(model, table, fields, repeats, seed)

    def drift(self, reference_name, current_name, fields, reference_revision=None, current_revision=None):
        reference = self.store.load_table(reference_name, reference_revision)
        current = self.store.load_table(current_name, current_revision)
        return drift(reference, current, fields)

    def encoded(self, dataset_name, feature_spec, dataset_revision=None):
        table = self.store.load_table(dataset_name, dataset_revision)
        features = Features(**feature_spec)
        frame = features.fit_transform(table)
        return {"frame": frame, "pipeline": features, "description": features.describe()}

    def cluster_quality(self, model_name, dataset_name, model_revision=None, dataset_revision=None):
        model = self.store.load_bundle(model_name, model_revision)
        if not isinstance(model, Clusterer):
            raise SchemaError("cluster quality requires a clusterer")
        table = self.store.load_table(dataset_name, dataset_revision)
        assignments = model.predict(table)
        frame = model.features.transform(table)
        return silhouette(frame, {key: value["cluster"] for key, value in assignments.items()})

    def category_frequencies(self, dataset_name, fields, dataset_revision=None):
        table = self.store.load_table(dataset_name, dataset_revision)
        result = {}
        for key in fields:
            if table.schema.field(key).kind != "category":
                raise SchemaError("frequency analysis requires categorical fields")
            values = frequency(table.column(key), table.weights())
            result[key] = [{"category": category, "weight": weight} for category, weight in values.items()]
        return result

    def classify(self, run_id, model_name, dataset_name, threshold=.5, group_field=None,
                 model_revision=None, dataset_revision=None):
        from .evaluation import classification_score
        model_info = self.store.bundle_info(model_name, model_revision)
        data_info = self.store.table_info(dataset_name, dataset_revision)
        predictor = self.store.load_bundle(model_name, model_info["revision"])
        table = self.store.load_table(dataset_name, data_info["revision"])
        result = classification_score(predictor, table, threshold, group_field)
        metrics = {"overall": result["metrics"], "groups": result["groups"],
                   "threshold": threshold, "model_digest": result["model_digest"],
                   "data_digest": result["data_digest"]}
        saved = self.store.save_evaluation(run_id, (model_name, model_info["revision"]),
                    (dataset_name, data_info["revision"]), result["predictions"], metrics)
        return {**saved, **result}

    def collinearity(self, dataset_name, feature_spec, dataset_revision=None):
        prepared = self.encoded(dataset_name, feature_spec, dataset_revision)
        return collinearity(prepared["frame"])

    def prediction_batches(self, model_name, dataset_name, size=256,
                           model_revision=None, dataset_revision=None):
        model_info = self.store.bundle_info(model_name, model_revision)
        data_info = self.store.table_info(dataset_name, dataset_revision)
        bundle = self.store.load_bundle(model_name, model_info["revision"])
        for batch in self.store.batches(dataset_name, data_info["revision"], size):
            yield {"dataset": [dataset_name, data_info["revision"]],
                   "model": [model_name, model_info["revision"]],
                   "dataset_digest": data_info["digest"], "model_digest": model_info["digest"],
                   "population": list(batch.ids()), "predictions": bundle.predict(batch)}

    def regression_batches(self, model_name, dataset_name, size=256,
                           model_revision=None, dataset_revision=None):
        from .metrics import RegressionAccumulator
        model_info = self.store.bundle_info(model_name, model_revision)
        data_info = self.store.table_info(dataset_name, dataset_revision)
        bundle = self.store.load_bundle(model_name, model_info["revision"])
        if not isinstance(bundle, Predictor):
            raise SchemaError("batch regression scoring requires a predictor")
        accumulator = RegressionAccumulator()
        for batch in self.store.batches(dataset_name, data_info["revision"], size):
            weights = dict(zip(batch.ids(), batch.weights()))
            accumulator.update(targets(batch, bundle.target), bundle.predict(batch), weights)
        return {"dataset": [dataset_name, data_info["revision"]],
                "model": [model_name, model_info["revision"]], "metrics": accumulator.result()}

    def neighbor_intervals(self, model_name, dataset_name, level=.9, model_revision=None, dataset_revision=None):
        from .estimators import Neighbors
        from .metrics import interval_score, pinball
        model = self.store.load_bundle(model_name, model_revision)
        if not isinstance(model, Predictor) or not isinstance(model.model, Neighbors):
            raise SchemaError("empirical prediction intervals require a neighbor predictor")
        if not 0 < level < 1:
            raise SchemaError("interval level must be inside (0,1)")
        table = self.store.load_table(dataset_name, dataset_revision)
        tail = (1 - level) / 2
        estimates = model.model.predict_quantiles(model.encoded(table), (tail, 1 - tail))
        lower = {key: values[str(tail)] for key, values in estimates.items()}
        upper = {key: values[str(1 - tail)] for key, values in estimates.items()}
        truth = targets(table, model.target)
        weights = dict(zip(table.ids(), table.weights()))
        return {"lower": lower, "upper": upper,
                "metrics": interval_score(truth, lower, upper, level, weights),
                "lower_pinball": pinball(truth, lower, tail, weights),
                "upper_pinball": pinball(truth, upper, 1 - tail, weights),
                "interpretation": "empirical neighbor intervals; no coverage guarantee"}
