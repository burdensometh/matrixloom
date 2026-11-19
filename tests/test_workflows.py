import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from matrixloom import Analysis, Field, Schema, Table
from matrixloom.table import Sample
from matrixloom.errors import IdentityError, MatrixError, StateError
from matrixloom.pipeline import Features, Predictor, Clusterer, restore_bundle
from matrixloom.splits import Split, random_split, kfold, time_split, stratified_split
from matrixloom.evaluation import bootstrap_mean, classification_score


def observations():
    schema = Schema([Field("x", "Measurement", unit="cm"), Field("z", "Context"),
                     Field("category", "Category", "category"),
                     Field("site", "Site", "category", role="group"),
                     Field("time", "Observed", "timestamp", role="time"),
                     Field("y", "Response", role="target"),
                     Field("label", "Label", "integer", role="target")])
    samples = []
    for i in range(16):
        values = (float(i), float(i % 3) if i != 3 else None, "a" if i % 2 == 0 else "b",
                  "north" if i < 8 else "south", f"2020-01-{i+1:02}T00:00:00Z", 2 * i + 1, int(i >= 8))
        samples.append(Sample(f"r{i:02}", values, 1 + (i % 2)))
    return Table(schema, samples)


SPEC = {"fields": ["x", "z", "category"], "unknown": "zero",
        "transforms": [{"kind": "impute"}, {"kind": "standardize"}]}

NEIGHBOR_SPEC = {"fields": ["x", "z", "category"], "unknown": "zero",
                 "transforms": [{"kind": "neighbor_impute", "neighbors": 3},
                                {"kind": "standardize"}]}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.app = Analysis(Path(self.temporary.name) / "analysis")
        self.table = observations()
        self.app.ingest("training", self.table)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fitted_state_survives_reopen_and_feature_reordering(self):
        result = self.app.train("linear", "training", SPEC, {"kind": "ridge", "alpha": .1}, "y")
        initial = self.app.predict("linear", "training")
        reordered = self.table.select(reversed(self.table.schema.ids()))
        self.app.ingest("reordered", reordered)
        self.assertEqual(initial, self.app.predict("linear", "reordered"))
        reopened = Analysis(self.app.home)
        self.assertEqual(initial, reopened.predict("linear", "training"))
        model = reopened.store.load_bundle("linear")
        self.assertEqual(restore_bundle(model.state()).digest(), model.digest())
        self.assertEqual(model.features.fresh().spec, model.features.spec)
        self.assertEqual(model.encoded(self.table).feature_ids(), model.features.output_features and tuple(f.id for f in model.features.output_features))
        evaluated = reopened.evaluate("evaluation-1", "linear", "training", group_field="site")
        self.assertEqual(evaluated["population"], list(self.table.ids()))
        self.assertTrue(reopened.evaluate("evaluation-1", "linear", "training", group_field="site")["replayed"])
        stored = reopened.store.evaluation("evaluation-1")
        self.assertEqual(stored["predictions"], initial)
        self.assertEqual(result["revision"], 1)

    def test_original_training_population_is_bound(self):
        self.app.train("linear", "training", SPEC, {"kind": "ridge"}, "y")
        model = self.app.store.load_bundle("linear")
        self.app.ingest("different", self.table.replace_column("y", [1] * len(self.table)))
        with self.assertRaises(StateError):
            self.app.store.save_bundle("bad", model, ("different", 1))
        with self.assertRaises(StateError):
            self.app.ingest("training", self.table, expected_revision=0)
        next_revision = self.app.ingest("training", self.table, expected_revision=1)
        self.assertEqual(next_revision["revision"], 2)
        self.assertEqual(len(self.app.store.revisions("training")), 2)
        with self.assertRaises(IdentityError):
            self.app.store.select_samples("training", ["absent"])

    def test_bundle_export_import_and_integrity(self):
        self.app.train("linear", "training", SPEC, {"kind": "ridge"}, "y")
        text = self.app.store.export_bundle("linear")
        self.app.store.import_bundle("copy", text)
        self.assertEqual(self.app.predict("linear", "training"), self.app.predict("copy", "training"))
        self.assertEqual(len(self.app.store.revisions("copy", "bundle")), 1)
        envelope = json.loads(text)
        envelope["payload"]["target"] = "wrong"
        with self.assertRaises(StateError):
            self.app.store.import_bundle("tampered", json.dumps(envelope))
        connection = self.app.store.connect()
        connection.execute("UPDATE samples SET weight=999 WHERE dataset_name='training' AND sample_id='r00'")
        connection.close()
        with self.assertRaises(StateError):
            self.app.store.load_table("training")

    def test_batches_pin_revisions_and_match_full_predictions(self):
        self.app.train("linear", "training", SPEC, {"kind": "ridge"}, "y")
        batches = list(self.app.prediction_batches("linear", "training", size=3))
        joined = {key: value for batch in batches for key, value in batch["predictions"].items()}
        self.assertEqual(joined, self.app.predict("linear", "training"))
        batch_score = self.app.regression_batches("linear", "training", size=3)["metrics"]
        full = self.app.evaluate("batch-reference", "linear", "training")["metrics"]
        for key in batch_score:
            self.assertAlmostEqual(batch_score[key], full[key])
        ids = ["r10", "r00", "r15"]
        self.assertEqual(self.app.store.select_samples("training", ids).ids(), tuple(ids))

    def test_holdout_and_learning_never_fit_on_test_rows(self):
        split = random_split(self.table, .25, seed=7)
        self.assertTrue(split.validate(self.table, exhaustive=True))
        self.assertEqual(Split.restore(split.state()), split)
        result = self.app.holdout("training", SPEC, {"kind": "ridge"}, "y", split)
        self.assertEqual(set(result["predictor"].training_ids), set(split.train))
        folds = kfold(self.table, 4)
        self.assertEqual(sorted(key for fold in folds for key in fold.test), sorted(self.table.ids()))
        temporal = time_split(self.table, "time", "2020-01-09T00:00:00Z")
        self.assertEqual(len(temporal.train), 8)
        stratified = stratified_split(self.table, "category", .25)
        self.assertTrue(stratified.validate(self.table, True))
        curves = self.app.learning_curve("training", split, [.5, 1], SPEC, {"kind": "ridge"}, "y")
        self.assertEqual(len(curves), 2)
        self.assertTrue(set(curves[0]["training_ids"]) <= set(curves[1]["training_ids"]))
        self.assertEqual(bootstrap_mean([2, 2, 2], repeats=20)["lower"], 2)

    def test_clustering_reopens_and_scores_real_assignments(self):
        result = self.app.cluster("clusters", "training", SPEC, {"kind": "kmeans", "clusters": 2})
        self.assertEqual(len(result["assignments"]), 16)
        self.assertEqual(self.app.predict("clusters", "training"), result["assignments"])
        self.assertGreater(self.app.cluster_quality("clusters", "training")["mean"], -1)
        clusterer = self.app.store.load_bundle("clusters")
        self.assertEqual(Clusterer.restore(clusterer.state()).predict(self.table), result["assignments"])

    def test_classifier_uses_probabilities_and_population_weights(self):
        spec = {"fields": ["x"], "transforms": [{"kind": "standardize"}]}
        self.app.train("binary", "training", spec, {"kind": "logistic"}, "label")
        result = self.app.classify("binary-score", "binary", "training", group_field="site")
        self.assertGreater(result["metrics"]["roc_auc"], .9)
        self.assertEqual(result["groups"]["north"]["ranking_status"], "SINGLE_OBSERVED_CLASS")
        self.assertEqual(classification_score(self.app.store.load_bundle("binary"), self.table)["predictions"], result["predictions"])
        self.assertTrue(self.app.classify("binary-score", "binary", "training", group_field="site")["replayed"])

    def test_analysis_consumers(self):
        self.app.train("ridge", "training", SPEC, {"kind": "ridge"}, "y")
        self.app.train("neighbors", "training", SPEC, {"kind": "neighbors", "neighbors": 3}, "y")
        self.assertEqual(self.app.inspect("training", ["x", "z"], group_field="site")["summary"]["x"]["rows"], 16)
        self.assertGreaterEqual(self.app.importance("ridge", "training", ["x"], repeats=2)["fields"]["x"]["mean_mse_increase"], 0)
        self.assertEqual(self.app.drift("training", "training", ["x", "category"])["category"]["total_variation"], 0)
        self.assertEqual(len(self.app.category_frequencies("training", ["category"])["category"]), 2)
        self.assertEqual(self.app.encoded("training", SPEC)["frame"].shape()[0], 16)
        self.assertEqual(self.app.collinearity("training", {"fields": ["x", "z"], "transforms": [{"kind": "impute"}]})["status"], "OK")
        self.assertIn("mae_difference", self.app.compare("training", "ridge", "neighbors"))
        self.assertGreaterEqual(self.app.neighbor_intervals("neighbors", "training")["metrics"]["coverage"], 0)

    def test_public_ingestion_paths(self):
        self.app.ingest_jsonl("json", self.table.schema, self.table.to_jsonl())
        self.app.ingest_csv("csv", self.table.schema, self.table.to_csv())
        self.assertEqual(self.app.store.load_table("json").digest(), self.app.store.load_table("csv").digest())
        self.app.ingest("metric-units", self.table, units={"x": "m"})
        self.assertEqual(self.app.store.load_table("metric-units").schema.field("x").unit, "m")

    def test_neighbor_imputation_enters_prediction_chain_and_reopens(self):
        result = self.app.train("knn-fill", "training", NEIGHBOR_SPEC,
                                {"kind": "ridge", "alpha": .1}, "y")
        initial = self.app.predict("knn-fill", "training")
        bundle = self.app.store.load_bundle("knn-fill")
        # The donor store and training-only scales are part of the persisted bundle.
        stage = bundle.features.stages[0]
        self.assertEqual(stage.kind, "neighbor_impute")
        self.assertEqual(stage.donors.ids, self.table.ids())
        reopened = Analysis(self.app.home)
        reloaded = reopened.store.load_bundle("knn-fill")
        self.assertEqual(reloaded.predict(self.table), initial)
        self.assertEqual(restore_bundle(reloaded.state()).digest(), bundle.digest())
        # Export/import keeps the donor provenance working end to end.
        text = self.app.store.export_bundle("knn-fill")
        self.app.store.import_bundle("knn-copy", text)
        self.assertEqual(self.app.predict("knn-copy", "training"), initial)
        # Prediction never updates training statistics: repeated prediction is identical.
        self.assertEqual(reloaded.predict(self.table), reloaded.predict(self.table))
        # Provenance is reachable through the feature pipeline and names original donors.
        sources = bundle.features.fill_sources(self.table)
        self.assertIn("z", sources["r03"])
        self.assertEqual(sources["r03"]["z"]["strategy"], "neighbors")
        self.assertTrue(all(donor["id"] in set(self.table.ids())
                            for donor in sources["r03"]["z"]["donors"]))
        self.assertNotIn("y", sources["r03"])
        # The original observations are immutable across repeated transforms.
        encoded_first = bundle.features.transform(self.table)
        encoded_second = bundle.features.transform(self.table)
        self.assertEqual(encoded_first.state(), encoded_second.state())
        self.assertEqual(result["revision"], 1)

    def test_neighbor_imputation_uses_original_mask_and_feeds_clusterer(self):
        # b is missing x1; its nearest donor on the originally shared x0 dimension is c.
        schema = Schema([Field("x0", "X0"), Field("x1", "X1"),
                         Field("label", "Label", "integer", role="target")])
        rows = [Sample("a", (-3.0, 99.0, 0)), Sample("b", (0.0, None, 0)),
                Sample("c", (1.0, 10.0, 1)), Sample("d", (2.0, 20.0, 1))]
        table = Table(schema, rows)
        self.app.ingest("sparse", table)
        spec = {"fields": ["x0", "x1"],
                "transforms": [{"kind": "neighbor_impute", "neighbors": 1,
                                "distance_weighted": False}]}
        predictor = Predictor(spec, {"kind": "ridge", "alpha": 0.0}, "label").fit(table)
        source = predictor.features.fill_sources(table)["b"]["x1"]
        self.assertEqual([donor["id"] for donor in source["donors"]], ["c"])
        self.assertEqual(source["value"], 10.0)
        # Replaying the fill on the now-complete frame with the original mask reproduces
        # the identical values and provenance: imputed entries never become observations.
        imputer = predictor.features.stages[0]
        encoded = predictor.features.encoder.transform(table)
        original_mask = [[value is not None for value in row] for row in encoded.values]
        filled = imputer.transform(encoded)
        replayed = imputer.transform_with_mask(filled, original_mask)
        self.assertEqual(replayed.values, filled.values)
        replay_sources = imputer.fill_sources(filled, original_mask)
        self.assertEqual([donor["id"] for donor in replay_sources["b"]["x1"]["donors"]], ["c"])
        self.assertEqual(replay_sources["b"]["x1"]["value"], source["value"])
        self.assertIsNone(table.sample("b").values[1])
        # The same imputation stage supports the unsupervised prediction chain as well.
        clustered = self.app.cluster("sparse-clusters", "sparse", spec,
                                     {"kind": "kmeans", "clusters": 2})
        self.assertEqual(len(clustered["assignments"]), len(table))
        self.assertEqual(Analysis(self.app.home).predict("sparse-clusters", "sparse"),
                         clustered["assignments"])

    def test_failure_keeps_previous_fitted_state(self):
        predictor = Predictor(SPEC, {"kind": "ridge"}, "y").fit(self.table)
        before = predictor.digest()
        with self.assertRaises(MatrixError):
            predictor.fit(self.table.replace_column("y", [None] * len(self.table)))
        self.assertEqual(before, predictor.digest())
        state = copy.deepcopy(predictor.features.state())
        state["output_features"][0]["id"] = "changed"
        with self.assertRaises(StateError):
            Features.restore(state)
