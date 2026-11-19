import copy
from pathlib import Path
import tempfile
import unittest

from matrixloom import Analysis, Field, Schema, Table
from matrixloom.errors import IdentityError, MatrixError, SchemaError, StateError
from matrixloom.frame import Feature, Frame
from matrixloom.graph import FeatureGraph, merge_frames
from matrixloom.pipeline import Features, Predictor, create_features, restore_features
from matrixloom.schema import canonical
from matrixloom.table import Sample


def observations(count=12):
    schema = Schema([Field("x", "Measurement", unit="cm"), Field("z", "Context"),
                     Field("category", "Category", "category"),
                     Field("y", "Response", role="target")])
    samples = []
    for i in range(count):
        values = (float(i), float((i * 7) % 4), "a" if i % 2 == 0 else "b", 0.5 * i * i + 1.0)
        samples.append(Sample(f"s{i:02}", values, 1 + (i % 2)))
    return Table(schema, samples)


def branching_spec():
    """Standardized measurements merged with nonlinear interaction products.

    The polynomial stage repeats its linear inputs, so a ``select`` node keeps
    only the product identity; that keeps the two branches disjoint and the
    merge therefore has no ambiguous lineage.
    """

    return {
        "nodes": [
            {"id": "raw", "kind": "source", "fields": ["x", "z"], "unknown": "zero"},
            {"id": "numeric", "kind": "transform", "inputs": ["raw"],
             "transform": {"kind": "standardize"}},
            {"id": "expansion", "kind": "transform", "inputs": ["numeric"],
             "transform": {"kind": "polynomial", "degree": 2, "interactions_only": True,
                           "max_features": 16}},
            {"id": "interaction", "kind": "transform", "inputs": ["expansion"],
             "transform": {"kind": "select", "ids": ["product:" + canonical(["x", "z"])]}},
            {"id": "joined", "kind": "merge", "inputs": ["numeric", "interaction"]},
        ],
        "outputs": ["joined"],
    }


class GraphConstructionTests(unittest.TestCase):
    def test_branching_graph_shares_one_fitted_source(self):
        table = observations()
        graph = FeatureGraph(**branching_spec())
        frame = graph.fit_transform(table)
        encoder = graph.states["raw"]["encoder"]
        self.assertIs(graph.states["raw"]["encoder"], encoder)
        standardizer = graph.states["numeric"]["stage"]
        self.assertEqual(standardizer.features, encoder.features)
        self.assertEqual(frame.shape()[0], len(table))
        ids = frame.feature_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("x", ids)
        self.assertIn("product:" + canonical(["x", "z"]), ids)
        self.assertEqual(frame.features, graph.output_features)

    def test_shared_source_is_fitted_once_from_training_rows(self):
        table = observations()
        graph = FeatureGraph(**branching_spec()).fit(table)
        means = {}
        for feature in graph.states["raw"]["encoder"].features:
            means[feature.id] = feature
        before = copy.deepcopy(graph.states["raw"]["encoder"].state())
        subset = table.take(table.ids()[:3])
        graph.transform(subset)
        self.assertEqual(graph.states["raw"]["encoder"].state(), before)
        self.assertEqual(graph.training_digest, table.digest())

    def test_state_round_trip_reproduces_predictions(self):
        table = observations()
        graph = FeatureGraph(**branching_spec()).fit(table)
        restored = FeatureGraph.restore(graph.state())
        self.assertEqual(restored.state(), graph.state())
        self.assertEqual(restored.transform(table).state(), graph.transform(table).state())
        self.assertEqual(restored.training_digest, table.digest())

    def test_shared_node_is_fitted_exactly_once(self):
        from unittest import mock
        from matrixloom.encoding import Encoder
        table = observations()
        original = Encoder.fit
        calls = []
        def counting(self, dataset):
            calls.append(self)
            return original(self, dataset)
        with mock.patch.object(Encoder, "fit", counting):
            FeatureGraph(**branching_spec()).fit(table)
        self.assertEqual(len(calls), 1)

    def test_source_failure_leaves_previous_state_intact(self):
        table = observations()
        graph = FeatureGraph(**branching_spec()).fit(table)
        before = graph.digest()
        broken = table.replace_column("y", [None] * len(table))
        broken = broken.replace_column("x", [None] * len(table))
        with self.assertRaises(MatrixError):
            graph.fit(broken)
        self.assertEqual(graph.digest(), before)
        self.assertEqual(graph.transform(table).state(), graph.fit(table).transform(table).state())

    def test_restore_rejects_tampered_axes_and_states(self):
        table = observations()
        graph = FeatureGraph(**branching_spec()).fit(table)
        changed = copy.deepcopy(graph.state())
        changed["output_features"][0]["id"] = "renamed"
        with self.assertRaises(StateError):
            FeatureGraph.restore(changed)
        changed = copy.deepcopy(graph.state())
        changed["nodes"]["raw"]["encoder"]["features"][0]["id"] = "other"
        with self.assertRaises(StateError):
            FeatureGraph.restore(changed)
        changed = copy.deepcopy(graph.state())
        changed["spec"]["nodes"][0]["fields"] = ["z"]
        with self.assertRaises(StateError):
            FeatureGraph.restore(changed)
        changed = copy.deepcopy(graph.state())
        changed["nodes"]["numeric"]["stage"]["centers"][0] += 100
        restored = FeatureGraph.restore(changed)
        self.assertNotEqual(restored.transform(table).state(), graph.transform(table).state())

    def test_graph_outputs_can_be_many(self):
        specification = branching_spec()
        specification["outputs"] = ["numeric", "interaction"]
        graph = FeatureGraph(**specification).fit(observations())
        self.assertIn("x", graph.output_features and graph.transform(observations()).feature_ids())
        restored = FeatureGraph.restore(graph.state())
        self.assertEqual(restored.output_features, graph.output_features)


class GraphRejectionTests(unittest.TestCase):
    def test_cycles_are_rejected_before_execution(self):
        specification = {"nodes": [
            {"id": "a", "kind": "transform", "inputs": ["b"], "transform": {"kind": "standardize"}},
            {"id": "b", "kind": "transform", "inputs": ["a"], "transform": {"kind": "standardize"}},
            {"id": "raw", "kind": "source", "fields": ["x"]},
        ], "outputs": ["a"]}
        with self.assertRaises(SchemaError):
            FeatureGraph(**specification)
        with self.assertRaises(SchemaError):
            FeatureGraph(nodes=[{"id": "a", "kind": "transform", "inputs": ["a"],
                                 "transform": {"kind": "standardize"}}], outputs=["a"])

    def test_missing_dependency_is_rejected(self):
        with self.assertRaises(SchemaError):
            FeatureGraph(nodes=[{"id": "a", "kind": "transform", "inputs": ["absent"],
                                 "transform": {"kind": "standardize"}}], outputs=["a"])

    def test_self_reference_and_duplicate_ids(self):
        with self.assertRaises(SchemaError):
            FeatureGraph(nodes=[{"id": "raw", "kind": "source"}, {"id": "raw", "kind": "source"}],
                         outputs=["raw"])

    def test_duplicate_output_identity_conflict_is_rejected(self):
        specification = {"nodes": [
            {"id": "left", "kind": "source", "fields": ["x"]},
            {"id": "right", "kind": "source", "fields": ["x"]},
            {"id": "join", "kind": "merge", "inputs": ["left", "right"]},
        ], "outputs": ["join"]}
        graph = FeatureGraph(**specification)
        with self.assertRaises(SchemaError):
            graph.fit(observations())

    def test_conflicting_numeric_outputs_are_rejected(self):
        specification = {"nodes": [
            {"id": "left", "kind": "source", "fields": ["x"]},
            {"id": "right", "kind": "transform", "inputs": ["left"],
             "transform": {"kind": "range"}},
        ], "outputs": ["left", "right"]}
        graph = FeatureGraph(**specification)
        with self.assertRaises(SchemaError):
            graph.fit(observations())

    def test_category_shared_through_merge_is_rejected(self):
        specification = {"nodes": [
            {"id": "left", "kind": "source", "fields": ["category"]},
            {"id": "right", "kind": "source", "fields": ["category"]},
            {"id": "join", "kind": "merge", "inputs": ["left", "right"]},
        ], "outputs": ["join"]}
        with self.assertRaises(SchemaError):
            FeatureGraph(**specification).fit(observations())

    def test_unknown_node_options_and_kinds_are_rejected(self):
        for node in ({"id": "a", "kind": "source", "colour": "red"},
                     {"id": "a", "kind": "mystery"}):
            with self.assertRaises(SchemaError):
                FeatureGraph(nodes=[node], outputs=["a"])

    def test_lineage_conflict_is_rejected_before_any_node_executes(self):
        from unittest import mock
        from matrixloom.encoding import Encoder
        specification = {"nodes": [
            {"id": "left", "kind": "source", "fields": ["x"]},
            {"id": "right", "kind": "source", "fields": ["x"]},
            {"id": "join", "kind": "merge", "inputs": ["left", "right"]},
        ], "outputs": ["join"]}
        calls = []
        def counting(self, dataset):
            calls.append(dataset)
            raise AssertionError("no node may fit when preflight already failed")
        with mock.patch.object(Encoder, "fit", counting):
            with self.assertRaises(SchemaError):
                FeatureGraph(**specification).fit(observations())
        self.assertEqual(calls, [])

    def test_merge_rejects_population_and_weight_mismatch(self):
        left = Frame(["a", "b"], [Feature("x", "x")], [[1], [2]], [1, 1])
        right = Frame(["b", "a"], [Feature("q", "q")], [[3], [4]], [1, 1])
        self.assertEqual(merge_frames([left, right]).values, ((1.0, 4.0), (2.0, 3.0)))
        with self.assertRaises(IdentityError):
            merge_frames([left, Frame(["a"], [Feature("q", "q")], [[5]])])
        with self.assertRaises(IdentityError):
            merge_frames([left, Frame(["a", "b"], [Feature("q", "q")], [[5], [6]], [2, 1])])
        with self.assertRaises(SchemaError):
            merge_frames([left, Frame(["a", "b"], [Feature("x", "x")], [[5], [6]])])


class GraphPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.home = Path(self.temporary.name) / "analysis"
        self.app = Analysis(self.home)
        self.table = observations()
        self.app.ingest("training", self.table)

    def tearDown(self):
        self.temporary.cleanup()

    def test_graph_trains_and_reopens_through_analysis(self):
        specification = {"kind": "graph", **branching_spec()}
        result = self.app.train("graph-model", "training", specification,
                                {"kind": "ridge", "alpha": .1}, "y")
        predictions = self.app.predict("graph-model", "training")
        self.assertEqual(len(predictions), len(self.table))
        self.assertEqual(result["training_rows"], len(self.table))
        reopened = Analysis(self.home)
        self.assertEqual(reopened.predict("graph-model", "training"), predictions)
        bundle = reopened.store.load_bundle("graph-model")
        self.assertIsInstance(bundle.features, FeatureGraph)
        self.assertEqual(Predictor.restore(bundle.state()).predict(self.table), predictions)
        self.assertEqual(reopened.evaluate("graph-run", "graph-model", "training")["metrics"]["rows"],
                         len(self.table))

    def test_graph_export_import_round_trip(self):
        specification = {"kind": "graph", **branching_spec()}
        self.app.train("graph-model", "training", specification, {"kind": "ridge"}, "y")
        text = self.app.store.export_bundle("graph-model")
        self.app.store.import_bundle("graph-copy", text)
        self.assertEqual(self.app.predict("graph-model", "training"),
                         self.app.predict("graph-copy", "training"))

    def test_batched_prediction_matches_full_prediction(self):
        specification = {"kind": "graph", **branching_spec()}
        self.app.train("graph-model", "training", specification, {"kind": "ridge"}, "y")
        batches = list(self.app.prediction_batches("graph-model", "training", size=5))
        joined = {key: value for batch in batches for key, value in batch["predictions"].items()}
        self.assertEqual(joined, self.app.predict("graph-model", "training"))

    def test_dependent_fitted_state_is_read_only_across_predictions(self):
        specification = {"kind": "graph", **branching_spec()}
        self.app.train("graph-model", "training", specification, {"kind": "ridge"}, "y")
        model = self.app.store.load_bundle("graph-model")
        state = model.features.state()
        model.predict(self.table.take(self.table.ids()[:4]))
        self.assertEqual(model.features.state(), state)


class GraphApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.home = Path(self.temporary.name) / "analysis"
        self.app = Analysis(self.home)
        self.table = observations()
        self.app.ingest("training", self.table)

    def tearDown(self):
        self.temporary.cleanup()

    def test_persisted_graph_applies_frozen_state(self):
        specification = branching_spec()
        saved = self.app.graph("features", "training", specification["nodes"], specification["outputs"])
        applied = self.app.graph_frame("features", "training")
        self.assertEqual(applied["frame"].ids, self.table.ids())
        reopened = Analysis(self.home)
        again = reopened.graph_frame("features", "training")
        self.assertEqual(again["frame"].state(), applied["frame"].state())
        self.assertEqual(saved["training_rows"], len(self.table))
        subset = self.table.take(self.table.ids()[:2])
        self.app.ingest("subset", subset)
        reduced = reopened.graph_frame("features", "subset")["frame"]
        full = applied["frame"]
        for key in reduced.ids:
            self.assertEqual(reduced.row(key), full.row(key))

    def test_graph_bundle_kind_is_recognised_by_storage(self):
        specification = branching_spec()
        self.app.graph("features", "training", specification["nodes"], specification["outputs"])
        info = self.app.store.bundle_info("features")
        self.assertEqual(info["kind"], "graph")
        text = self.app.store.export_bundle("features")
        self.app.store.import_bundle("features-copy", text)
        self.assertEqual(self.app.graph_frame("features", "training")["frame"].state(),
                         self.app.graph_frame("features-copy", "training")["frame"].state())


class LegacyCompatibilityTests(unittest.TestCase):
    def test_sequential_specs_still_restore_and_predict_identically(self):
        table = observations()
        spec = {"fields": ["x", "z", "category"], "unknown": "zero",
                "transforms": [{"kind": "impute"}, {"kind": "standardize"}]}
        before = Predictor(spec, {"kind": "ridge"}, "y").fit(table)
        state = copy.deepcopy(before.state())
        restored = Predictor.restore(state)
        self.assertIsInstance(restored.features, Features)
        self.assertEqual(restored.predict(table), before.predict(table))
        self.assertEqual(restored.digest(), before.digest())
        self.assertIsInstance(create_features(spec), Features)
        self.assertEqual(restore_features(restored.features.state()).spec, restored.features.spec)

    def test_sequence_node_matches_standalone_features(self):
        table = observations()
        spec = {"fields": ["x", "z"], "transforms": [{"kind": "standardize"}]}
        standalone = Features(**spec).fit_transform(table)
        graph = FeatureGraph(nodes=[{"id": "seq", "kind": "sequence", "feature_spec": spec}],
                             outputs=["seq"]).fit(table)
        self.assertEqual(graph.transform(table).state(), standalone.state())
        self.assertEqual(FeatureGraph.restore(graph.state()).transform(table).state(),
                         standalone.state())

    def test_graph_with_sequence_branch_and_numeric_branch_merge(self):
        table = observations()
        specification = {"nodes": [
            {"id": "raw", "kind": "source", "fields": ["x", "z"], "unknown": "zero"},
            {"id": "scaled", "kind": "sequence",
             "feature_spec": {"fields": ["x", "z"], "transforms": [{"kind": "standardize"}]}},
            {"id": "join", "kind": "merge", "inputs": ["raw", "scaled"]},
        ], "outputs": ["join"]}
        with self.assertRaises(SchemaError):
            FeatureGraph(**specification).fit(table)
