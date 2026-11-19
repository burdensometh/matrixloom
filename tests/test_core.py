import json
import math
import unittest

from matrixloom.errors import IdentityError, MatrixError, NumericalError, SchemaError, StateError
from matrixloom.schema import Field, Schema, canonical, timestamp
from matrixloom.table import Sample, Table
from matrixloom.frame import Feature, Frame, numerical_frame
from matrixloom.encoding import Encoder
from matrixloom import linalg as la
from matrixloom import statistics as stats
from matrixloom import metrics
from matrixloom.transforms import (Imputer, Standardizer, RangeScaler, Winsorizer,
                                   Polynomial, FeatureSelector, restore_transform, create_transform)
from matrixloom.expressions import Derivations, Expression
from matrixloom.estimators import Ridge, Neighbors, KMeans, Logistic, create_model, restore_model
from matrixloom.units import normalize, conversion, convert_values
from matrixloom.diagnostics import (associations, category_shift, collinearity, contingency,
                                   drift, missing_by_group, outliers, weighted_ks)


def frame(values, weights=None):
    width = len(values[0])
    return Frame([f"s{i}" for i in range(len(values))], [Feature(f"x{i}", f"x{i}") for i in range(width)], values, weights)


class SchemaTests(unittest.TestCase):
    def test_types_and_time(self):
        self.assertEqual(timestamp("2020-01-01T08:00:00+08:00"), "2020-01-01T00:00:00.000000+00:00")
        self.assertEqual(Field("n", "N", "integer").parse_text("12"), 12)
        self.assertEqual(Field("n", "N", "integer").parse(9007199254740993), 9007199254740993)
        self.assertEqual(Field("b", "B", "boolean").parse_text("false"), False)
        self.assertEqual(Field("t", "T", "timestamp").parse("2020-01-01T00:00:00Z"), timestamp("2020-01-01T00:00:00Z"))
        for action in [lambda: timestamp("2020-01-01"), lambda: Field("x", "X").parse(True),
                       lambda: Field("x", "X", nullable=False).parse(None),
                       lambda: Field("x", "X").parse(float("nan")),
                       lambda: Field("x", "X", "integer").parse_text("1.2")]:
            with self.assertRaises(MatrixError):
                action()

    def test_schema_identity(self):
        schema = Schema([Field("a", "Alpha"), Field("b", "Beta", "category")])
        restored = Schema.restore(schema.state())
        self.assertEqual(schema.digest(), restored.digest())
        self.assertTrue(schema.compatible(schema.select(["b", "a"])))
        self.assertEqual(schema.rename({"a": "New"}).field("a").name, "New")
        self.assertEqual(schema.without(["b"]).ids(), ("a",))
        self.assertEqual(schema.parse_record({"Alpha": 1, "Beta": "z"}, by_name=True), (1.0, "z"))
        with self.assertRaises(SchemaError):
            Schema([Field("a", "One"), Field("a", "Two")])

    def test_transport_preserves_missing_and_literals(self):
        schema = Schema([Field("x", "X"), Field("c", "C", "category"), Field("b", "B", "boolean")])
        table = Table(schema, [Sample("a", (None, "", True)), Sample("b", (3, "\\N", False), 2),
                               Sample("c", (4, "x,\nq", None))])
        self.assertEqual(Table.from_jsonl(schema, table.to_jsonl()).state(), table.state())
        self.assertEqual(Table.from_csv(schema, table.to_csv()).state(), table.state())
        self.assertEqual(Table.restore(table.state()).digest(), table.digest())
        self.assertEqual(table.take(["b", "a"]).ids(), ("b", "a"))
        self.assertEqual(table.drop_missing(["x"]).ids(), ("b", "c"))
        self.assertEqual(table.filter("x", lambda x: x is not None and x > 3).ids(), ("c",))
        self.assertEqual(table.select(["c"]).column("c"), ("", "\\N", "x,\nq"))
        self.assertEqual(table.rename({"x": "New"}).schema.field("x").name, "New")
        self.assertEqual(table.replace_column("x", [1, 2, 3]).column("x"), (1.0, 2.0, 3.0))
        self.assertEqual(table.with_column(Field("q", "Q"), [1, 2, 3]).schema.ids()[-1], "q")
        self.assertEqual(len(table.take(["a"]).append(table.take(["b", "c"]))), 3)
        with self.assertRaises(IdentityError):
            table.take(["a", "a"])

    def test_lookup_and_units(self):
        schema = Schema([Field("x", "X", unit="cm"), Field("g", "G", "category", role="group")])
        table = Table(schema, [Sample("a", (100, "a")), Sample("b", (None, "b"))])
        converted = normalize(table, {"x": "m"})
        self.assertEqual(converted.column("x"), (1.0, None))
        self.assertEqual(convert_values([0, 100], "degC", "K"), (273.15, 373.15))
        with self.assertRaises(SchemaError):
            conversion("m", "kg")
        lookup = Table(Schema([Field("key", "Key", "category"), Field("q", "Q")]), [Sample("r", ("a", 7))])
        joined = table.join_lookup(lookup, "g", "key", ["q"])
        self.assertEqual(joined.column("lookup/q"), (7.0, None))
        self.assertEqual(numerical_frame(converted, ["x"]).values[0], (1.0,))


class MatrixTests(unittest.TestCase):
    def test_axis_and_population(self):
        f = frame([[1, 2], [3, None], [4, 5]], [1, 0, 2])
        self.assertEqual(f.shape(), (3, 2))
        self.assertEqual(f.row("s1"), (3.0, None))
        self.assertEqual(f.take(["s2", "s0"]).ids, ("s2", "s0"))
        self.assertEqual(f.select(["x1", "x0"]).align_features(f.features).values, f.values)
        self.assertEqual(f.positive_population().ids, ("s0", "s2"))
        self.assertEqual(f.with_weights({"s0": 2, "s1": 0, "s2": 1}).total_weight(), 3)
        self.assertEqual(Frame.restore(f.state()).digest(), f.digest())
        self.assertEqual(len(list(f.batches(2))), 2)
        self.assertEqual(f.missingness()["x1"]["missing"], 1)
        extra = Frame(f.ids, [Feature("q", "q")], [[7], [8], [9]], f.weights)
        self.assertEqual(f.hstack(extra).width, 3)
        self.assertEqual(len(f.take(["s0"]).vstack(f.take(["s1", "s2"]))), 3)
        with self.assertRaises(SchemaError):
            f.complete()

    def test_linear_algebra(self):
        a = [[4, 1], [1, 3]]
        solution = la.solve(a, [1, 2])
        self.assertAlmostEqual(solution[0], 1 / 11)
        self.assertAlmostEqual(solution[1], 7 / 11)
        lower = la.cholesky(a)
        reconstructed = la.matmul(lower, la.transpose(lower))
        self.assertAlmostEqual(reconstructed[1][1], 3)
        self.assertAlmostEqual(la.matmul(a, la.inverse(a))[0][0], 1)
        self.assertEqual(la.add([1, 2], [3, 4]), [4, 6])
        self.assertEqual(la.scale([1, 2], 3), [3, 6])
        self.assertEqual(la.outer([1, 2], [3]), [[3], [6]])
        self.assertEqual(la.diagonal([2, 3]), [[2, 0], [0, 3]])
        self.assertEqual(la.trace(a), 7)
        self.assertEqual(la.symmetrize([[2, 1], [3, 4]]), [[2, 2], [2, 4]])
        self.assertEqual(la.residuals([[1], [2]], [2, 5], [2]), [0, 1])
        self.assertEqual(la.quadratic([[2]], [3]), 18)
        with self.assertRaises(NumericalError):
            la.solve([[1, 2], [2, 4]], [1, 2])

    def test_weighted_and_missing_statistics(self):
        self.assertAlmostEqual(stats.mean([1, 3, None], [1, 3, 4]), 2.5)
        self.assertAlmostEqual(stats.variance([1, 3], [1, 1]), 1)
        self.assertAlmostEqual(stats.variance([1, 3], [1, 1], ddof=1), 2)
        self.assertEqual(stats.quantile([1, 2, 3], .5), 2)
        self.assertEqual(stats.absolute_deviation([1, 2, 3]), 1)
        f = frame([[1, 10], [3, None], [None, 14]])
        covariance = stats.covariance_matrix(f, pairwise=True)
        self.assertEqual(covariance["support"], [[2, 1], [1, 2]])
        self.assertEqual(covariance["covariance"], [[1.0, 0.0], [0.0, 4.0]])
        self.assertIsNone(stats.covariance_matrix(f, ddof=1, pairwise=True)["covariance"][0][1])
        self.assertEqual(stats.summarize(f)["x0"]["mean"], 2)
        self.assertEqual(stats.histogram([0, 1, 2, 3], [0, 1, 2])["weight"], [1, 2])
        self.assertAlmostEqual(stats.spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertEqual(stats.rank([3, 1, 1]), [3, 1.5, 1.5])

    def test_covariance_accumulator_matches_full_pairwise_calculation(self):
        import random
        random.seed(11)
        width, rows = 4, 48
        features = [Feature(f"c{i}", f"c{i}") for i in range(width)]
        offsets = (1e9, -1e9, 0.0, 42.0)
        values, weights, ids = [], [], []
        for r in range(rows):
            row = [None if random.random() < .3 else offsets[c] + random.gauss(0, .4)
                   for c in range(width)]
            values.append(row)
            weights.append(random.choice([0.0, .5, 1.0, 2.0]))
            ids.append(f"r{r:02}")
        full = Frame(ids, features, values, weights)
        for ddof in (0, 1):
            expected = stats.covariance_matrix(full, ddof=ddof, pairwise=True)
            accumulator = stats.CovarianceAccumulator(features, ddof)
            for start in range(0, rows, 5):
                accumulator.update(Frame(ids[start:start + 5], features,
                                         values[start:start + 5], weights[start:start + 5]))
            streamed = accumulator.result()
            self.assertEqual(streamed["support"], expected["support"])
            self.assertEqual(streamed["population"], "pairwise")
            for i in range(width):
                for j in range(width):
                    if expected["covariance"][i][j] is None:
                        self.assertIsNone(streamed["covariance"][i][j])
                    else:
                        self.assertAlmostEqual(streamed["covariance"][i][j],
                                               expected["covariance"][i][j], delta=1e-5)

    def test_covariance_merge_is_order_independent(self):
        features = [Feature("a", "a"), Feature("b", "b")]
        first = Frame(["p", "q"], features, [[1.0, 10.0], [3.0, None]])
        second = Frame(["r", "s"], features, [[5.0, 20.0], [7.0, 22.0]])
        full = Frame(["p", "q", "r", "s"], features,
                     [[1.0, 10.0], [3.0, None], [5.0, 20.0], [7.0, 22.0]])
        expected = stats.covariance_matrix(full, pairwise=True)
        left = stats.CovarianceAccumulator(features).update(first)
        left.merge(stats.CovarianceAccumulator(features).update(second))
        right = stats.CovarianceAccumulator(features).update(second)
        right.merge(stats.CovarianceAccumulator(features).update(first))
        self.assertEqual(left.result()["covariance"], right.result()["covariance"])
        self.assertEqual(left.result()["support"], expected["support"])
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(left.result()["covariance"][i][j],
                                       expected["covariance"][i][j], places=10)
        # direct per-batch covariance averaging must disagree: missingness differs
        c1 = stats.covariance_matrix(first, pairwise=True)["covariance"]
        c2 = stats.covariance_matrix(second, pairwise=True)["covariance"]
        self.assertNotAlmostEqual((c1[0][0] + c2[0][0]) / 2, expected["covariance"][0][0])

    def test_covariance_state_roundtrip_and_validation(self):
        features = [Feature("a", "a"), Feature("b", "b")]
        accumulator = stats.CovarianceAccumulator(features, ddof=1, training_digest="digest-x")
        accumulator.update(Frame(["p", "q", "r"], features,
                                 [[1.0, 4.0], [2.0, None], [3.0, 8.0]], [1, 2, 1]))
        restored = stats.CovarianceAccumulator.restore(accumulator.state())
        self.assertEqual(restored.digest(), accumulator.digest())
        self.assertEqual(restored.ddof, 1)
        self.assertEqual(restored.training_digest, "digest-x")
        self.assertEqual(restored.result()["covariance"], accumulator.result()["covariance"])
        empty = stats.CovarianceAccumulator(features)
        self.assertEqual(stats.CovarianceAccumulator.restore(empty.state()).result()["support"],
                         [[0, 0], [0, 0]])
        with self.assertRaises(SchemaError):
            stats.CovarianceAccumulator(features, ddof=2)
        with self.assertRaises(SchemaError):
            accumulator.merge(stats.CovarianceAccumulator(features, ddof=0))
        other_axis = [Feature("a", "a"), Feature("z", "z")]
        with self.assertRaises(SchemaError):
            accumulator.merge(stats.CovarianceAccumulator(other_axis))
        with self.assertRaises(SchemaError):
            accumulator.update(Frame(["p"], other_axis, [[1.0, 2.0]]))
        with self.assertRaises(IdentityError):
            stats.CovarianceAccumulator([Feature("a", "a"), Feature("a", "other")])

    def test_covariance_edge_populations(self):
        features = [Feature("a", "a"), Feature("b", "b")]
        # empty batch followed by real observations
        accumulator = stats.CovarianceAccumulator(features)
        accumulator.update(Frame([], features, []))
        accumulator.update(Frame(["p"], features, [[1.0, 2.0]]))
        self.assertEqual(accumulator.result()["support"], [[1, 1], [1, 1]])
        self.assertEqual(accumulator.result()["covariance"], [[0.0, 0.0], [0.0, 0.0]])
        # single paired point under ddof=1 is undefined, not zero
        single = stats.CovarianceAccumulator(features, ddof=1).update(
            Frame(["p"], features, [[1.0, 2.0]]))
        self.assertTrue(all(x is None for row in single.result()["covariance"] for x in row))
        # zero-weight rows contribute nothing and are not counted as support
        zero = stats.CovarianceAccumulator(features).update(
            Frame(["p"], features, [[1.0, 2.0]], [0.0]))
        self.assertEqual(zero.result()["support"], [[0, 0], [0, 0]])
        self.assertTrue(all(x is None for row in zero.result()["covariance"] for x in row))
        # missing entries are pairwise deleted, never treated as zero
        missing = stats.CovarianceAccumulator(features).update(
            Frame(["p", "q"], features, [[1.0, None], [3.0, 4.0]]))
        self.assertEqual(missing.result()["support"], [[2, 1], [1, 1]])
        self.assertEqual(missing.result()["covariance"][0][0], 1.0)
        self.assertEqual(missing.pair("b", "a").observed, 1)
        # large offset with small variation retains precision instead of collapsing
        big = Frame([f"r{i}" for i in range(3)], [Feature("v", "v")],
                    [[1e12 + 1.0], [1e12 + 2.0], [1e12 + 3.0]])
        variance = stats.covariance_matrix(big)["covariance"][0][0]
        self.assertAlmostEqual(variance, 2 / 3, delta=1e-7)
        # weighted ddof=1 stays on the same reliability-weight denominator as covariance()
        weighted = Frame(["p", "q", "r", "s"], features,
                         [[1.0, 2.0], [2.0, None], [3.0, 6.0], [4.0, 8.0]], [1, 2, .5, 1])
        profile = stats.CovarianceAccumulator(features, 1).update(weighted).result()
        self.assertAlmostEqual(profile["covariance"][0][1],
                               stats.covariance([1, 2, 3, 4], [2, None, 6, 8],
                                                [1, 2, .5, 1], ddof=1))
        with self.assertRaises(IdentityError):
            stats.CovarianceAccumulator(features).pair("a", "missing")

    def test_complete_row_covariance_uses_shared_population(self):
        features = [Feature("a", "a"), Feature("b", "b")]
        data = Frame(["p", "q", "r"], features,
                     [[1.0, 10.0], [3.0, None], [5.0, 20.0]])
        profile = stats.covariance_matrix(data, pairwise=False)
        self.assertEqual(profile["population"], "complete_rows")
        self.assertEqual(profile["support"], [[2, 2], [2, 2]])
        accumulator = stats.CovarianceAccumulator(features)
        complete = data.take(key for key, row in zip(data.ids, data.values)
                             if all(x is not None for x in row))
        accumulator.update(complete)
        self.assertEqual(accumulator.result("complete_rows")["covariance"],
                         profile["covariance"])


class FeatureTests(unittest.TestCase):
    def test_encoder_preserves_frozen_identity(self):
        schema = Schema([Field("n", "Numeric"), Field("c", "Category", "category")])
        table = Table(schema, [Sample("a", (1, "x")), Sample("b", (2, "y")), Sample("c", (None, None))])
        encoder = Encoder(unknown="zero").fit(table)
        f = encoder.transform(table)
        self.assertEqual(f.width, 3)
        changed = Table(schema.select(["c", "n"]), [Sample("d", ("new", 3))])
        self.assertEqual(encoder.transform(changed).values, ((3.0, 0.0, 0.0),))
        self.assertEqual(Encoder.restore(encoder.state()).transform(table).state(), f.state())
        self.assertEqual(Encoder().fit_transform(table).features, f.features)

    def test_fitted_transform_lifecycle(self):
        f = frame([[1, None], [3, 8], [100, 1000]], [1, 1, 0])
        imputer = Imputer(indicators=True).fit(f)
        filled = imputer.transform(f)
        self.assertEqual(filled.values[0], (1.0, 8.0, 0.0, 1.0))
        self.assertEqual(restore_transform(imputer.state()).transform(f).state(), filled.state())
        self.assertEqual(Imputer("median").fit_transform(f).values[0][1], 8)
        self.assertEqual(Imputer("constant", -1).fit_transform(f).values[0][1], -1)
        scaler = Standardizer().fit(f)
        transformed = scaler.transform(f)
        self.assertEqual(transformed.values[0][0], -1)
        self.assertEqual(scaler.inverse_transform(transformed).values, f.values)
        self.assertEqual(restore_transform(scaler.state()).transform(f).values, transformed.values)
        for stage in [RangeScaler(clip=True), Winsorizer(0, 1), Polynomial(2), FeatureSelector(["x0"])]:
            output = stage.fit_transform(f)
            self.assertEqual(restore_transform(stage.state()).transform(f).state(), output.state())
        self.assertEqual(create_transform({"kind": "impute"}).kind, "impute")

    def test_expressions_are_numeric_and_bounded(self):
        schema = Schema([Field("x", "X"), Field("y", "Y")])
        table = Table(schema, [Sample("a", (2, 3)), Sample("b", (None, 4))])
        definitions = [{"field": Field("z", "Z").state(), "expression": 'field("x") * field("y")'},
                       {"field": Field("q", "Q").state(), "expression": 'coalesce(field("z"), 0) + 1'}]
        derivations = Derivations(definitions)
        self.assertEqual(derivations.transform(table).column("q"), (7, 1))
        self.assertEqual(Derivations.restore(derivations.state()).order(schema), ("z", "q"))
        self.assertEqual(Expression('sqrt(4) + log(1) + maximum(2, 3)').evaluate({}), 5)
        self.assertEqual(Expression('seconds_between(field("a"), field("b"))').evaluate(
            {"a": "2020-01-01T00:01:00+00:00", "b": "2020-01-01T00:00:00+00:00"}), 60)
        for text in ['__import__("os")', 'field("x").real', '[1,2]', 'field(x)']:
            with self.assertRaises(SchemaError):
                Expression(text)


class EstimatorTests(unittest.TestCase):
    def test_ridge_identity_and_saved_state(self):
        f = frame([[0], [1], [2], [3]])
        y = {key: 2 * i + 1 for i, key in enumerate(f.ids)}
        ridge = Ridge(0).fit(f, y)
        self.assertAlmostEqual(ridge.coefficients[0], 2)
        self.assertAlmostEqual(ridge.offset, 1)
        self.assertEqual(restore_model(ridge.state()).predict(f), ridge.predict(f))
        self.assertAlmostEqual(sum(ridge.contributions(f)["s2"]["features"].values()) + ridge.offset, 5)
        self.assertEqual(create_model({"kind": "ridge"}).kind, "ridge")
        shuffled = f.take(reversed(f.ids))
        self.assertEqual(ridge.predict(shuffled), ridge.predict(f))

    def test_neighbor_ties_weights_and_quantiles(self):
        f = Frame(["b", "a", "ignored"], [Feature("x", "x")], [[-1], [1], [0]], [1, 2, 0])
        model = Neighbors(1).fit(f, {"b": 0, "a": 10, "ignored": 1000})
        query = Frame(["q"], f.features, [[0]])
        self.assertEqual(model.kneighbors(query)["q"][0]["id"], "a")
        self.assertEqual(model.predict(query), {"q": 10})
        weighted = Neighbors(2, True).fit(f, {"b": 0, "a": 10, "ignored": 1000})
        self.assertAlmostEqual(weighted.predict(query)["q"], 20 / 3)
        self.assertEqual(weighted.predict_quantiles(query)["q"]["0.5"], 10)
        self.assertEqual(restore_model(weighted.state()).predict(query), weighted.predict(query))

    def test_kmeans_and_silhouette(self):
        f = frame([[0], [1], [9], [10]])
        model = KMeans(2).fit(f)
        labels = {key: value["cluster"] for key, value in model.predict(f).items()}
        self.assertEqual(labels["s0"], labels["s1"])
        self.assertNotEqual(labels["s0"], labels["s2"])
        self.assertAlmostEqual(model.inertia, 1)
        self.assertGreater(metrics.silhouette(f, labels)["mean"], .8)
        self.assertEqual(restore_model(model.state()).predict(f), model.predict(f))

    def test_logistic_against_symmetry_and_replication(self):
        f = frame([[-2], [-1], [1], [2]])
        y = dict(zip(f.ids, [0, 0, 1, 1]))
        model = Logistic().fit(f, y)
        p = model.predict(f)
        self.assertAlmostEqual(p["s0"] + p["s3"], 1)
        self.assertGreater(p["s3"], .5)
        self.assertEqual(model.predict_labels(f), y)
        self.assertEqual(restore_model(model.state()).predict(f), p)
        duplicate = Frame(["a", "b", "c", "d"], f.features, [[-1], [-1], [1], [1]])
        weighted = Frame(["a", "c"], f.features, [[-1], [1]], [2, 2])
        m1 = Logistic().fit(duplicate, dict(zip(duplicate.ids, [0, 0, 1, 1])))
        m2 = Logistic().fit(weighted, {"a": 0, "c": 1})
        self.assertAlmostEqual(m1.coefficients[0], m2.coefficients[0])

    def test_metrics_and_streaming_population(self):
        truth = {"a": 1, "b": 3, "c": 6}
        pred = {"c": 5, "a": 2, "b": 3}
        weights = {"a": 1, "b": 2, "c": 0}
        direct = metrics.regression(truth, pred, weights)
        acc = metrics.RegressionAccumulator().update({"c": 6}, {"c": 5}, {"c": 0})
        acc.update({"a": 1, "b": 3}, {"a": 2, "b": 3}, {"a": 1, "b": 2})
        for key in acc.result():
            self.assertAlmostEqual(acc.result()[key], direct[key])
        self.assertEqual(len(metrics.residual_table(truth, pred)), 3)
        self.assertEqual(metrics.compare(truth, pred, truth)["right"]["mse"], 0)
        self.assertEqual(metrics.grouped_regression(truth, pred, {"a": "g", "b": "g", "c": "q"}, weights)["q"]["status"], "NO_POSITIVE_WEIGHT")
        binary = {"a": 0, "b": 1, "c": 1}
        probs = {"a": .1, "b": .8, "c": .8}
        self.assertEqual(metrics.roc_auc(binary, probs), 1)
        self.assertEqual(metrics.average_precision(binary, probs), 1)
        self.assertEqual(metrics.classification_summary(binary, probs)["accuracy"], 1)
        self.assertEqual(metrics.pinball({"a": 2}, {"a": 1}, .9), .9)
        self.assertEqual(metrics.interval_score({"a": 2}, {"a": 1}, {"a": 3})["coverage"], 1)

    def test_diagnostics_use_actual_data(self):
        f = frame([[1, 2], [2, 3], [3, 7], [4, 5]])
        self.assertEqual(associations(f)[0]["status"], "OK")
        self.assertEqual(associations(f, "spearman")[0]["status"], "OK")
        self.assertEqual(collinearity(f)["status"], "OK")
        self.assertEqual(weighted_ks([1, 2], [3, 4]), 1)
        self.assertEqual(category_shift(["a"], ["b"])["total_variation"], 1)
        self.assertEqual(outliers(frame([[1], [1], [1], [100]]))["x0"]["sample_ids"], ["s3"])
        schema = Schema([Field("x", "X"), Field("g", "G", "category", role="group")])
        table = Table(schema, [Sample("a", (1, "g")), Sample("b", (None, "g"))])
        self.assertEqual(missing_by_group(table, "g", ["x"])[0]["missing_ids"], ["b"])
        self.assertEqual(contingency(table, "g", "g")["cells"][0]["weight"], 2)
        self.assertAlmostEqual(drift(table, table, ["x", "g"])["x"]["mean_shift"], 0)
