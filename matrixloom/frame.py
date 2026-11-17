from dataclasses import dataclass
import math

from .errors import IdentityError, SchemaError
from .schema import finite, fingerprint


@dataclass(frozen=True)
class Feature:
    id: str
    source: str
    operation: str = "numeric"
    unit: str | None = None
    category: str | None = None

    def __post_init__(self):
        if not self.id or not self.source or not self.operation:
            raise SchemaError("feature identity and lineage must be nonempty")

    def state(self):
        return {"id": self.id, "source": self.source, "operation": self.operation,
                "unit": self.unit, "category": self.category}


class Frame:
    """Immutable dense numeric matrix. None is unobserved, never a numeric zero."""

    def __init__(self, ids, features, values, weights=None):
        self.ids = tuple(ids)
        self.features = tuple(features)
        if any(not isinstance(x, str) or not x for x in self.ids):
            raise IdentityError("frame sample identities must be nonempty strings")
        if len(set(self.ids)) != len(self.ids):
            raise IdentityError("duplicate frame sample identity")
        if any(not isinstance(x, Feature) for x in self.features):
            raise SchemaError("frame features must have lineage")
        if len(set(self.feature_ids())) != len(self.features):
            raise IdentityError("duplicate feature identity")
        values = tuple(tuple(row) for row in values)
        if len(values) != len(self.ids):
            raise IdentityError("matrix row count differs from identities")
        checked = []
        for row in values:
            if len(row) != len(self.features):
                raise SchemaError("matrix row width differs from feature axis")
            checked.append(tuple(None if x is None else finite(x, "matrix entry") for x in row))
        self.values = tuple(checked)
        weights = (1.0,) * len(self.ids) if weights is None else tuple(weights)
        if len(weights) != len(self.ids):
            raise IdentityError("weight population differs from matrix")
        self.weights = tuple(finite(x, "weight") for x in weights)
        if any(x < 0 for x in self.weights):
            raise SchemaError("weights cannot be negative")
        self._rows = {key: i for i, key in enumerate(self.ids)}
        self._columns = {f.id: i for i, f in enumerate(self.features)}

    def __len__(self):
        return len(self.ids)

    @property
    def width(self):
        return len(self.features)

    def feature_ids(self):
        return tuple(f.id for f in self.features)

    def shape(self):
        return len(self), self.width

    def column(self, feature_id):
        try:
            index = self._columns[feature_id]
        except KeyError as error:
            raise IdentityError(f"unknown feature {feature_id}") from error
        return tuple(row[index] for row in self.values)

    def row(self, sample_id):
        try:
            return self.values[self._rows[sample_id]]
        except KeyError as error:
            raise IdentityError(f"unknown sample {sample_id}") from error

    def take(self, ids):
        ids = tuple(ids)
        try:
            indexes = [self._rows[key] for key in ids]
        except KeyError as error:
            raise IdentityError("row selection contains unknown identity") from error
        return Frame(ids, self.features, [self.values[i] for i in indexes], [self.weights[i] for i in indexes])

    def select(self, ids):
        ids = tuple(ids)
        try:
            indexes = [self._columns[key] for key in ids]
        except KeyError as error:
            raise IdentityError("column selection contains unknown identity") from error
        return Frame(self.ids, [self.features[i] for i in indexes],
                     [[row[i] for i in indexes] for row in self.values], self.weights)

    def align_features(self, features):
        features = tuple(features)
        ids = tuple(f.id for f in features)
        if set(ids) != set(self.feature_ids()):
            raise SchemaError("prediction feature identities differ from fitted axis")
        aligned = self.select(ids)
        if aligned.features != features:
            raise SchemaError("feature identities match but lineage or units differ")
        return aligned

    def complete(self):
        if any(x is None for row in self.values for x in row):
            raise SchemaError("estimator requires complete features; fit an imputer first")
        return self

    def positive_population(self):
        return self.take(key for key, weight in zip(self.ids, self.weights) if weight > 0)

    def total_weight(self):
        return math.fsum(self.weights)

    def with_values(self, values, features=None):
        return Frame(self.ids, self.features if features is None else features, values, self.weights)

    def with_weights(self, weights_by_id):
        if set(weights_by_id) != set(self.ids):
            raise IdentityError("replacement weights must identify exactly the same samples")
        return Frame(self.ids, self.features, self.values, [weights_by_id[key] for key in self.ids])

    def hstack(self, other):
        if set(self.ids) != set(other.ids):
            raise IdentityError("horizontal concatenation populations differ")
        aligned = other.take(self.ids)
        if self.weights != aligned.weights:
            raise IdentityError("horizontal concatenation weights differ")
        return Frame(self.ids, self.features + aligned.features,
                     [a + b for a, b in zip(self.values, aligned.values)], self.weights)

    def vstack(self, other):
        aligned = other.align_features(self.features)
        return Frame(self.ids + aligned.ids, self.features, self.values + aligned.values,
                     self.weights + aligned.weights)

    def batches(self, size):
        if type(size) is not int or size <= 0:
            raise SchemaError("batch size must be a positive integer")
        for start in range(0, len(self), size):
            yield self.take(self.ids[start:start + size])

    def missingness(self):
        return {f.id: {"observed": sum(row[j] is not None for row in self.values),
                       "missing": sum(row[j] is None for row in self.values),
                       "observed_weight": math.fsum(w for row, w in zip(self.values, self.weights)
                                                    if row[j] is not None)}
                for j, f in enumerate(self.features)}

    def state(self):
        return {"ids": list(self.ids), "features": [f.state() for f in self.features],
                "values": [list(row) for row in self.values], "weights": list(self.weights)}

    def digest(self):
        return fingerprint(self.state())

    @classmethod
    def restore(cls, state):
        return cls(state["ids"], [Feature(**f) for f in state["features"]],
                   state["values"], state["weights"])


def align_targets(frame, targets):
    if not isinstance(targets, dict) or set(targets) != set(frame.ids):
        raise IdentityError("targets must identify exactly the input samples")
    return tuple(finite(targets[key], "target") for key in frame.ids)


def numerical_frame(table, fields=None):
    fields = table.schema.ids("feature") if fields is None else tuple(fields)
    selected = [table.schema.field(key) for key in fields]
    if any(f.kind not in {"number", "integer", "boolean"} for f in selected):
        raise SchemaError("numeric extraction cannot encode categorical or timestamp fields")
    indexes = [table.schema.index(f.id) for f in selected]
    return Frame(table.ids(), [Feature(f.id, f.id, "numeric", f.unit) for f in selected],
                 [[None if sample.values[j] is None else float(sample.values[j]) for j in indexes]
                  for sample in table.samples], table.weights())
