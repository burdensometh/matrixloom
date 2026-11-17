from .errors import NotFitted, SchemaError, StateError
from .frame import Feature, Frame
from .schema import Schema, canonical


class Encoder:
    """Fixed-schema dense encoding; no target-dependent features or hidden refits."""

    def __init__(self, fields=None, unknown="error", max_features=4096):
        if unknown not in {"error", "zero"}:
            raise SchemaError("unknown category policy must be error or zero")
        if type(max_features) is not int or max_features <= 0:
            raise SchemaError("max_features must be positive")
        self.fields = None if fields is None else tuple(fields)
        self.unknown = unknown
        self.max_features = max_features
        self.input_schema = None
        self.vocabularies = {}
        self.features = ()

    def fit(self, table):
        keys = table.schema.ids("feature") if self.fields is None else self.fields
        if not keys or len(set(keys)) != len(keys):
            raise SchemaError("encoder requires unique feature fields")
        schema = table.schema.select(keys)
        vocabularies = {}
        features = []
        for field in schema.fields:
            if field.role == "target":
                raise SchemaError("encoder cannot use a target field")
            if field.kind == "timestamp":
                raise SchemaError("derive explicit numeric time features before encoding")
            if field.kind == "category":
                values = table.column(field.id)
                vocabulary = tuple(sorted({value for value, w in zip(values, table.weights())
                                           if value is not None and w > 0}))
                if not vocabulary:
                    raise SchemaError(f"cannot learn an empty category vocabulary for {field.id}")
                vocabularies[field.id] = vocabulary
                for value in vocabulary:
                    key = "category:" + canonical([field.id, value])
                    features.append(Feature(key, field.id, "onehot", None, value))
            else:
                features.append(Feature(field.id, field.id, "numeric", field.unit))
        if len(features) > self.max_features:
            raise SchemaError("dense encoding exceeds configured feature width")
        self.input_schema = schema
        self.vocabularies = vocabularies
        self.features = tuple(features)
        return self

    def transform(self, table):
        if self.input_schema is None:
            raise NotFitted("encoder has not been fitted")
        current = table.schema.select(self.input_schema.ids())
        self.input_schema.compatible(current)
        positions = [table.schema.index(field.id) for field in self.input_schema.fields]
        values = []
        for sample in table.samples:
            row = []
            for field, position in zip(self.input_schema.fields, positions):
                value = sample.values[position]
                if field.kind != "category":
                    row.append(None if value is None else float(value))
                    continue
                vocabulary = self.vocabularies[field.id]
                if value is None:
                    row.extend([None] * len(vocabulary))
                elif value not in vocabulary:
                    if self.unknown == "error":
                        raise SchemaError(f"unknown category for {field.id} in sample {sample.id}")
                    row.extend([0.0] * len(vocabulary))
                else:
                    row.extend(float(value == category) for category in vocabulary)
            values.append(row)
        return Frame(table.ids(), self.features, values, table.weights())

    def fit_transform(self, table):
        return self.fit(table).transform(table)

    def state(self):
        if self.input_schema is None:
            raise NotFitted("cannot serialize an unfitted encoder")
        return {"fields": list(self.input_schema.ids()), "unknown": self.unknown,
                "max_features": self.max_features, "schema": self.input_schema.state(),
                "vocabularies": {key: list(values) for key, values in self.vocabularies.items()},
                "features": [f.state() for f in self.features]}

    @classmethod
    def restore(cls, state):
        result = cls(state["fields"], state["unknown"], state["max_features"])
        result.input_schema = Schema.restore(state["schema"])
        result.vocabularies = {key: tuple(values) for key, values in state["vocabularies"].items()}
        result.features = tuple(Feature(**f) for f in state["features"])
        expected = []
        for field in result.input_schema.fields:
            if field.kind == "category":
                vocabulary = result.vocabularies.get(field.id, ())
                if not vocabulary or len(vocabulary) != len(set(vocabulary)):
                    raise StateError("invalid stored category vocabulary")
                expected.extend(Feature("category:" + canonical([field.id, value]), field.id, "onehot", None, value)
                                for value in vocabulary)
            else:
                expected.append(Feature(field.id, field.id, "numeric", field.unit))
        if tuple(expected) != result.features:
            raise StateError("stored encoder axis contradicts vocabulary")
        if len(expected) > result.max_features:
            raise StateError("stored encoder exceeds its width budget")
        return result
