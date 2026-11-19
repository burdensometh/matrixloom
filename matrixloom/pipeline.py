import copy

from .encoding import Encoder
from .errors import NotFitted, SchemaError, StateError
from .estimators import KMeans, create_model, restore_model
from .expressions import Derivations
from .frame import Feature
from .graph import FeatureGraph
from .schema import fingerprint
from .transforms import create_transform, restore_transform


class Features:
    def __init__(self, fields=None, transforms=(), derivations=(), unknown="error", max_features=4096):
        self.spec = {"fields": None if fields is None else list(fields),
                     "transforms": copy.deepcopy(list(transforms)),
                     "derivations": copy.deepcopy(list(derivations)),
                     "unknown": unknown, "max_features": max_features}
        self.encoder = None
        self.stages = None
        self.derivations = None
        self.training_digest = None
        self.output_features = None

    def fit(self, table):
        derived = Derivations(self.spec["derivations"])
        prepared = derived.transform(table)
        encoder = Encoder(self.spec["fields"], self.spec["unknown"], self.spec["max_features"])
        frame = encoder.fit_transform(prepared)
        stages = []
        for specification in self.spec["transforms"]:
            stage = create_transform(specification)
            frame = stage.fit_transform(frame)
            stages.append(stage)
        self.encoder = encoder
        self.stages = tuple(stages)
        self.derivations = derived
        self.training_digest = table.digest()
        self.output_features = frame.features
        return self

    def transform(self, table):
        if self.encoder is None:
            raise NotFitted("feature pipeline has not been fitted")
        frame = self.encoder.transform(self.derivations.transform(table))
        for stage in self.stages:
            frame = stage.transform(frame)
        if frame.features != self.output_features:
            raise StateError("fitted feature axis changed unexpectedly")
        return frame

    def fit_transform(self, table):
        return self.fit(table).transform(table)

    def fresh(self):
        return Features(**copy.deepcopy(self.spec))

    def describe(self):
        if self.encoder is None:
            raise NotFitted("feature pipeline has not been fitted")
        return {"training_digest": self.training_digest,
                "inputs": list(self.encoder.input_schema.ids()),
                "outputs": [f.state() for f in self.output_features],
                "stages": [stage.kind for stage in self.stages],
                "derivations": self.derivations.state()}

    def state(self):
        if self.encoder is None:
            raise NotFitted("cannot serialize unfitted features")
        return {"version": 1, "spec": copy.deepcopy(self.spec), "encoder": self.encoder.state(),
                "stages": [stage.state() for stage in self.stages],
                "training_digest": self.training_digest,
                "output_features": [f.state() for f in self.output_features]}

    @classmethod
    def restore(cls, state):
        if state.get("version") != 1:
            raise StateError("unsupported feature pipeline version")
        result = cls(**state["spec"])
        result.encoder = Encoder.restore(state["encoder"])
        result.stages = tuple(restore_transform(s) for s in state["stages"])
        result.derivations = Derivations(result.spec["derivations"])
        result.training_digest = state["training_digest"]
        result.output_features = tuple(Feature(**f) for f in state["output_features"])
        axis = result.encoder.features
        for stage in result.stages:
            if axis != stage.features:
                raise StateError("serialized feature stages do not connect")
            from .frame import Frame
            axis = stage.transform(Frame([], axis, [])).features
        if axis != result.output_features:
            raise StateError("serialized output axis does not match stages")
        return result


def is_graph_spec(spec):
    return isinstance(spec, dict) and (spec.get("kind") == "graph" or "nodes" in spec)


def create_features(spec):
    """Build an unfitted feature plan from either a sequential or graph spec."""

    spec = copy.deepcopy(spec)
    if is_graph_spec(spec):
        if spec.get("kind", "graph") != "graph":
            raise SchemaError("unsupported feature graph kind")
        options = {key: value for key, value in spec.items() if key != "kind"}
        try:
            return FeatureGraph(**options)
        except TypeError as error:
            raise SchemaError("unknown feature graph option") from error
    try:
        return Features(**spec)
    except TypeError as error:
        raise SchemaError("unknown feature pipeline option") from error


def restore_features(state):
    if isinstance(state, dict) and state.get("kind") == "graph":
        return FeatureGraph.restore(state)
    return Features.restore(state)


class Predictor:
    def __init__(self, feature_spec, model_spec, target):
        self.feature_spec = copy.deepcopy(feature_spec)
        self.model_spec = copy.deepcopy(model_spec)
        self.target = target
        self.features = None
        self.model = None
        self.training_ids = None
        self.training_digest = None

    def fit(self, table):
        field = table.schema.field(self.target)
        if field.role != "target" or field.kind not in {"number", "integer"}:
            raise SchemaError("regression target must be a numeric target field")
        targets = dict(zip(table.ids(), table.column(self.target)))
        if any(value is None for value in targets.values()):
            raise SchemaError("training labels cannot be missing")
        features = create_features(self.feature_spec)
        frame = features.fit_transform(table)
        model = create_model(self.model_spec)
        if isinstance(model, KMeans):
            raise SchemaError("use cluster fitting for an unsupervised model")
        model.fit(frame, targets)
        self.features = features
        self.model = model
        self.training_ids = table.ids()
        self.training_digest = table.digest()
        return self

    def predict(self, table):
        if self.model is None:
            raise NotFitted("predictor has not been fitted")
        return self.model.predict(self.features.transform(table))

    def encoded(self, table):
        if self.features is None:
            raise NotFitted("predictor has not been fitted")
        return self.features.transform(table)

    def state(self):
        if self.model is None:
            raise NotFitted("cannot serialize unfitted predictor")
        return {"version": 1, "kind": "predictor", "feature_spec": copy.deepcopy(self.feature_spec),
                "model_spec": copy.deepcopy(self.model_spec), "target": self.target,
                "features": self.features.state(), "model": self.model.state(),
                "training_ids": list(self.training_ids), "training_digest": self.training_digest}

    def digest(self):
        return fingerprint(self.state())

    @classmethod
    def restore(cls, state):
        if state.get("version") != 1 or state.get("kind") != "predictor":
            raise StateError("unsupported predictor bundle")
        result = cls(state["feature_spec"], state["model_spec"], state["target"])
        result.features = restore_features(state["features"])
        result.model = restore_model(state["model"])
        result.training_ids = tuple(state["training_ids"])
        result.training_digest = state["training_digest"]
        if result.features.output_features != result.model.features:
            raise StateError("predictor preprocessing and model axes differ")
        if result.features.training_digest != result.training_digest:
            raise StateError("predictor training provenance differs")
        return result


class Clusterer:
    def __init__(self, feature_spec, model_spec):
        self.feature_spec = copy.deepcopy(feature_spec)
        self.model_spec = copy.deepcopy(model_spec)
        self.features = None
        self.model = None
        self.training_digest = None

    def fit(self, table):
        features = create_features(self.feature_spec)
        frame = features.fit_transform(table)
        model = create_model(self.model_spec)
        if not isinstance(model, KMeans):
            raise SchemaError("cluster fitting requires kmeans")
        model.fit(frame)
        self.features = features
        self.model = model
        self.training_digest = table.digest()
        return self

    def predict(self, table):
        if self.model is None:
            raise NotFitted("clusterer has not been fitted")
        return self.model.predict(self.features.transform(table))

    def state(self):
        if self.model is None:
            raise NotFitted("cannot serialize unfitted clusterer")
        return {"version": 1, "kind": "clusterer", "feature_spec": copy.deepcopy(self.feature_spec),
                "model_spec": copy.deepcopy(self.model_spec), "features": self.features.state(),
                "model": self.model.state(), "training_digest": self.training_digest}

    @classmethod
    def restore(cls, state):
        if state.get("version") != 1 or state.get("kind") != "clusterer":
            raise StateError("unsupported clusterer bundle")
        result = cls(state["feature_spec"], state["model_spec"])
        result.features = restore_features(state["features"])
        result.model = restore_model(state["model"])
        result.training_digest = state["training_digest"]
        if not isinstance(result.model, KMeans) or result.features.output_features != result.model.features:
            raise StateError("clusterer model kind or axis mismatch")
        return result


def restore_bundle(state):
    if state.get("kind") == "predictor":
        return Predictor.restore(state)
    if state.get("kind") == "clusterer":
        return Clusterer.restore(state)
    if state.get("kind") == "graph":
        return FeatureGraph.restore(state)
    raise StateError("unknown analysis bundle kind")
