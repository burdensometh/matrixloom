"""Feature graphs with explicit node dependencies and shared fitted state.

A graph is a directed acyclic collection of nodes.  Every node declares which
nodes it consumes; a node that feeds several branches is fitted exactly once
and all consumers observe the same frozen state.  Tables enter through
``source`` nodes (or through a ``sequence`` node that wraps an existing
sequential :class:`~matrixloom.pipeline.Features` plan), ``transform`` nodes
apply one of the registered numerical stages, and ``merge`` nodes join
branches by sample identity.  Merging never concatenates by row position and
never silently replaces a column whose identity already exists.
"""

import copy
import itertools

from .encoding import Encoder
from .errors import IdentityError, NotFitted, SchemaError, StateError
from .expressions import Derivations
from .frame import Feature, Frame
from .schema import canonical, fingerprint
from .transforms import create_transform, restore_transform


_KIND_ALIASES = {
    "source": "source", "input": "source", "encode": "source",
    "transform": "transform", "stage": "transform", "apply": "transform",
    "merge": "merge", "join": "merge", "concat": "merge",
    "sequence": "sequence", "pipeline": "sequence", "features": "sequence",
}
_INPUT_KEYS = ("inputs", "dependencies", "depends_on")
_TRANSFORM_KEYS = ("transform", "stage")
_SEQUENCE_KEYS = ("feature_spec", "spec", "features")
_COMMON_KEYS = {"id", "kind", *_INPUT_KEYS}
_KIND_KEYS = {
    "source": _COMMON_KEYS | {"fields", "unknown", "max_features"},
    "transform": _COMMON_KEYS | set(_TRANSFORM_KEYS),
    "sequence": _COMMON_KEYS | set(_SEQUENCE_KEYS),
    "merge": _COMMON_KEYS,
}


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} must be nonempty text")
    return value


def _alias_value(raw, keys, label):
    present = [raw[key] for key in keys if key in raw]
    if not present:
        return None
    first = present[0]
    for value in present[1:]:
        if value != first:
            raise SchemaError(f"conflicting {label} declarations on one node")
    return first


def _dependency_list(raw):
    value = _alias_value(raw, _INPUT_KEYS, "dependency")
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise SchemaError("node dependencies must be a list of node identifiers")
    result = [_identifier(item, "dependency") for item in value]
    if len(set(result)) != len(result):
        raise SchemaError("node dependencies must be unique")
    return result


def _validate_sequential(specification):
    from .pipeline import Features
    Features(**copy.deepcopy(specification))


def _combine_axes(axes, label):
    features = []
    seen = set()
    for axis in axes:
        for feature in axis:
            if feature.id in seen:
                raise SchemaError(f"{label} would silently replace feature identity {feature.id}")
            seen.add(feature.id)
            features.append(feature)
    return tuple(features)


def merge_frames(frames, label="merge"):
    """Join frames by sample identity, rejecting any duplicated feature axis.

    Rows are aligned through ``sample id`` rather than position.  Populations
    and weights must agree, and two inputs may never contribute the same
    feature identity -- even when the lineage text happens to match, because
    identical lineage does not imply identical values.
    """

    frames = tuple(frames)
    if not frames:
        raise SchemaError(f"{label} requires at least one input frame")
    base = frames[0]
    ids = base.ids
    expected = set(ids)
    weights = base.weights
    columns = [[] for _ in ids]
    features = []
    seen = set()
    for frame in frames:
        if set(frame.ids) != expected:
            raise IdentityError(f"{label} inputs have different sample populations")
        aligned = frame.take(ids)
        if aligned.weights != weights:
            raise IdentityError(f"{label} inputs have different sample weights")
        for feature in aligned.features:
            if feature.id in seen:
                raise SchemaError(f"{label} would silently replace feature identity {feature.id}")
            seen.add(feature.id)
            features.append(feature)
        for target, row in zip(columns, aligned.values):
            target.extend(row)
    return Frame(ids, features, columns, weights)


class FeatureGraph:
    """A fitted-once directed feature graph over identity-keyed tables."""

    def __init__(self, nodes, outputs=None, derivations=()):
        if not isinstance(nodes, (list, tuple)) or not nodes:
            raise SchemaError("a feature graph requires at least one node")
        normalized = []
        identifiers = set()
        for raw in nodes:
            node = self._normalize(raw)
            if node["id"] in identifiers:
                raise SchemaError(f"duplicate feature graph node identity {node['id']}")
            identifiers.add(node["id"])
            normalized.append(node)
        self._nodes = {node["id"]: node for node in normalized}
        self._order = self._topological(normalized)
        self.spec = {"derivations": copy.deepcopy(list(derivations)),
                     "nodes": normalized, "outputs": self._normalize_outputs(outputs)}
        self.derivations = Derivations(self.spec["derivations"])
        self.states = None
        self.node_features = None
        self.output_features = None
        self.training_digest = None

    @classmethod
    def _normalize(cls, raw):
        if not isinstance(raw, dict):
            raise SchemaError("feature graph nodes must be mappings")
        node_id = _identifier(raw.get("id"), "node identity")
        if "kind" not in raw:
            raise SchemaError(f"node {node_id} must declare its kind")
        kind = _KIND_ALIASES.get(raw["kind"])
        if kind is None:
            raise SchemaError(f"unsupported feature graph node kind for {node_id}")
        inputs = _dependency_list(raw)
        if kind in {"source", "sequence"} and inputs:
            raise SchemaError(f"{kind} node {node_id} cannot declare upstream dependencies")
        if kind == "transform" and len(inputs) != 1:
            raise SchemaError(f"transform node {node_id} requires exactly one input")
        if kind == "merge" and not inputs:
            raise SchemaError(f"merge node {node_id} requires at least one input")
        unknown_keys = set(raw) - _KIND_KEYS[kind]
        if unknown_keys:
            raise SchemaError(f"unsupported node options for {node_id}: {sorted(unknown_keys)}")
        if kind == "source":
            fields = raw.get("fields")
            if fields is not None:
                if isinstance(fields, str) or not isinstance(fields, (list, tuple)) or not fields:
                    raise SchemaError(f"source node {node_id} fields must be a nonempty list")
                fields = [_identifier(field, "source field") for field in fields]
                if len(set(fields)) != len(fields):
                    raise SchemaError(f"source node {node_id} repeats a field identity")
            Encoder(fields, raw.get("unknown", "error"), raw.get("max_features", 4096))
            return {"id": node_id, "kind": kind, "fields": fields,
                    "unknown": raw.get("unknown", "error"), "max_features": raw.get("max_features", 4096)}
        if kind == "transform":
            specification = _alias_value(raw, _TRANSFORM_KEYS, "transform")
            if not isinstance(specification, dict):
                raise SchemaError(f"transform node {node_id} requires a transform specification")
            create_transform(specification)
            return {"id": node_id, "kind": kind, "inputs": inputs,
                    "transform": copy.deepcopy(specification)}
        if kind == "sequence":
            specification = _alias_value(raw, _SEQUENCE_KEYS, "feature specification")
            if not isinstance(specification, dict):
                raise SchemaError(f"sequence node {node_id} requires a feature specification")
            if specification.get("kind") == "graph" or "nodes" in specification:
                raise SchemaError("sequence nodes embed sequential feature plans, not graphs")
            _validate_sequential(specification)
            return {"id": node_id, "kind": kind, "feature_spec": copy.deepcopy(specification)}
        return {"id": node_id, "kind": kind, "inputs": inputs}

    def _normalize_outputs(self, outputs):
        if isinstance(outputs, str):
            outputs = [outputs]
        elif outputs is None:
            consumed = {key for node in self._nodes.values() for key in node.get("inputs", ())}
            sinks = [node["id"] for node in self._nodes.values() if node["id"] not in consumed]
            if len(sinks) != 1:
                raise SchemaError("graphs with several sinks must declare their outputs explicitly")
            outputs = sinks
        else:
            outputs = list(outputs)
        if not outputs:
            raise SchemaError("a feature graph requires at least one output node")
        checked = [_identifier(item, "output node") for item in outputs]
        if len(set(checked)) != len(checked):
            raise SchemaError("duplicate feature graph output node")
        for key in checked:
            if key not in self._nodes:
                raise SchemaError(f"output references unknown node {key}")
        return checked

    def _dependencies(self, node_id):
        return tuple(self._nodes[node_id].get("inputs", ()))

    def _topological(self, nodes):
        identifiers = {node["id"] for node in nodes}
        for node in nodes:
            for dependency in node.get("inputs", ()):
                if dependency not in identifiers:
                    raise SchemaError(f"node {node['id']} depends on absent node {dependency}")
        resolved = set()
        order = []
        pending = [node["id"] for node in nodes]
        while pending:
            ready = [key for key in pending
                     if all(dependency in resolved for dependency in self._dependencies(key))]
            if not ready:
                raise SchemaError("feature graph contains a cycle: " + ", ".join(pending))
            order.extend(ready)
            resolved.update(ready)
            pending = [key for key in pending if key not in resolved]
        return tuple(order)

    def _check_sources(self, table):
        for node_id in self._order:
            node = self._nodes[node_id]
            if node["kind"] != "source":
                continue
            keys = table.schema.ids("feature") if node["fields"] is None else node["fields"]
            if not keys:
                raise SchemaError(f"source node {node_id} has no input fields")
            table.schema.select(keys)
            for key in keys:
                field = table.schema.field(key)
                if field.role == "target":
                    raise SchemaError(f"source node {node_id} cannot consume target field {key}")
                if field.kind == "timestamp":
                    raise SchemaError(f"source node {node_id} needs explicit numeric time features for {key}")
            if len(keys) > node["max_features"]:
                raise SchemaError(f"source node {node_id} exceeds its feature width budget")

    def _prepare(self, table, preflight):
        prepared = self.derivations.transform(table)
        self._check_sources(prepared)
        if preflight:
            self._preflight(prepared)
        return prepared

    def _static_axis(self, node_id, table):
        """Exact output axis computed without fitting any node.

        Numeric sources, learned category vocabularies, polynomial expansions,
        selectors, scalers and merges are all deterministic given the training
        table, so lineage conflicts are rejected before any node is executed.
        Returns ``None`` when the axis genuinely cannot be known up front.
        """

        node = self._nodes[node_id]
        kind = node["kind"]
        if kind == "source":
            keys = table.schema.ids("feature") if node["fields"] is None else node["fields"]
            features = []
            for key in keys:
                field = table.schema.field(key)
                if field.kind == "category":
                    vocabulary = tuple(sorted({value for value, weight
                                               in zip(table.column(key), table.weights())
                                               if value is not None and weight > 0}))
                    if not vocabulary:
                        raise SchemaError(f"source node {node_id} cannot learn an empty vocabulary for {key}")
                    features.extend(Feature("category:" + canonical([key, value]), key, "onehot", None, value)
                                    for value in vocabulary)
                else:
                    features.append(Feature(field.id, field.id, "numeric", field.unit))
            return tuple(features)
        if kind == "merge":
            return self._union_axes(self._static_axis(key, table) for key in node["inputs"])
        if kind == "transform":
            upstream = self._static_axis(node["inputs"][0], table)
            if upstream is None:
                return None
            specification = node["transform"]
            name = specification["kind"]
            if name == "select":
                positions = {feature.id: feature for feature in upstream}
                absent = set(specification["ids"]) - positions.keys()
                if absent:
                    raise SchemaError(f"select node {node_id} refers to absent features: {sorted(absent)}")
                return tuple(positions[key] for key in specification["ids"])
            if name in {"impute", "standardize", "range", "winsorize"}:
                if name == "impute" and specification.get("indicators", False):
                    extra = tuple(Feature("missing:" + canonical(feature.id), feature.id, "missing_indicator")
                                  for feature in upstream)
                    return upstream + extra
                return upstream
            if name == "polynomial":
                return self._polynomial_axis(node_id, upstream, specification)
            return None
        if kind == "sequence":
            return self._sequence_axis(node["feature_spec"], table)
        return None

    def _polynomial_axis(self, node_id, upstream, specification):
        degree = specification.get("degree", 2)
        interactions_only = specification.get("interactions_only", False)
        budget = specification.get("max_features", 4096)
        iterator = itertools.combinations if interactions_only else itertools.combinations_with_replacement
        terms = []
        for power in range(1, degree + 1):
            for term in iterator(range(len(upstream)), power):
                terms.append(term)
                if len(terms) > budget:
                    raise SchemaError(f"polynomial node {node_id} exceeds its width budget")
        features = []
        for term in terms:
            if len(term) == 1:
                features.append(upstream[term[0]])
                continue
            ids = [upstream[index].id for index in term]
            units = [upstream[index].unit or "1" for index in term]
            features.append(Feature("product:" + canonical(ids), canonical(ids), "product", "*".join(units)))
        return tuple(features)

    def _sequence_axis(self, specification, table):
        fields = specification["fields"]
        if fields is None:
            fields = list(table.schema.ids("feature"))
        axis = []
        for key in fields:
            field = table.schema.field(key)
            if field.kind == "timestamp":
                return None
            if field.kind == "category":
                vocabulary = tuple(sorted({value for value, weight
                                           in zip(table.column(key), table.weights())
                                           if value is not None and weight > 0}))
                if not vocabulary:
                    return None
                axis.extend(Feature("category:" + canonical([key, value]), key, "onehot", None, value)
                            for value in vocabulary)
            else:
                axis.append(Feature(field.id, field.id, "numeric", field.unit))
        for transform in specification["transforms"]:
            name = transform["kind"]
            if name == "select":
                positions = {feature.id: feature for feature in axis}
                if not set(transform["ids"]) <= positions.keys():
                    return None
                axis = [positions[key] for key in transform["ids"]]
            elif name in {"impute", "standardize", "range", "winsorize"}:
                if name == "impute" and transform.get("indicators", False):
                    axis = axis + [Feature("missing:" + canonical(feature.id), feature.id, "missing_indicator")
                                   for feature in axis]
            elif name == "polynomial":
                axis = list(self._polynomial_axis("sequence", axis, transform))
            else:
                return None
        return tuple(axis)

    def _union_axes(self, axes):
        axes = list(axes)
        if any(axis is None for axis in axes):
            return None
        result = []
        seen = set()
        for axis in axes:
            for feature in axis:
                if feature.id in seen:
                    raise SchemaError(f"merge would silently replace feature identity {feature.id}")
                seen.add(feature.id)
                result.append(feature)
        return tuple(result)

    def _preflight(self, prepared):
        for node_id in self._order:
            if self._nodes[node_id]["kind"] == "merge":
                self._static_axis(node_id, prepared)
        if len(self.spec["outputs"]) > 1:
            self._union_axes(self._static_axis(node_id, prepared) for node_id in self.spec["outputs"])

    def fit(self, table):
        prepared = self._prepare(table, preflight=True)
        states, node_features, output = self._execute(prepared)
        self.states = states
        self.node_features = node_features
        self.output_features = output.features
        self.training_digest = table.digest()
        return self

    def fit_transform(self, table):
        return self.fit(table).transform(table)

    def _execute(self, table):
        states = {}
        frames = {}
        node_features = {}
        for node_id in self._order:
            node = self._nodes[node_id]
            kind = node["kind"]
            if kind == "source":
                encoder = Encoder(node["fields"], node["unknown"], node["max_features"]).fit(table)
                frame = encoder.transform(table)
                states[node_id] = {"kind": kind, "encoder": encoder}
            elif kind == "sequence":
                from .pipeline import Features
                features = Features(**copy.deepcopy(node["feature_spec"]))
                frame = features.fit_transform(table)
                states[node_id] = {"kind": kind, "features": features}
            elif kind == "transform":
                upstream = frames[node["inputs"][0]]
                stage = create_transform(node["transform"]).fit(upstream)
                frame = stage.transform(upstream)
                states[node_id] = {"kind": kind, "stage": stage}
            else:
                frame = merge_frames([frames[key] for key in node["inputs"]], f"merge node {node_id}")
                states[node_id] = {"kind": kind}
            frames[node_id] = frame
            node_features[node_id] = frame.features
        outputs = [frames[key] for key in self.spec["outputs"]]
        final = outputs[0] if len(outputs) == 1 else merge_frames(outputs, "graph outputs")
        return states, node_features, final

    def transform(self, table):
        if self.states is None:
            raise NotFitted("feature graph has not been fitted")
        prepared = self._prepare(table, preflight=False)
        frames = {}
        for node_id in self._order:
            node = self._nodes[node_id]
            kind = node["kind"]
            if kind == "source":
                frame = self.states[node_id]["encoder"].transform(prepared)
            elif kind == "sequence":
                frame = self.states[node_id]["features"].transform(prepared)
            elif kind == "transform":
                frame = self.states[node_id]["stage"].transform(frames[node["inputs"][0]])
            else:
                frame = merge_frames([frames[key] for key in node["inputs"]], f"merge node {node_id}")
            if frame.features != self.node_features[node_id]:
                raise StateError(f"fitted feature graph axis changed at node {node_id}")
            frames[node_id] = frame
        outputs = [frames[key] for key in self.spec["outputs"]]
        final = outputs[0] if len(outputs) == 1 else merge_frames(outputs, "graph outputs")
        if final.features != self.output_features:
            raise StateError("fitted feature graph output axis changed unexpectedly")
        return final

    def fresh(self):
        return FeatureGraph(**copy.deepcopy(self.spec))

    def describe(self):
        if self.states is None:
            raise NotFitted("feature graph has not been fitted")
        return {"training_digest": self.training_digest,
                "outputs": list(self.spec["outputs"]),
                "features": [feature.state() for feature in self.output_features],
                "nodes": [{"id": node_id, "kind": self._nodes[node_id]["kind"],
                           "inputs": list(self._dependencies(node_id)),
                           "features": [feature.state() for feature in self.node_features[node_id]]}
                          for node_id in self._order],
                "derivations": copy.deepcopy(self.spec["derivations"])}

    def state(self):
        if self.states is None:
            raise NotFitted("cannot serialize an unfitted feature graph")
        encoded = {}
        for node_id in self._order:
            stored = self.states[node_id]
            if stored["kind"] == "source":
                encoded[node_id] = {"kind": "source", "encoder": stored["encoder"].state()}
            elif stored["kind"] == "sequence":
                encoded[node_id] = {"kind": "sequence", "features": stored["features"].state()}
            elif stored["kind"] == "transform":
                encoded[node_id] = {"kind": "transform", "stage": stored["stage"].state()}
            else:
                encoded[node_id] = {"kind": "merge"}
        return {"version": 1, "kind": "graph", "spec": copy.deepcopy(self.spec),
                "nodes": encoded,
                "node_features": {node_id: [feature.state() for feature in self.node_features[node_id]]
                                  for node_id in self._order},
                "training_digest": self.training_digest,
                "output_features": [feature.state() for feature in self.output_features]}

    def digest(self):
        return fingerprint(self.state())

    @classmethod
    def restore(cls, state):
        if not isinstance(state, dict) or state.get("version") != 1 or state.get("kind") != "graph":
            raise StateError("unsupported feature graph state")
        specification = state.get("spec")
        if not isinstance(specification, dict):
            raise StateError("feature graph state lacks a specification")
        result = cls(**copy.deepcopy(specification))
        if result.spec != specification:
            raise StateError("stored feature graph specification is not canonical")
        stored = state.get("nodes")
        if not isinstance(stored, dict) or set(stored) != set(result._nodes):
            raise StateError("stored feature graph nodes do not match its specification")
        expected_axes = state.get("node_features")
        if not isinstance(expected_axes, dict) or set(expected_axes) != set(result._nodes):
            raise StateError("stored feature graph lacks node axes")
        axes = {}
        states = {}
        for node_id in result._order:
            node = result._nodes[node_id]
            node_state = stored[node_id]
            if node_state.get("kind") != node["kind"]:
                raise StateError(f"stored node {node_id} kind contradicts the specification")
            kind = node["kind"]
            if kind == "source":
                encoder = Encoder.restore(node_state["encoder"])
                if encoder.unknown != node["unknown"] or encoder.max_features != node["max_features"]:
                    raise StateError(f"stored source node {node_id} contradicts its options")
                if node["fields"] is not None and list(encoder.input_schema.ids()) != node["fields"]:
                    raise StateError(f"stored source node {node_id} field selection differs")
                axis = encoder.features
                states[node_id] = {"kind": kind, "encoder": encoder}
            elif kind == "sequence":
                from .pipeline import Features
                features = Features.restore(node_state["features"])
                if features.spec != Features(**copy.deepcopy(node["feature_spec"])).spec:
                    raise StateError(f"stored sequence node {node_id} differs from its specification")
                axis = features.output_features
                states[node_id] = {"kind": kind, "features": features}
            elif kind == "transform":
                stage = restore_transform(node_state["stage"])
                upstream = axes[node["inputs"][0]]
                if stage.features != upstream:
                    raise StateError(f"stored transform node {node_id} does not connect to its input")
                axis = stage.transform(Frame([], stage.features, [])).features
                states[node_id] = {"kind": kind, "stage": stage}
            else:
                axis = _combine_axes([axes[key] for key in node["inputs"]], f"merge node {node_id}")
                states[node_id] = {"kind": kind}
            if tuple(Feature(**item) for item in expected_axes[node_id]) != axis:
                raise StateError(f"stored axis for node {node_id} contradicts its state")
            axes[node_id] = axis
        outputs = [axes[key] for key in result.spec["outputs"]]
        final = outputs[0] if len(outputs) == 1 else _combine_axes(outputs, "graph outputs")
        expected = tuple(Feature(**item) for item in state["output_features"])
        if final != expected:
            raise StateError("stored feature graph output axis does not connect")
        result.states = states
        result.node_features = {node_id: axes[node_id] for node_id in result._order}
        result.output_features = expected
        result.training_digest = state.get("training_digest")
        if not isinstance(result.training_digest, str) or not result.training_digest:
            raise StateError("stored feature graph lacks training provenance")
        return result
