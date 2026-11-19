import itertools
import math

from .errors import NotFitted, NumericalError, SchemaError, StateError
from .frame import Feature, Frame
from .schema import canonical, finite
from . import statistics as stats


class Transform:
    kind = None

    def __init__(self):
        self.features = None

    def _fit_axis(self, frame):
        if not frame.features:
            raise SchemaError("a transform requires at least one feature")
        self.features = frame.features

    def _check(self, frame):
        if self.features is None:
            raise NotFitted("transform has not been fitted")
        return frame.align_features(self.features)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

    def _axis_state(self):
        if self.features is None:
            raise NotFitted("cannot serialize an unfitted transform")
        return {"kind": self.kind, "features": [f.state() for f in self.features]}

    def _restore_axis(self, state):
        self.features = tuple(Feature(**f) for f in state["features"])
        if len({f.id for f in self.features}) != len(self.features):
            raise StateError("stored transform axis is not unique")


class Imputer(Transform):
    kind = "impute"

    def __init__(self, strategy="mean", empty_value=0.0, indicators=False):
        super().__init__()
        if strategy not in {"mean", "median", "constant"}:
            raise SchemaError("unsupported imputation strategy")
        if type(indicators) is not bool:
            raise SchemaError("indicators must be boolean")
        self.strategy = strategy
        self.empty_value = finite(empty_value)
        self.indicators = indicators
        self.fill = None

    def fit(self, frame):
        fill = []
        for feature in frame.features:
            values = frame.column(feature.id)
            if self.strategy == "constant":
                value = self.empty_value
            else:
                try:
                    method = stats.mean if self.strategy == "mean" else stats.median
                    value = method(values, frame.weights)
                except NumericalError:
                    value = self.empty_value
            fill.append(value)
        self._fit_axis(frame)
        self.fill = tuple(fill)
        return self

    def transform(self, frame):
        frame = self._check(frame)
        features = frame.features
        if self.indicators:
            features += tuple(Feature("missing:" + canonical(f.id), f.id, "missing_indicator")
                              for f in frame.features)
        values = []
        for row in frame.values:
            output = [self.fill[j] if value is None else value for j, value in enumerate(row)]
            if self.indicators:
                output.extend(float(value is None) for value in row)
            values.append(output)
        return frame.with_values(values, features)

    def state(self):
        return {**self._axis_state(), "strategy": self.strategy, "empty_value": self.empty_value,
                "indicators": self.indicators, "fill": list(self.fill)}

    @classmethod
    def restore(cls, state):
        result = cls(state["strategy"], state["empty_value"], state["indicators"])
        result._restore_axis(state)
        result.fill = tuple(finite(x) for x in state["fill"])
        if len(result.fill) != len(result.features):
            raise StateError("imputer state width mismatch")
        return result


class NeighborImputer(Transform):
    """Donor-based imputation over the original observation mask.

    Distances are mean standardized squared differences over the dimensions observed
    in both the query row and the donor. Scales are learned from the training frame
    only; prediction never refits them. A donor for a missing column must observe that
    column in the frozen training data, and donors only ever contribute their original
    observations, so values filled by this stage cannot feed back into a distance.
    """

    kind = "neighbor_impute"

    def __init__(self, neighbors=5, min_common=1, distance_weighted=True,
                 fallback="mean", empty_value=0.0, indicators=False):
        super().__init__()
        if type(neighbors) is not int or neighbors < 1:
            raise SchemaError("neighbor count must be a positive integer")
        if type(min_common) is not int or min_common < 1:
            raise SchemaError("minimum common dimension count must be a positive integer")
        if type(distance_weighted) is not bool or type(indicators) is not bool:
            raise SchemaError("neighbor imputation flags must be boolean")
        if fallback not in {"mean", "constant"}:
            raise SchemaError("unsupported neighbor imputation fallback")
        self.neighbors = neighbors
        self.min_common = min_common
        self.distance_weighted = distance_weighted
        self.fallback = fallback
        self.empty_value = finite(empty_value)
        self.indicators = indicators
        self.scales = None
        self.fallback_fill = None
        self.donors = None

    def fit(self, frame):
        if self.min_common > frame.width:
            raise SchemaError("minimum common dimensions cannot exceed the feature width")
        scales = []
        fallback_fill = []
        for feature in frame.features:
            column = frame.column(feature.id)
            try:
                scale = stats.standard_deviation(column, frame.weights)
            except NumericalError:
                scale = 0.0
            scales.append(scale if scale > 0 else 1.0)
            if self.fallback == "constant":
                fallback_fill.append(self.empty_value)
            else:
                try:
                    fallback_fill.append(stats.mean(column, frame.weights))
                except NumericalError:
                    fallback_fill.append(self.empty_value)
        positive = frame.positive_population()
        self._fit_axis(frame)
        self.scales = tuple(scales)
        self.fallback_fill = tuple(fallback_fill)
        self.donors = positive
        return self

    def _distance(self, query, donor, query_mask, donor_mask):
        total = 0.0
        common = 0
        for j in range(len(query)):
            if not query_mask[j] or not donor_mask[j]:
                continue
            scaled = (query[j] - donor[j]) / self.scales[j]
            total += scaled * scaled
            common += 1
        if common < self.min_common:
            return None
        return total / common

    def _fill(self, frame, mask=None):
        """Fill missing entries.

        ``mask`` is the original observation matrix: True where a value was observed.
        When it is supplied, unmasked positions are treated as fill targets even if
        the frame currently holds a previously imputed value, and distances use the
        mask rather than the frame's None pattern. Filled values therefore can never
        be mistaken for real observations, and replaying a fill over its own output
        reproduces the same values.
        """
        if mask is None:
            mask = tuple(tuple(value is not None for value in row) for row in frame.values)
        elif len(mask) != len(frame.values) or any(len(row) != frame.width for row in mask):
            raise SchemaError("observation mask population differs from the frame")
        values = []
        sources = {}
        donor_rows = dict(zip(self.donors.ids, self.donors.values))
        weight_map = dict(zip(self.donors.ids, self.donors.weights))
        donor_masks = {key: tuple(value is not None for value in row)
                       for key, row in donor_rows.items()}
        for sample_id, row, observed_mask in zip(frame.ids, frame.values, mask):
            output = list(row)
            for j, observed in enumerate(row):
                if observed_mask[j]:
                    continue
                candidates = []
                for donor_id, donor_row in donor_rows.items():
                    if donor_row[j] is None:
                        continue
                    distance = self._distance(row, donor_row, observed_mask, donor_masks[donor_id])
                    if distance is not None:
                        candidates.append((distance, donor_id))
                candidates.sort(key=lambda item: (item[0], item[1]))
                chosen = candidates[:self.neighbors]
                if not chosen:
                    output[j] = self.fallback_fill[j]
                    sources[(sample_id, frame.features[j].id)] = {
                        "strategy": "fallback", "fallback": self.fallback,
                        "value": self.fallback_fill[j], "donors": []}
                    continue
                if self.distance_weighted and any(distance == 0.0 for distance, _ in chosen):
                    chosen = [(distance, donor_id) for distance, donor_id in chosen if distance == 0.0]
                contributions = []
                weights = []
                for distance, donor_id in chosen:
                    if self.distance_weighted and distance > 0.0:
                        weight = weight_map[donor_id] / distance
                    else:
                        weight = weight_map[donor_id]
                    contributions.append((donor_id, donor_rows[donor_id][j],
                                          distance, weight_map[donor_id], weight))
                    weights.append(weight)
                total_weight = math.fsum(weights)
                value = math.fsum(item[1] * (item[4] / total_weight) for item in contributions)
                output[j] = value
                sources[(sample_id, frame.features[j].id)] = {
                    "strategy": "neighbors", "fallback": None, "value": value,
                    "donors": [{"id": donor_id, "value": donor_value, "distance": distance,
                                "sample_weight": sample_weight, "weight": weight}
                               for donor_id, donor_value, distance, sample_weight, weight in contributions]}
            values.append(output)
        return values, sources

    def transform(self, frame):
        frame = self._check(frame)
        features = frame.features
        if self.indicators:
            features += tuple(Feature("missing:" + canonical(f.id), f.id, "missing_indicator")
                              for f in frame.features)
        values, _ = self._fill(frame)
        if self.indicators:
            for output, row in zip(values, frame.values):
                output.extend(float(value is None) for value in row)
        return frame.with_values(values, features)

    def transform_with_mask(self, frame, mask):
        """Transform while treating ``mask`` (not the frame's Nones) as the original observations."""
        frame = self._check(frame)
        values, _ = self._fill(frame, mask)
        return frame.with_values(values)

    def fill_sources(self, frame, mask=None):
        """Report the donor provenance of every missing entry without changing the frame."""
        frame = self._check(frame)
        _, sources = self._fill(frame, mask)
        return {sample_id: {feature_id: dict(source)
                            for (row_id, feature_id), source in sources.items() if row_id == sample_id}
                for sample_id in frame.ids}

    def state(self):
        if self.donors is None:
            raise NotFitted("cannot serialize an unfitted neighbor imputer")
        return {**self._axis_state(), "neighbors": self.neighbors, "min_common": self.min_common,
                "distance_weighted": self.distance_weighted, "fallback": self.fallback,
                "empty_value": self.empty_value, "indicators": self.indicators,
                "scales": list(self.scales), "fallback_fill": list(self.fallback_fill),
                "donors": self.donors.state()}

    @classmethod
    def restore(cls, state):
        result = cls(state["neighbors"], state["min_common"], state["distance_weighted"],
                     state["fallback"], state["empty_value"], state["indicators"])
        result._restore_axis(state)
        result.scales = tuple(finite(x) for x in state["scales"])
        result.fallback_fill = tuple(finite(x) for x in state["fallback_fill"])
        donors = Frame.restore(state["donors"])
        if donors.features != result.features:
            raise StateError("neighbor imputer donor axis mismatch")
        if len(result.scales) != len(result.features) or len(result.fallback_fill) != len(result.features):
            raise StateError("neighbor imputer state width mismatch")
        if any(x <= 0 for x in result.scales):
            raise StateError("neighbor imputer scales must be positive")
        result.donors = donors
        return result


class Standardizer(Transform):
    kind = "standardize"

    def __init__(self, center=True, scale=True):
        super().__init__()
        if type(center) is not bool or type(scale) is not bool:
            raise SchemaError("standardization options must be boolean")
        self.center = center
        self.scale = scale
        self.centers = None
        self.scales = None

    def fit(self, frame):
        centers = []
        scales = []
        for feature in frame.features:
            values = frame.column(feature.id)
            average = stats.mean(values, frame.weights)
            deviation = stats.standard_deviation(values, frame.weights)
            centers.append(average if self.center else 0.0)
            scales.append(deviation if self.scale and deviation > 0 else 1.0)
        self._fit_axis(frame)
        self.centers = tuple(centers)
        self.scales = tuple(scales)
        return self

    def transform(self, frame):
        frame = self._check(frame)
        return frame.with_values([[None if value is None else (value - self.centers[j]) / self.scales[j]
                                   for j, value in enumerate(row)] for row in frame.values])

    def inverse_transform(self, frame):
        frame = self._check(frame)
        return frame.with_values([[None if value is None else value * self.scales[j] + self.centers[j]
                                   for j, value in enumerate(row)] for row in frame.values])

    def state(self):
        return {**self._axis_state(), "center": self.center, "scale": self.scale,
                "centers": list(self.centers), "scales": list(self.scales)}

    @classmethod
    def restore(cls, state):
        result = cls(state["center"], state["scale"])
        result._restore_axis(state)
        result.centers = tuple(finite(x) for x in state["centers"])
        result.scales = tuple(finite(x) for x in state["scales"])
        if len(result.centers) != len(result.features) or len(result.scales) != len(result.features):
            raise StateError("standardizer state width mismatch")
        if any(x <= 0 for x in result.scales):
            raise StateError("standardizer scales must be positive")
        return result


class RangeScaler(Transform):
    kind = "range"

    def __init__(self, lower=0.0, upper=1.0, clip=False):
        super().__init__()
        self.lower = finite(lower)
        self.upper = finite(upper)
        if self.lower >= self.upper or type(clip) is not bool:
            raise SchemaError("invalid range scaling interval")
        self.clip = clip
        self.minimum = None
        self.span = None

    def fit(self, frame):
        minimum = []
        span = []
        for f in frame.features:
            pairs = stats.observed(frame.column(f.id), frame.weights)
            if not pairs:
                raise NumericalError("range scaling requires observed support")
            left = min(value for value, _ in pairs)
            right = max(value for value, _ in pairs)
            minimum.append(left)
            span.append(right - left if right > left else 1.0)
        self._fit_axis(frame)
        self.minimum = tuple(minimum)
        self.span = tuple(span)
        return self

    def transform(self, frame):
        frame = self._check(frame)
        values = []
        for row in frame.values:
            output = []
            for j, value in enumerate(row):
                if value is None:
                    output.append(None)
                    continue
                fraction = (value - self.minimum[j]) / self.span[j]
                if self.clip:
                    fraction = min(1.0, max(0.0, fraction))
                output.append(self.lower + fraction * (self.upper - self.lower))
            values.append(output)
        return frame.with_values(values)

    def state(self):
        return {**self._axis_state(), "lower": self.lower, "upper": self.upper, "clip": self.clip,
                "minimum": list(self.minimum), "span": list(self.span)}

    @classmethod
    def restore(cls, state):
        result = cls(state["lower"], state["upper"], state["clip"])
        result._restore_axis(state)
        result.minimum = tuple(finite(x) for x in state["minimum"])
        result.span = tuple(finite(x) for x in state["span"])
        if len(result.minimum) != len(result.features) or len(result.span) != len(result.features):
            raise StateError("range state width mismatch")
        if any(x <= 0 for x in result.span):
            raise StateError("range spans must be positive")
        return result


class Winsorizer(Transform):
    kind = "winsorize"

    def __init__(self, lower=.05, upper=.95):
        super().__init__()
        self.lower = finite(lower)
        self.upper = finite(upper)
        if not 0 <= self.lower <= self.upper <= 1:
            raise SchemaError("invalid winsorization quantiles")
        self.bounds = None

    def fit(self, frame):
        bounds = []
        for f in frame.features:
            values = frame.column(f.id)
            bounds.append((stats.quantile(values, self.lower, frame.weights),
                           stats.quantile(values, self.upper, frame.weights)))
        self._fit_axis(frame)
        self.bounds = tuple(bounds)
        return self

    def transform(self, frame):
        frame = self._check(frame)
        return frame.with_values([[None if value is None else min(self.bounds[j][1], max(self.bounds[j][0], value))
                                   for j, value in enumerate(row)] for row in frame.values])

    def state(self):
        return {**self._axis_state(), "lower": self.lower, "upper": self.upper,
                "bounds": [list(pair) for pair in self.bounds]}

    @classmethod
    def restore(cls, state):
        result = cls(state["lower"], state["upper"])
        result._restore_axis(state)
        result.bounds = tuple(tuple(finite(x) for x in pair) for pair in state["bounds"])
        if len(result.bounds) != len(result.features) or any(len(x) != 2 or x[0] > x[1] for x in result.bounds):
            raise StateError("invalid stored winsorization bounds")
        return result


class Polynomial(Transform):
    kind = "polynomial"

    def __init__(self, degree=2, interactions_only=False, max_features=4096):
        super().__init__()
        if type(degree) is not int or not 1 <= degree <= 4:
            raise SchemaError("polynomial degree must be between 1 and 4")
        if type(interactions_only) is not bool or type(max_features) is not int or max_features < 1:
            raise SchemaError("invalid polynomial configuration")
        self.degree = degree
        self.interactions_only = interactions_only
        self.max_features = max_features
        self.terms = None
        self.output_features = None

    def fit(self, frame):
        terms = []
        for degree in range(1, self.degree + 1):
            iterator = itertools.combinations if self.interactions_only else itertools.combinations_with_replacement
            for term in iterator(range(frame.width), degree):
                terms.append(term)
                if len(terms) > self.max_features:
                    raise SchemaError("polynomial expansion exceeds width budget")
        self._fit_axis(frame)
        self.terms = tuple(terms)
        features = []
        for term in terms:
            if len(term) == 1:
                features.append(frame.features[term[0]])
            else:
                ids = [frame.features[j].id for j in term]
                units = [frame.features[j].unit or "1" for j in term]
                features.append(Feature("product:" + canonical(ids), canonical(ids), "product", "*".join(units)))
        self.output_features = tuple(features)
        return self

    def transform(self, frame):
        frame = self._check(frame)
        values = []
        for row in frame.values:
            output = []
            for term in self.terms:
                output.append(None if any(row[j] is None for j in term) else math.prod(row[j] for j in term))
            values.append(output)
        return frame.with_values(values, self.output_features)

    def state(self):
        return {**self._axis_state(), "degree": self.degree, "interactions_only": self.interactions_only,
                "max_features": self.max_features, "terms": [list(t) for t in self.terms],
                "output_features": [f.state() for f in self.output_features]}

    @classmethod
    def restore(cls, state):
        result = cls(state["degree"], state["interactions_only"], state["max_features"])
        features = tuple(Feature(**f) for f in state["features"])
        result.fit(Frame([], features, []))
        if result.state() != state:
            raise StateError("polynomial state contradicts its declared expansion")
        return result


class FeatureSelector(Transform):
    kind = "select"

    def __init__(self, ids):
        super().__init__()
        self.ids = tuple(ids)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise SchemaError("feature selection requires unique identities")

    def fit(self, frame):
        frame.select(self.ids)
        self._fit_axis(frame)
        return self

    def transform(self, frame):
        return self._check(frame).select(self.ids)

    def state(self):
        return {**self._axis_state(), "ids": list(self.ids)}

    @classmethod
    def restore(cls, state):
        result = cls(state["ids"])
        result._restore_axis(state)
        if not set(result.ids) <= {f.id for f in result.features}:
            raise StateError("selection refers to absent feature")
        return result


TRANSFORMS = {c.kind: c for c in (Imputer, NeighborImputer, Standardizer, RangeScaler,
                                  Winsorizer, Polynomial, FeatureSelector)}


def restore_transform(state):
    try:
        method = TRANSFORMS[state["kind"]]
    except KeyError as error:
        raise StateError("unsupported transform kind") from error
    return method.restore(state)


def create_transform(spec):
    spec = dict(spec)
    try:
        method = TRANSFORMS[spec.pop("kind")]
    except KeyError as error:
        raise SchemaError("unsupported transform specification") from error
    try:
        return method(**spec)
    except TypeError as error:
        raise SchemaError("unknown transform option") from error
