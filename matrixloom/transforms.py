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


TRANSFORMS = {c.kind: c for c in (Imputer, Standardizer, RangeScaler, Winsorizer, Polynomial, FeatureSelector)}


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
