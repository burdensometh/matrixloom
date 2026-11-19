import math

from .errors import IdentityError, NumericalError, SchemaError, StateError
from .frame import Feature, Frame
from .schema import finite, fingerprint


def observed(values, weights=None):
    values = tuple(values)
    weights = (1.0,) * len(values) if weights is None else tuple(weights)
    if len(weights) != len(values):
        raise SchemaError("statistics populations differ")
    result = []
    for value, weight in zip(values, weights):
        weight = finite(weight, "weight")
        if weight < 0:
            raise SchemaError("negative statistics weight")
        if value is not None and weight > 0:
            result.append((finite(value), weight))
    return result


def total_weight(pairs):
    result = math.fsum(w for _, w in pairs)
    if result <= 0:
        raise NumericalError("no positive observed weight")
    return result


def mean(values, weights=None):
    pairs = observed(values, weights)
    total = total_weight(pairs)
    return math.fsum(x * (w / total) for x, w in pairs)


def variance(values, weights=None, ddof=0):
    pairs = observed(values, weights)
    total = total_weight(pairs)
    if ddof not in (0, 1):
        raise SchemaError("ddof must be 0 or 1")
    center = mean([x for x, _ in pairs], [w for _, w in pairs])
    denominator = total if ddof == 0 else total - math.fsum(w * w for _, w in pairs) / total
    if denominator <= 0:
        raise NumericalError("insufficient effective population for variance")
    return math.fsum(w * (x - center) ** 2 for x, w in pairs) / denominator


def standard_deviation(values, weights=None, ddof=0):
    return math.sqrt(variance(values, weights, ddof))


def quantile(values, probability, weights=None):
    probability = finite(probability, "quantile probability")
    if not 0 <= probability <= 1:
        raise SchemaError("quantile probability outside [0,1]")
    pairs = sorted(observed(values, weights))
    total = total_weight(pairs)
    threshold = probability * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def median(values, weights=None):
    return quantile(values, 0.5, weights)


def absolute_deviation(values, weights=None):
    pairs = observed(values, weights)
    center = median([x for x, _ in pairs], [w for _, w in pairs])
    return median([abs(x - center) for x, _ in pairs], [w for _, w in pairs])


def covariance(left, right, weights=None, ddof=0):
    if len(left) != len(right):
        raise SchemaError("covariance populations differ")
    weights = (1.0,) * len(left) if weights is None else tuple(weights)
    if len(weights) != len(left):
        raise SchemaError("covariance weight population differs")
    pairs = []
    for a, b, weight in zip(left, right, weights):
        weight = finite(weight, "weight")
        if weight < 0:
            raise SchemaError("negative covariance weight")
        if a is not None and b is not None and weight > 0:
            pairs.append((finite(a), finite(b), weight))
    if ddof not in (0, 1):
        raise SchemaError("ddof must be 0 or 1")
    total = total_weight([(a, w) for a, _, w in pairs])
    a_mean = math.fsum(a * (w / total) for a, _, w in pairs)
    b_mean = math.fsum(b * (w / total) for _, b, w in pairs)
    denominator = total if ddof == 0 else total - math.fsum(w * w for _, _, w in pairs) / total
    if denominator <= 0:
        raise NumericalError("insufficient paired population")
    return math.fsum(w * (a - a_mean) * (b - b_mean) for a, b, w in pairs) / denominator


class _PairMoments:
    """Weighted Welford moments for one feature pair over its own jointly observed samples.

    Missing entries never reach this class: the caller only feeds rows where both
    features are observed and the sample carries positive weight. Each pair therefore
    keeps its own center and support; no column-wide mean is ever shared across pairs.
    """

    __slots__ = ("weight", "mean_a", "mean_b", "comoment", "squared_weight", "observed")

    def __init__(self):
        self.weight = 0.0
        self.mean_a = 0.0
        self.mean_b = 0.0
        self.comoment = 0.0
        self.squared_weight = 0.0
        self.observed = 0

    def observe(self, a, b, weight):
        self.observed += 1
        self.squared_weight += weight * weight
        new_weight = self.weight + weight
        delta_a = a - self.mean_a
        delta_b = b - self.mean_b
        self.mean_a += delta_a * (weight / new_weight)
        self.mean_b += delta_b * (weight / new_weight)
        self.comoment += weight * delta_a * (b - self.mean_b)
        self.weight = new_weight

    @classmethod
    def combine(cls, left, right):
        result = cls()
        if left.weight <= 0.0:
            return cls._copy(right)
        if right.weight <= 0.0:
            return cls._copy(left)
        total = left.weight + right.weight
        gap_a = right.mean_a - left.mean_a
        gap_b = right.mean_b - left.mean_b
        result.weight = total
        result.mean_a = left.mean_a + gap_a * (right.weight / total)
        result.mean_b = left.mean_b + gap_b * (right.weight / total)
        result.comoment = (left.comoment + right.comoment
                           + gap_a * gap_b * (left.weight * right.weight / total))
        result.squared_weight = left.squared_weight + right.squared_weight
        result.observed = left.observed + right.observed
        return result

    @classmethod
    def _copy(cls, other):
        result = cls()
        result.weight = other.weight
        result.mean_a = other.mean_a
        result.mean_b = other.mean_b
        result.comoment = other.comoment
        result.squared_weight = other.squared_weight
        result.observed = other.observed
        return result

    def denominator(self, ddof):
        scale = self.squared_weight / self.weight if ddof == 1 else 0.0
        return self.weight - scale

    def covariance(self, ddof):
        if self.weight <= 0.0:
            return None
        denominator = self.denominator(ddof)
        if denominator <= 0.0:
            return None
        return self.comoment / denominator

    def state(self):
        return {"weight": self.weight, "mean_a": self.mean_a, "mean_b": self.mean_b,
                "comoment": self.comoment, "squared_weight": self.squared_weight,
                "observed": self.observed}

    @classmethod
    def restore(cls, state):
        result = cls()
        result.weight = float(state["weight"])
        result.mean_a = float(state["mean_a"])
        result.mean_b = float(state["mean_b"])
        result.comoment = float(state["comoment"])
        result.squared_weight = float(state["squared_weight"])
        result.observed = int(state["observed"])
        return result


class CovarianceAccumulator:
    """Batch-wise weighted covariance with pair-specific pairwise-deletion support.

    Every feature pair maintains independent moments over the samples where both
    features are observed with positive weight, so batches with different missing
    patterns combine exactly as if all observations were processed together.
    Merging is order independent; no original rows are retained.
    """

    def __init__(self, features, ddof=0, training_digest=None):
        features = tuple(features)
        if not features or any(not isinstance(feature, Feature) for feature in features):
            raise SchemaError("covariance accumulator requires feature lineage")
        if len({feature.id for feature in features}) != len(features):
            raise IdentityError("duplicate feature identity")
        if ddof not in (0, 1):
            raise SchemaError("ddof must be 0 or 1")
        self.features = features
        self.ddof = ddof
        self.training_digest = training_digest
        size = len(features)
        self._moments = [[_PairMoments() if j >= i else None for j in range(size)]
                         for i in range(size)]

    @property
    def width(self):
        return len(self.features)

    def _check_axis(self, frame):
        if frame.features != self.features:
            raise SchemaError("covariance batch feature axis differs from the accumulator")

    def update(self, frame):
        if not isinstance(frame, Frame):
            raise SchemaError("covariance updates require a Frame")
        self._check_axis(frame)
        width = self.width
        for row, weight in zip(frame.values, frame.weights):
            if weight <= 0:
                continue
            observed = [i for i in range(width) if row[i] is not None]
            for position, i in enumerate(observed):
                for j in observed[position:]:
                    self._moments[i][j].observe(row[i], row[j], weight)
        return self

    def merge(self, other):
        if not isinstance(other, CovarianceAccumulator):
            raise SchemaError("can only merge covariance accumulators")
        if other.features != self.features:
            raise SchemaError("covariance merge feature axes differ")
        if other.ddof != self.ddof:
            raise SchemaError("covariance merge requires matching ddof")
        for i in range(self.width):
            for j in range(i, self.width):
                self._moments[i][j] = _PairMoments.combine(self._moments[i][j], other._moments[i][j])
        return self

    def pair(self, left_id, right_id):
        indexes = {feature.id: i for i, feature in enumerate(self.features)}
        if left_id not in indexes or right_id not in indexes:
            raise IdentityError("unknown covariance feature")
        i, j = indexes[left_id], indexes[right_id]
        if i > j:
            i, j = j, i
        return self._moments[i][j]

    def result(self, population="pairwise"):
        if population not in {"pairwise", "complete_rows"}:
            raise SchemaError("covariance population must be pairwise or complete_rows")
        covariance, support, effective_support, weights = [], [], [], []
        for i in range(self.width):
            row_values, row_support, row_effective, row_weights = [], [], [], []
            for j in range(self.width):
                moments = self._moments[min(i, j)][max(i, j)]
                row_values.append(moments.covariance(self.ddof))
                row_support.append(moments.observed)
                row_weights.append(moments.weight if moments.weight > 0 else None)
                if moments.weight > 0:
                    denominator = moments.denominator(self.ddof)
                    row_effective.append(moments.weight * moments.weight / moments.squared_weight
                                         if denominator > 0 else None)
                else:
                    row_effective.append(None)
            covariance.append(row_values)
            support.append(row_support)
            effective_support.append(row_effective)
            weights.append(row_weights)
        return {"features": [feature.id for feature in self.features], "covariance": covariance,
                "support": support, "effective_support": effective_support, "pair_weight": weights,
                "population": population, "ddof": self.ddof}

    def state(self):
        pairs = []
        for i in range(self.width):
            for j in range(i, self.width):
                pairs.append({"left": i, "right": j, **self._moments[i][j].state()})
        return {"version": 1, "kind": "covariance_accumulator", "ddof": self.ddof,
                "training_digest": self.training_digest,
                "features": [feature.state() for feature in self.features], "pairs": pairs}

    def digest(self):
        return fingerprint(self.state())

    @classmethod
    def restore(cls, state):
        if state.get("version") != 1 or state.get("kind") != "covariance_accumulator":
            raise StateError("unsupported covariance accumulator state")
        result = cls([Feature(**feature) for feature in state["features"]], state["ddof"],
                     state.get("training_digest"))
        for entry in state["pairs"]:
            i, j = entry["left"], entry["right"]
            if not (0 <= i <= j < result.width):
                raise StateError("covariance pair index outside feature axis")
            result._moments[i][j] = _PairMoments.restore(entry)
        return result


def correlation(left, right, weights=None):
    weights = (1.0,) * len(left) if weights is None else tuple(weights)
    if len(left) != len(right) or len(left) != len(weights):
        raise SchemaError("correlation populations differ")
    kept = [(a, b, w) for a, b, w in zip(left, right, weights) if a is not None and b is not None]
    a = [x for x, _, _ in kept]
    b = [y for _, y, _ in kept]
    w = [z for _, _, z in kept]
    product = variance(a, w) * variance(b, w)
    if product <= 0:
        raise NumericalError("correlation undefined for a constant feature")
    return max(-1.0, min(1.0, covariance(a, b, w) / math.sqrt(product)))


def summarize(frame):
    result = {}
    for feature in frame.features:
        values = frame.column(feature.id)
        pairs = observed(values, frame.weights)
        info = {"rows": len(frame), "observed": len(pairs),
                "missing": sum(x is None for x in values),
                "positive_weight": math.fsum(w for _, w in pairs),
                "unit": feature.unit}
        if pairs:
            info.update(mean=mean(values, frame.weights), variance=variance(values, frame.weights),
                        minimum=min(x for x, _ in pairs), maximum=max(x for x, _ in pairs),
                        q25=quantile(values, .25, frame.weights), median=median(values, frame.weights),
                        q75=quantile(values, .75, frame.weights))
        else:
            info.update({key: None for key in ("mean", "variance", "minimum", "maximum", "q25", "median", "q75")})
        result[feature.id] = info
    return result


def covariance_matrix(frame, ddof=0, pairwise=False):
    if not pairwise:
        frame = frame.take(key for key, row in zip(frame.ids, frame.values) if all(x is not None for x in row))
    accumulator = CovarianceAccumulator(frame.features, ddof).update(frame)
    return accumulator.result("pairwise" if pairwise else "complete_rows")


def frequency(values, weights=None):
    values = tuple(values)
    weights = (1.0,) * len(values) if weights is None else tuple(weights)
    if len(values) != len(weights):
        raise SchemaError("frequency populations differ")
    buckets = {}
    for value, weight in zip(values, weights):
        weight = finite(weight, "weight")
        if weight < 0:
            raise SchemaError("negative frequency weight")
        if weight > 0:
            buckets.setdefault(value, []).append(weight)
    return {key: math.fsum(items) for key, items in buckets.items()}


def effective_sample_size(weights):
    pairs = observed([1.0] * len(weights), weights)
    total = total_weight(pairs)
    return total * total / math.fsum(w * w for _, w in pairs)


def histogram(values, boundaries, weights=None):
    boundaries = tuple(finite(x, "bin boundary") for x in boundaries)
    if len(boundaries) < 2 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise SchemaError("histogram boundaries must be strictly increasing")
    pairs = observed(values, weights)
    bins = [[] for _ in range(len(boundaries) - 1)]
    outside = []
    for value, weight in pairs:
        if value < boundaries[0] or value > boundaries[-1]:
            outside.append(weight)
            continue
        for j, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            if left <= value < right or j == len(bins) - 1 and value == right:
                bins[j].append(weight)
                break
    return {"boundaries": list(boundaries), "weight": [math.fsum(items) for items in bins],
            "outside_weight": math.fsum(outside)}


def rank(values):
    if any(x is None for x in values):
        raise SchemaError("ranking requires explicit removal of missing observations")
    indexes = sorted(range(len(values)), key=lambda i: finite(values[i]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexes):
        end = start + 1
        while end < len(indexes) and values[indexes[end]] == values[indexes[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for i in indexes[start:end]:
            ranks[i] = average
        start = end
    return ranks


def spearman(left, right):
    if len(left) != len(right):
        raise SchemaError("rank populations differ")
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    return correlation(rank([a for a, _ in pairs]), rank([b for _, b in pairs]))
