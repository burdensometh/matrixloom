import math

from .errors import NumericalError, SchemaError
from .schema import finite


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
    columns = [frame.column(f.id) for f in frame.features]
    result = []
    supports = []
    for left in columns:
        values = []
        counts = []
        for right in columns:
            try:
                values.append(covariance(left, right, frame.weights, ddof))
            except NumericalError:
                values.append(None)
            counts.append(sum(a is not None and b is not None and w > 0
                              for a, b, w in zip(left, right, frame.weights)))
        result.append(values)
        supports.append(counts)
    return {"features": list(frame.feature_ids()), "covariance": result,
            "support": supports, "population": "pairwise" if pairwise else "complete_rows", "ddof": ddof}


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
