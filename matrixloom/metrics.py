import math

from .errors import IdentityError, NumericalError, SchemaError
from .schema import finite
from .statistics import mean, quantile, effective_sample_size


def aligned(truth, predictions, weights=None, require_positive=True):
    if not isinstance(truth, dict) or not isinstance(predictions, dict):
        raise IdentityError("metrics require identity-keyed observations")
    if set(truth) != set(predictions):
        raise IdentityError("prediction and target populations differ")
    weights = {key: 1.0 for key in truth} if weights is None else dict(weights)
    if set(weights) != set(truth):
        raise IdentityError("metric weight population differs")
    result = []
    for key in truth:
        target = finite(truth[key], "metric target")
        predicted = finite(predictions[key], "metric prediction")
        weight = finite(weights[key], "metric weight")
        if weight < 0:
            raise SchemaError("metric weights cannot be negative")
        result.append((key, target, predicted, weight))
    if require_positive and not any(row[3] > 0 for row in result):
        raise NumericalError("metric population has no positive weight")
    return result


def regression(truth, predictions, weights=None):
    rows = aligned(truth, predictions, weights)
    target = [y for _, y, _, _ in rows]
    residual = [y - p for _, y, p, _ in rows]
    weight = [w for _, _, _, w in rows]
    average = mean(target, weight)
    mse = mean([x * x for x in residual], weight)
    target_variance = mean([(x - average) ** 2 for x in target], weight)
    r2 = None if target_variance == 0 else 1 - mse / target_variance
    return {"rows": len(rows), "positive_rows": sum(w > 0 for w in weight),
            "weight": math.fsum(weight), "effective_rows": effective_sample_size(weight),
            "mae": mean([abs(x) for x in residual], weight), "mse": mse,
            "rmse": math.sqrt(mse), "bias": mean(residual, weight), "r2": r2,
            "median_absolute_error": quantile([abs(x) for x in residual], .5, weight)}


def residual_table(truth, predictions, weights=None):
    return [{"id": key, "target": y, "prediction": p, "residual": y - p,
             "absolute_error": abs(y - p), "squared_error": (y - p) ** 2, "weight": w}
            for key, y, p, w in aligned(truth, predictions, weights)]


def grouped_regression(truth, predictions, groups, weights=None):
    aligned(truth, predictions, weights)
    if set(groups) != set(truth):
        raise IdentityError("group population differs")
    buckets = {}
    for key in truth:
        group = groups[key]
        if not isinstance(group, str) or not group:
            raise SchemaError("groups must be nonempty strings")
        buckets.setdefault(group, []).append(key)
    result = {}
    for group in sorted(buckets):
        keys = buckets[group]
        local_weights = None if weights is None else {key: weights[key] for key in keys}
        if local_weights is not None and not any(local_weights.values()):
            result[group] = {"status": "NO_POSITIVE_WEIGHT", "rows": len(keys)}
        else:
            result[group] = {"status": "OK", **regression({key: truth[key] for key in keys},
                                {key: predictions[key] for key in keys}, local_weights)}
    return result


def compare(truth, left, right, weights=None):
    left_rows = aligned(truth, left, weights)
    aligned(truth, right, weights)
    differences = [abs(y - p) - abs(y - right[key]) for key, y, p, w in left_rows]
    weight = [w for _, _, _, w in left_rows]
    return {"left": regression(truth, left, weights), "right": regression(truth, right, weights),
            "mae_difference": mean(differences, weight),
            "paired_differences": {row[0]: value for row, value in zip(left_rows, differences)}}


def binary(truth, probabilities, threshold=.5, weights=None):
    rows = aligned(truth, probabilities, weights)
    threshold = finite(threshold)
    if not 0 <= threshold <= 1:
        raise SchemaError("classification threshold outside [0,1]")
    if any(y not in (0, 1) or not 0 <= p <= 1 for _, y, p, _ in rows):
        raise SchemaError("binary metric requires binary targets and probabilities")
    tp = math.fsum(w for _, y, p, w in rows if y == 1 and p >= threshold)
    fp = math.fsum(w for _, y, p, w in rows if y == 0 and p >= threshold)
    tn = math.fsum(w for _, y, p, w in rows if y == 0 and p < threshold)
    fn = math.fsum(w for _, y, p, w in rows if y == 1 and p < threshold)
    total = tp + fp + tn + fn
    epsilon = 1e-15
    logloss = -math.fsum(w * (y * math.log(max(epsilon, p)) + (1 - y) * math.log(max(epsilon, 1 - p)))
                         for _, y, p, w in rows) / total
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / total,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
            "brier": math.fsum(w * (y - p) ** 2 for _, y, p, w in rows) / total,
            "logloss": logloss, "weight": total, "threshold": threshold}


def roc_auc(truth, probabilities, weights=None):
    rows = aligned(truth, probabilities, weights)
    if any(y not in (0, 1) for _, y, _, _ in rows):
        raise SchemaError("AUC requires binary targets")
    positive = [(p, w) for _, y, p, w in rows if y == 1 and w > 0]
    negative = [(p, w) for _, y, p, w in rows if y == 0 and w > 0]
    if not positive or not negative:
        raise NumericalError("AUC requires positive support for both labels")
    numerator = math.fsum(wa * wb * (1 if a > b else .5 if a == b else 0)
                          for a, wa in positive for b, wb in negative)
    return numerator / (math.fsum(w for _, w in positive) * math.fsum(w for _, w in negative))


def silhouette(frame, labels):
    """Mean silhouette over assignable samples with available-case scaling.

    ``labels`` may carry ``None`` for unassignable samples (for example fully
    unobserved rows); those samples are excluded from the population instead
    of being treated as an ordinary cluster.  Pairwise distances use the same
    observed-dimension scaling as the clustering objective, and pairs that
    share no observed dimension are excluded from that sample's comparison.
    """
    if set(labels) != set(frame.ids):
        raise IdentityError("clustering labels do not match population")
    populated = [key for key in frame.ids if labels[key] is not None]
    unassignable = [key for key in frame.ids if labels[key] is None]
    indexes = {key: i for i, key in enumerate(frame.ids)}
    groups = {}
    for key in populated:
        groups.setdefault(labels[key], []).append(key)
    if len(groups) < 2 or len(groups) == len(populated):
        raise NumericalError("silhouette requires 2..n-1 populated clusters")
    width = frame.width
    scores = []
    scored = []
    for key in populated:
        i = indexes[key]
        own = [frame.values[indexes[peer]] for peer in groups[labels[key]] if peer != key]
        if not own:
            scores.append(0.0)
            scored.append(key)
            continue
        a = _mean_observed_distance(frame.values[i], own, width)
        other_distances = []
        for group, members in groups.items():
            if group != labels[key]:
                value = _mean_observed_distance(frame.values[i],
                                                [frame.values[indexes[peer]] for peer in members], width)
                if value is not None:
                    other_distances.append(value)
        if a is None or not other_distances:
            continue
        b = min(other_distances)
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
        scored.append(key)
    if not scores:
        raise NumericalError("silhouette has no comparable observed population")
    return {"mean": mean(scores), "by_sample": dict(zip(scored, scores)), "weighted": False,
            "assignable": len(populated), "unassignable": len(unassignable),
            "unassignable_ids": unassignable,
            "support": "available_case" if any(value is None for row in frame.values for value in row)
                       else "complete"}


def _mean_observed_distance(left, others, width):
    distances = []
    for right in others:
        total = 0.0
        observed = 0
        for a, b in zip(left, right):
            if a is None or b is None:
                continue
            observed += 1
            total += (a - b) ** 2
        if observed:
            distances.append(math.sqrt(total if observed == width else total * width / observed))
    return math.fsum(distances) / len(distances) if distances else None


def threshold_curve(truth, probabilities, weights=None):
    rows = aligned(truth, probabilities, weights)
    if any(y not in (0, 1) or not 0 <= p <= 1 for _, y, p, _ in rows):
        raise SchemaError("threshold curve requires binary targets and probabilities")
    buckets = {}
    for _, target, probability, weight in rows:
        positive, negative = buckets.setdefault(probability, ([], []))
        (positive if target == 1 else negative).append(weight)
    positive_total = math.fsum(w for _, y, _, w in rows if y == 1)
    negative_total = math.fsum(w for _, y, _, w in rows if y == 0)
    if positive_total <= 0 or negative_total <= 0:
        raise NumericalError("threshold curve needs both labels")
    points = [{"threshold": None, "tp": 0.0, "fp": 0.0, "recall": 0.0, "fpr": 0.0, "precision": 1.0}]
    tp = fp = 0.0
    for threshold in sorted(buckets, reverse=True):
        positive, negative = buckets[threshold]
        tp += math.fsum(positive)
        fp += math.fsum(negative)
        points.append({"threshold": threshold, "tp": tp, "fp": fp,
                       "recall": tp / positive_total, "fpr": fp / negative_total,
                       "precision": tp / (tp + fp) if tp + fp > 0 else 1.0})
    return points


def average_precision(truth, probabilities, weights=None):
    points = threshold_curve(truth, probabilities, weights)
    return math.fsum((b["recall"] - a["recall"]) * b["precision"] for a, b in zip(points, points[1:]))


def classification_summary(truth, probabilities, weights=None, threshold=.5):
    summary = binary(truth, probabilities, threshold, weights)
    try:
        auc = roc_auc(truth, probabilities, weights)
        ap = average_precision(truth, probabilities, weights)
        status = "OK"
    except NumericalError:
        auc = ap = None
        status = "SINGLE_OBSERVED_CLASS"
    return {**summary, "roc_auc": auc, "average_precision": ap, "ranking_status": status}


class RegressionAccumulator:
    """Online population-weighted regression losses with centered target variance."""

    def __init__(self):
        self.rows = 0
        self.positive_rows = 0
        self.weight = 0.0
        self.squared_weight = 0.0
        self.target_mean = 0.0
        self.target_m2 = 0.0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.signed_error = 0.0

    def update(self, truth, predictions, weights=None):
        rows = aligned(truth, predictions, weights, require_positive=False)
        for _, y, prediction, weight in rows:
            self.rows += 1
            if weight == 0:
                continue
            new_weight = self.weight + weight
            delta = y - self.target_mean
            new_mean = self.target_mean + delta * (weight / new_weight)
            self.target_m2 += weight * delta * (y - new_mean)
            self.target_mean = new_mean
            self.weight = new_weight
            self.squared_weight += weight * weight
            self.positive_rows += 1
            error = y - prediction
            self.absolute_error += weight * abs(error)
            self.squared_error += weight * error * error
            self.signed_error += weight * error
        return self

    def result(self):
        if self.weight <= 0:
            raise NumericalError("accumulator has no positive support")
        mse = self.squared_error / self.weight
        return {"rows": self.rows, "positive_rows": self.positive_rows, "weight": self.weight,
                "effective_rows": self.weight * self.weight / self.squared_weight,
                "mae": self.absolute_error / self.weight, "mse": mse, "rmse": math.sqrt(mse),
                "bias": self.signed_error / self.weight,
                "r2": None if self.target_m2 <= 0 else 1 - self.squared_error / self.target_m2}


def pinball(truth, predictions, probability, weights=None):
    probability = finite(probability)
    if not 0 < probability < 1:
        raise SchemaError("quantile probability must be inside (0,1)")
    rows = aligned(truth, predictions, weights)
    loss = []
    for _, target, prediction, weight in rows:
        error = target - prediction
        loss.append(weight * (probability * error if error >= 0 else (probability - 1) * error))
    return math.fsum(loss) / math.fsum(w for _, _, _, w in rows)


def interval_score(truth, lower, upper, level=.9, weights=None):
    level = finite(level)
    if not 0 < level < 1:
        raise SchemaError("interval level must be inside (0,1)")
    rows = aligned(truth, lower, weights)
    aligned(truth, upper, weights)
    if any(lo > upper[key] for key, _, lo, _ in rows):
        raise SchemaError("prediction interval endpoints are reversed")
    coverage = []
    width = []
    penalty = []
    for key, y, lo, weight in rows:
        hi = upper[key]
        coverage.append(weight * (lo <= y <= hi))
        width.append(weight * (hi - lo))
        outside = lo - y if y < lo else y - hi if y > hi else 0.0
        penalty.append(weight * ((hi - lo) + 2 * outside / (1 - level)))
    total = math.fsum(w for _, _, _, w in rows)
    return {"coverage": math.fsum(coverage) / total, "mean_width": math.fsum(width) / total,
            "interval_score": math.fsum(penalty) / total, "level": level, "weight": total}
