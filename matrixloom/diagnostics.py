import math

from .errors import NumericalError, SchemaError
from .schema import finite
from . import statistics as stats
from . import linalg as la


def missing_by_group(table, group_field, fields):
    groups = {}
    for sample, group in zip(table.samples, table.column(group_field)):
        groups.setdefault(group, []).append(sample.id)
    result = []
    for group, ids in groups.items():
        subset = table.take(ids)
        for field_id in fields:
            values = subset.column(field_id)
            missing = [key for key, value in zip(subset.ids(), values) if value is None]
            observed_weight = math.fsum(w for value, w in zip(values, subset.weights()) if value is not None)
            result.append({"group": group, "field": field_id, "rows": len(subset),
                           "missing_ids": missing, "observed_weight": observed_weight})
    return result


def contingency(table, left, right):
    a = table.column(left)
    b = table.column(right)
    buckets = {}
    omitted = 0.0
    for x, y, w in zip(a, b, table.weights()):
        if x is None or y is None:
            omitted += w
        elif w > 0:
            buckets.setdefault((x, y), []).append(w)
    cells = [{"left": x, "right": y, "weight": math.fsum(values)}
             for (x, y), values in sorted(buckets.items(), key=lambda item: repr(item[0]))]
    return {"left": left, "right": right, "cells": cells, "omitted_weight": omitted}


def associations(frame, method="pearson"):
    if method not in {"pearson", "spearman"}:
        raise SchemaError("unsupported association method")
    result = []
    for i, left in enumerate(frame.features):
        for right in frame.features[i + 1:]:
            a, b = frame.column(left.id), frame.column(right.id)
            try:
                value = stats.correlation(a, b, frame.weights) if method == "pearson" else stats.spearman(a, b)
                status = "OK"
            except NumericalError:
                value, status = None, "INSUFFICIENT_OR_CONSTANT_SUPPORT"
            result.append({"left": left.id, "right": right.id, "correlation": value,
                           "status": status, "method": method,
                           "weighted": method == "pearson"})
    return result


def outliers(frame, factor=1.5):
    factor = finite(factor)
    if factor <= 0:
        raise SchemaError("outlier factor must be positive")
    result = {}
    for feature in frame.features:
        values = frame.column(feature.id)
        try:
            q1 = stats.quantile(values, .25, frame.weights)
            q3 = stats.quantile(values, .75, frame.weights)
        except NumericalError:
            result[feature.id] = {"status": "NO_SUPPORT", "sample_ids": []}
            continue
        lower = q1 - factor * (q3 - q1)
        upper = q3 + factor * (q3 - q1)
        ids = [key for key, value in zip(frame.ids, values) if value is not None and not lower <= value <= upper]
        result[feature.id] = {"status": "OK", "lower": lower, "upper": upper, "sample_ids": ids}
    return result


def weighted_ks(reference, current, reference_weights=None, current_weights=None):
    a = stats.observed(reference, reference_weights)
    b = stats.observed(current, current_weights)
    total_a = stats.total_weight(a)
    total_b = stats.total_weight(b)
    a = sorted(a)
    b = sorted(b)
    points = sorted({x for x, _ in a} | {x for x, _ in b})
    i = j = 0
    cum_a = cum_b = distance = 0.0
    for point in points:
        while i < len(a) and a[i][0] <= point:
            cum_a += a[i][1] / total_a
            i += 1
        while j < len(b) and b[j][0] <= point:
            cum_b += b[j][1] / total_b
            j += 1
        distance = max(distance, abs(cum_a - cum_b))
    return min(1.0, distance)


def category_shift(reference, current, reference_weights=None, current_weights=None):
    a = stats.frequency(reference, reference_weights)
    b = stats.frequency(current, current_weights)
    total_a = math.fsum(a.values())
    total_b = math.fsum(b.values())
    if total_a <= 0 or total_b <= 0:
        raise NumericalError("categorical shift requires positive support")
    keys = set(a) | set(b)
    divergence = .5 * math.fsum(abs(a.get(key, 0) / total_a - b.get(key, 0) / total_b) for key in keys)
    unseen = sorted((key for key in b if key not in a), key=repr)
    return {"total_variation": divergence, "new_categories": unseen,
            "reference_weight": total_a, "current_weight": total_b}


def drift(reference, current, fields):
    results = {}
    for key in fields:
        a = reference.schema.field(key)
        b = current.schema.field(key)
        if (a.kind, a.unit) != (b.kind, b.unit):
            raise SchemaError("drift requires compatible feature semantics")
        left, right = reference.column(key), current.column(key)
        if a.kind == "category":
            results[key] = {"kind": "category", **category_shift(left, right, reference.weights(), current.weights())}
        elif a.kind in {"number", "integer", "boolean"}:
            left = [None if x is None else float(x) for x in left]
            right = [None if x is None else float(x) for x in right]
            try:
                results[key] = {"kind": "number", "status": "OK",
                    "ks_distance": weighted_ks(left, right, reference.weights(), current.weights()),
                    "mean_shift": stats.mean(right, current.weights()) - stats.mean(left, reference.weights())}
            except NumericalError:
                results[key] = {"kind": "number", "status": "NO_OBSERVED_SUPPORT"}
        else:
            raise SchemaError("derive numeric time representations before drift analysis")
    return results


def collinearity(frame):
    frame.complete()
    profile = stats.covariance_matrix(frame)
    return collinearity_profile(profile)


def collinearity_profile(profile):
    """Collinearity diagnostic on a covariance result, including a batch-merged one."""
    covariance = profile["covariance"]
    features = profile["features"]
    if not features or any(x is None for row in covariance for x in row):
        raise NumericalError("collinearity requires observed support")
    try:
        inverse = la.inverse(covariance)
        condition = la.condition_indicator(covariance)
    except NumericalError:
        return {"status": "SINGULAR", "features": list(features), "vif": None, "condition": None}
    vif = {feature: covariance[i][i] * inverse[i][i] for i, feature in enumerate(features)}
    return {"status": "OK", "features": list(features), "vif": vif, "condition": condition}
