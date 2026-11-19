import math
import random

from .errors import IdentityError, NumericalError, SchemaError
from .metrics import regression, grouped_regression, grouped_paired_bootstrap
from .pipeline import Predictor
from .statistics import mean, quantile


def targets(table, field_id):
    field = table.schema.field(field_id)
    if field.role != "target" or field.kind not in {"number", "integer"}:
        raise SchemaError("evaluation target must be numeric and explicitly marked")
    values = dict(zip(table.ids(), table.column(field_id)))
    if any(value is None for value in values.values()):
        raise SchemaError("evaluation cannot treat absent targets as zero")
    return values


def score(predictor, table, group_field=None):
    truth = targets(table, predictor.target)
    predictions = predictor.predict(table)
    weights = dict(zip(table.ids(), table.weights()))
    result = {"predictions": predictions, "metrics": regression(truth, predictions, weights),
              "population": list(table.ids()), "data_digest": table.digest(),
              "model_digest": predictor.digest()}
    if group_field is not None:
        groups = dict(zip(table.ids(), table.column(group_field)))
        result["groups"] = grouped_regression(truth, predictions, groups, weights)
    return result


def holdout(table, split, feature_spec, model_spec, target, group_field=None):
    train, test = split.apply(table)
    predictor = Predictor(feature_spec, model_spec, target).fit(train)
    return {"split": split.state(), "training_digest": train.digest(),
            "testing_digest": test.digest(), "result": score(predictor, test, group_field),
            "predictor": predictor}


def learning_curve(table, split, fractions, feature_spec, model_spec, target, seed=0):
    train, test = split.apply(table)
    fractions = tuple(fractions)
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise SchemaError("learning fractions must be within (0,1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise SchemaError("learning fractions must be strictly increasing")
    ids = sorted(train.ids())
    random.Random(seed).shuffle(ids)
    result = []
    previous = 0
    for fraction in fractions:
        count = max(1, round(len(ids) * fraction))
        if count == previous:
            continue
        subset = train.take(ids[:count])
        predictor = Predictor(feature_spec, model_spec, target).fit(subset)
        evaluated = score(predictor, test)
        result.append({"fraction": fraction, "training_rows": count, "training_ids": list(subset.ids()),
                       "testing_ids": list(test.ids()), "metrics": evaluated["metrics"]})
        previous = count
    return result


def permutation_importance(predictor, table, fields, repeats=5, seed=0):
    if type(repeats) is not int or repeats < 1:
        raise SchemaError("permutation repetitions must be positive")
    fields = tuple(fields)
    if len(set(fields)) != len(fields):
        raise IdentityError("duplicate importance field")
    baseline = score(predictor, table)["metrics"]["mse"]
    randomizer = random.Random(seed)
    results = {}
    for key in fields:
        field = table.schema.field(key)
        if field.role != "feature":
            raise SchemaError("permutation importance only changes feature fields")
        original = table.column(key)
        differences = []
        for _ in range(repeats):
            values = list(original)
            randomizer.shuffle(values)
            changed = table.replace_column(key, values)
            differences.append(score(predictor, changed)["metrics"]["mse"] - baseline)
        results[key] = {"mean_mse_increase": mean(differences), "repetitions": differences}
    return {"baseline_mse": baseline, "fields": results, "seed": seed,
            "interpretation": "unconditional row permutations; not a causal or grouped estimate"}


def bootstrap_mean(values, repeats=500, level=.95, seed=0):
    values = tuple(values)
    if not values or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise NumericalError("bootstrap needs finite observations")
    if type(repeats) is not int or repeats < 2 or not 0 < level < 1:
        raise SchemaError("invalid bootstrap configuration")
    generator = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        sample = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    tail = (1 - level) / 2
    return {"estimate": mean(values), "lower": quantile(estimates, tail),
            "upper": quantile(estimates, 1 - tail), "repeats": repeats, "seed": seed,
            "level": level, "resampling_unit": "independent_observation"}


def grouped_comparison(table, left, right, group_field, metric="mae", repeats=1000,
                        level=.95, seed=0):
    """Compare two fixed predictors by group-level paired bootstrap.

    The two predictors must share a target field, and predictions, targets, groups, and
    weights must all cover exactly the same sample identities.  Anything else is rejected
    rather than silently intersected, because dropping identities would change both the
    point estimate and the resampling population.
    """
    if not isinstance(left, Predictor) or not isinstance(right, Predictor):
        raise SchemaError("grouped comparison requires two predictors")
    if left.target != right.target:
        raise SchemaError("grouped comparison requires the same target field")
    if group_field not in table.schema.ids():
        raise IdentityError("group field is absent from the compared dataset")
    truth = targets(table, left.target)
    left_predictions = left.predict(table)
    right_predictions = right.predict(table)
    ids = set(table.ids())
    if set(truth) != ids or set(left_predictions) != ids or set(right_predictions) != ids:
        raise IdentityError("target, prediction, and dataset populations differ")
    groups = dict(zip(table.ids(), table.column(group_field)))
    if any(value is None for value in groups.values()):
        raise SchemaError("grouped comparison cannot treat missing groups as a category")
    weights = dict(zip(table.ids(), table.weights()))
    result = grouped_paired_bootstrap(truth, left_predictions, right_predictions, groups,
                                      weights, metric, repeats, level, seed, target=left.target)
    result.update({"group_field": group_field,
                   "predictions": {"left": left_predictions, "right": right_predictions},
                   "target_values": truth, "group_membership": groups,
                   "weights": weights, "population": list(table.ids()),
                   "data_digest": table.digest(),
                   "left_model_digest": left.digest(), "right_model_digest": right.digest()})
    return result


def classification_score(predictor, table, threshold=.5, group_field=None):
    from .estimators import Logistic
    from .metrics import classification_summary
    if not isinstance(predictor.model, Logistic):
        raise SchemaError("classification scoring requires a probability estimator")
    truth = targets(table, predictor.target)
    predictions = predictor.predict(table)
    weights = dict(zip(table.ids(), table.weights()))
    summary = classification_summary(truth, predictions, weights, threshold)
    groups = {}
    if group_field is not None:
        buckets = {}
        for key, group in zip(table.ids(), table.column(group_field)):
            if not isinstance(group, str) or not group:
                raise SchemaError("classification group must be nonempty text")
            buckets.setdefault(group, []).append(key)
        for group, ids in buckets.items():
            local_weights = {key: weights[key] for key in ids}
            if not any(local_weights.values()):
                groups[group] = {"status": "NO_POSITIVE_WEIGHT"}
            else:
                groups[group] = classification_summary({key: truth[key] for key in ids},
                    {key: predictions[key] for key in ids}, local_weights, threshold)
    return {"predictions": predictions, "metrics": summary, "groups": groups,
            "population": list(table.ids()), "data_digest": table.digest(), "model_digest": predictor.digest()}
