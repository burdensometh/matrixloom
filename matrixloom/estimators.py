import math
import random

from .errors import NotFitted, NumericalError, SchemaError, StateError
from .frame import Feature, Frame, align_targets
from . import linalg as la
from .schema import finite
from .statistics import mean, quantile


class Regressor:
    kind = None

    def __init__(self):
        self.features = None

    def _training(self, frame, targets):
        frame.complete()
        values = align_targets(frame, targets)
        if not frame.features or not frame.ids or frame.total_weight() <= 0:
            raise SchemaError("training requires features and positive sample weight")
        return values

    def _prediction(self, frame):
        if self.features is None:
            raise NotFitted("estimator has not been fitted")
        return frame.align_features(self.features).complete()


class Ridge(Regressor):
    kind = "ridge"

    def __init__(self, alpha=1.0, intercept=True, pivot_tolerance=1e-12):
        super().__init__()
        self.alpha = finite(alpha, "ridge alpha")
        self.pivot_tolerance = finite(pivot_tolerance, "pivot tolerance")
        if self.alpha < 0 or self.pivot_tolerance <= 0 or type(intercept) is not bool:
            raise SchemaError("invalid ridge configuration")
        self.intercept = intercept
        self.coefficients = None
        self.offset = None
        self.training_weight = None
        self.condition = None

    def fit(self, frame, targets):
        y = self._training(frame, targets)
        total = frame.total_weight()
        center_x = [mean(frame.column(f.id), frame.weights) if self.intercept else 0.0 for f in frame.features]
        center_y = mean(y, frame.weights) if self.intercept else 0.0
        values = [[x - center_x[j] for j, x in enumerate(row)] for row in frame.values]
        centered_y = [value - center_y for value in y]
        gram = la.weighted_gram(values, frame.weights)
        rhs = la.weighted_rhs(values, centered_y, frame.weights)
        for j in range(frame.width):
            gram[j][j] += self.alpha
        coefficients = la.solve(gram, rhs, self.pivot_tolerance)
        condition = la.condition_indicator(gram)
        offset = center_y - la.dot(coefficients, center_x)
        self.features = frame.features
        self.coefficients = tuple(coefficients)
        self.offset = offset
        self.training_weight = total
        self.condition = condition
        return self

    def predict(self, frame):
        frame = self._prediction(frame)
        return {key: self.offset + la.dot(row, self.coefficients) for key, row in zip(frame.ids, frame.values)}

    def contributions(self, frame):
        frame = self._prediction(frame)
        return {key: {"intercept": self.offset,
                      "features": {f.id: value * coefficient for f, value, coefficient
                                   in zip(frame.features, row, self.coefficients)}}
                for key, row in zip(frame.ids, frame.values)}

    def state(self):
        if self.features is None:
            raise NotFitted("cannot serialize an unfitted ridge")
        return {"kind": self.kind, "alpha": self.alpha, "intercept": self.intercept,
                "pivot_tolerance": self.pivot_tolerance, "features": [f.state() for f in self.features],
                "coefficients": list(self.coefficients), "offset": self.offset,
                "training_weight": self.training_weight, "condition": self.condition}

    @classmethod
    def restore(cls, state):
        result = cls(state["alpha"], state["intercept"], state["pivot_tolerance"])
        result.features = tuple(Feature(**f) for f in state["features"])
        result.coefficients = tuple(finite(x) for x in state["coefficients"])
        result.offset = finite(state["offset"])
        result.training_weight = finite(state["training_weight"])
        result.condition = finite(state["condition"])
        if not result.features or len(result.coefficients) != len(result.features):
            raise StateError("ridge coefficient axis mismatch")
        if result.training_weight <= 0 or result.condition < 1 - 1e-10:
            raise StateError("invalid ridge support or condition")
        return result


class Neighbors(Regressor):
    kind = "neighbors"

    def __init__(self, neighbors=5, distance_weighted=False):
        super().__init__()
        if type(neighbors) is not int or neighbors < 1 or type(distance_weighted) is not bool:
            raise SchemaError("invalid nearest-neighbor configuration")
        self.neighbors = neighbors
        self.distance_weighted = distance_weighted
        self.training = None
        self.targets = None

    def fit(self, frame, targets):
        self._training(frame, targets)
        positive = frame.positive_population()
        self.features = positive.features
        self.training = positive
        self.targets = {key: finite(targets[key]) for key in positive.ids}
        return self

    def kneighbors(self, frame, exclude_same_id=False):
        frame = self._prediction(frame)
        result = {}
        for key, row in zip(frame.ids, frame.values):
            distances = []
            for peer, values in zip(self.training.ids, self.training.values):
                if exclude_same_id and peer == key:
                    continue
                distances.append((la.squared_distance(row, values), peer))
            distances.sort(key=lambda item: (item[0], item[1]))
            result[key] = [{"id": peer, "distance": math.sqrt(distance)}
                           for distance, peer in distances[:self.neighbors]]
        return result

    def predict(self, frame):
        neighbors = self.kneighbors(frame)
        weight_map = dict(zip(self.training.ids, self.training.weights))
        result = {}
        for key, items in neighbors.items():
            if not items:
                raise NumericalError("no training neighbors")
            if self.distance_weighted and any(item["distance"] == 0 for item in items):
                items = [item for item in items if item["distance"] == 0]
            weights = [weight_map[item["id"]] / item["distance"]
                       if self.distance_weighted and item["distance"] > 0 else weight_map[item["id"]]
                       for item in items]
            result[key] = mean([self.targets[item["id"]] for item in items], weights)
        return result

    def state(self):
        if self.training is None:
            raise NotFitted("cannot serialize unfitted neighbors")
        return {"kind": self.kind, "neighbors": self.neighbors,
                "distance_weighted": self.distance_weighted, "training": self.training.state(),
                "targets": dict(self.targets)}

    def predict_quantiles(self, frame, probabilities=(.1, .5, .9)):
        probabilities = tuple(finite(p) for p in probabilities)
        if not probabilities or any(not 0 < p < 1 for p in probabilities):
            raise SchemaError("neighbor quantiles must lie inside (0,1)")
        if len(set(probabilities)) != len(probabilities):
            raise SchemaError("duplicate quantile probability")
        weight_map = dict(zip(self.training.ids, self.training.weights)) if self.training is not None else {}
        result = {}
        for key, items in self.kneighbors(frame).items():
            if self.distance_weighted and any(item["distance"] == 0 for item in items):
                items = [item for item in items if item["distance"] == 0]
            weights = [weight_map[item["id"]] / item["distance"]
                       if self.distance_weighted and item["distance"] > 0 else weight_map[item["id"]]
                       for item in items]
            targets = [self.targets[item["id"]] for item in items]
            result[key] = {str(p): quantile(targets, p, weights) for p in probabilities}
        return result

    @classmethod
    def restore(cls, state):
        result = cls(state["neighbors"], state["distance_weighted"])
        training = Frame.restore(state["training"])
        result.fit(training, state["targets"])
        return result


class KMeans:
    kind = "kmeans"

    def __init__(self, clusters=2, max_iterations=100, tolerance=1e-8, seed=0):
        if type(clusters) is not int or clusters < 1:
            raise SchemaError("cluster count must be positive")
        if type(max_iterations) is not int or max_iterations < 1:
            raise SchemaError("iteration limit must be positive")
        self.tolerance = finite(tolerance)
        if self.tolerance < 0 or type(seed) is not int:
            raise SchemaError("invalid clustering tolerance or seed")
        self.clusters = clusters
        self.max_iterations = max_iterations
        self.seed = seed
        self.features = None
        self.centers = None
        self.inertia = None
        self.iterations = None

    def _assign(self, values, centers):
        labels = []
        distances = []
        for row in values:
            options = [la.squared_distance(row, center) for center in centers]
            label = min(range(len(options)), key=lambda i: (options[i], i))
            labels.append(label)
            distances.append(options[label])
        return labels, distances

    def fit(self, frame):
        frame = frame.complete().positive_population()
        if frame.width == 0 or len(frame) < self.clusters:
            raise SchemaError("insufficient positive samples or features for clustering")
        unique = {}
        for key, row in zip(frame.ids, frame.values):
            unique.setdefault(row, key)
        if len(unique) < self.clusters:
            raise NumericalError("not enough distinct points for requested clusters")
        candidates = sorted(unique, key=lambda row: unique[row])
        first = random.Random(self.seed).randrange(len(candidates))
        centers = [candidates.pop(first)]
        while len(centers) < self.clusters:
            distances = [min(la.squared_distance(row, c) for c in centers) for row in candidates]
            position = max(range(len(candidates)), key=lambda i: (distances[i], unique[candidates[i]]))
            centers.append(candidates.pop(position))
        for iteration in range(1, self.max_iterations + 1):
            labels, distances = self._assign(frame.values, centers)
            updated = []
            for cluster in range(self.clusters):
                indexes = [i for i, label in enumerate(labels) if label == cluster]
                if not indexes:
                    raise NumericalError("clustering produced an empty cluster")
                weights = [frame.weights[i] for i in indexes]
                updated.append(tuple(mean([frame.values[i][j] for i in indexes], weights)
                                     for j in range(frame.width)))
            movement = max(la.squared_distance(a, b) for a, b in zip(centers, updated))
            centers = updated
            if movement <= self.tolerance * self.tolerance:
                break
        labels, distances = self._assign(frame.values, centers)
        if len(set(labels)) != self.clusters:
            raise NumericalError("final clustering has an empty cluster")
        self.features = frame.features
        self.centers = tuple(centers)
        self.inertia = math.fsum(w * d for w, d in zip(frame.weights, distances))
        self.iterations = iteration
        return self

    def predict(self, frame):
        if self.features is None:
            raise NotFitted("clustering model has not been fitted")
        frame = frame.align_features(self.features).complete()
        labels, distances = self._assign(frame.values, self.centers)
        return {key: {"cluster": label, "squared_distance": distance}
                for key, label, distance in zip(frame.ids, labels, distances)}

    def state(self):
        if self.features is None:
            raise NotFitted("cannot serialize unfitted clustering")
        return {"kind": self.kind, "clusters": self.clusters, "max_iterations": self.max_iterations,
                "tolerance": self.tolerance, "seed": self.seed, "features": [f.state() for f in self.features],
                "centers": [list(row) for row in self.centers], "inertia": self.inertia,
                "iterations": self.iterations}

    @classmethod
    def restore(cls, state):
        result = cls(state["clusters"], state["max_iterations"], state["tolerance"], state["seed"])
        result.features = tuple(Feature(**f) for f in state["features"])
        result.centers = la.matrix(state["centers"])
        if la.shape(result.centers) != (result.clusters, len(result.features)):
            raise StateError("cluster center shape mismatch")
        result.inertia = finite(state["inertia"])
        result.iterations = state["iterations"]
        if result.inertia < 0 or type(result.iterations) is not int or not 1 <= result.iterations <= result.max_iterations:
            raise StateError("invalid clustering diagnostics")
        return result


class Logistic(Regressor):
    kind = "logistic"

    def __init__(self, alpha=1.0, intercept=True, max_iterations=100, tolerance=1e-8):
        super().__init__()
        self.alpha = finite(alpha)
        self.tolerance = finite(tolerance)
        if self.alpha < 0 or self.tolerance <= 0:
            raise SchemaError("invalid logistic penalty or tolerance")
        if type(intercept) is not bool or type(max_iterations) is not int or max_iterations < 1:
            raise SchemaError("invalid logistic optimization configuration")
        self.intercept = intercept
        self.max_iterations = max_iterations
        self.coefficients = None
        self.offset = None
        self.iterations = None
        self.objective = None
        self.training_weight = None

    @staticmethod
    def sigmoid(value):
        if value >= 0:
            return 1 / (1 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1 + exponential)

    def _objective(self, values, target, weights, coefficients):
        logits = la.matvec(values, coefficients)
        loss = math.fsum(w * (max(z, 0) + math.log1p(math.exp(-abs(z))) - y * z)
                         for z, y, w in zip(logits, target, weights))
        start = 1 if self.intercept else 0
        return loss + self.alpha * math.fsum(x * x for x in coefficients[start:]) / 2

    def fit(self, frame, targets):
        target = self._training(frame, targets)
        if any(value not in (0, 1) for value in target):
            raise SchemaError("logistic targets must be zero or one")
        observed = {value for value, weight in zip(target, frame.weights) if weight > 0}
        if observed != {0, 1}:
            raise NumericalError("logistic fitting needs both labels with positive support")
        values = [[1.0, *row] if self.intercept else list(row) for row in frame.values]
        width = len(values[0])
        coefficients = [0.0] * width
        total = frame.total_weight()
        objective = self._objective(values, target, frame.weights, coefficients)
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            probabilities = [self.sigmoid(z) for z in la.matvec(values, coefficients)]
            gradient = [math.fsum(w * (p - y) * row[j]
                        for row, p, y, w in zip(values, probabilities, target, frame.weights)) for j in range(width)]
            curvature = [w * p * (1 - p) for p, w in zip(probabilities, frame.weights)]
            hessian = la.weighted_gram(values, curvature)
            start = 1 if self.intercept else 0
            for j in range(start, width):
                gradient[j] += self.alpha * coefficients[j]
                hessian[j][j] += self.alpha
            if la.norm(gradient) <= self.tolerance * (1 + total):
                converged = True
                break
            direction = la.solve(hessian, gradient)
            descent = la.dot(gradient, direction)
            if descent <= 0:
                raise NumericalError("logistic Newton direction is not descending")
            step = 1.0
            accepted = False
            for _ in range(60):
                candidate = [value - step * delta for value, delta in zip(coefficients, direction)]
                new_objective = self._objective(values, target, frame.weights, candidate)
                if new_objective <= objective - 1e-4 * step * descent:
                    coefficients = candidate
                    objective = new_objective
                    accepted = True
                    break
                step *= .5
            if not accepted:
                raise NumericalError("logistic line search did not converge")
        if not converged:
            raise NumericalError("logistic fitting exceeded iteration budget")
        self.features = frame.features
        self.offset = coefficients[0] if self.intercept else 0.0
        self.coefficients = tuple(coefficients[1:] if self.intercept else coefficients)
        self.iterations = iteration
        self.objective = objective
        self.training_weight = total
        return self

    def decision_function(self, frame):
        frame = self._prediction(frame)
        return {key: self.offset + la.dot(row, self.coefficients) for key, row in zip(frame.ids, frame.values)}

    def predict(self, frame):
        return {key: self.sigmoid(value) for key, value in self.decision_function(frame).items()}

    def predict_labels(self, frame, threshold=.5):
        threshold = finite(threshold)
        if not 0 <= threshold <= 1:
            raise SchemaError("decision threshold outside [0,1]")
        return {key: int(value >= threshold) for key, value in self.predict(frame).items()}

    def state(self):
        if self.features is None:
            raise NotFitted("cannot serialize unfitted logistic model")
        return {"kind": self.kind, "alpha": self.alpha, "intercept": self.intercept,
                "max_iterations": self.max_iterations, "tolerance": self.tolerance,
                "features": [f.state() for f in self.features], "coefficients": list(self.coefficients),
                "offset": self.offset, "iterations": self.iterations, "objective": self.objective,
                "training_weight": self.training_weight}

    @classmethod
    def restore(cls, state):
        result = cls(state["alpha"], state["intercept"], state["max_iterations"], state["tolerance"])
        result.features = tuple(Feature(**f) for f in state["features"])
        result.coefficients = tuple(finite(x) for x in state["coefficients"])
        result.offset = finite(state["offset"])
        result.iterations = state["iterations"]
        result.objective = finite(state["objective"])
        result.training_weight = finite(state["training_weight"])
        if len(result.coefficients) != len(result.features) or not result.features:
            raise StateError("logistic coefficient axis mismatch")
        if type(result.iterations) is not int or not 1 <= result.iterations <= result.max_iterations:
            raise StateError("invalid logistic iteration state")
        if result.objective < 0 or result.training_weight <= 0:
            raise StateError("invalid logistic objective or support")
        return result


MODELS = {c.kind: c for c in (Ridge, Neighbors, KMeans, Logistic)}


def create_model(spec):
    spec = dict(spec)
    try:
        method = MODELS[spec.pop("kind")]
        return method(**spec)
    except (KeyError, TypeError) as error:
        raise SchemaError("invalid model specification") from error


def restore_model(state):
    try:
        method = MODELS[state["kind"]]
    except KeyError as error:
        raise StateError("unsupported stored model") from error
    return method.restore(state)
