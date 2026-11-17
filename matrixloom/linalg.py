import math

from .errors import NumericalError


def shape(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if any(len(row) != cols for row in matrix):
        raise NumericalError("ragged matrix")
    return rows, cols


def vector(values):
    result = tuple(float(x) for x in values)
    if any(not math.isfinite(x) for x in result):
        raise NumericalError("nonfinite vector")
    return result


def matrix(values):
    result = tuple(vector(row) for row in values)
    shape(result)
    return result


def zeros(rows, cols):
    if rows < 0 or cols < 0:
        raise NumericalError("negative shape")
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(size):
    result = zeros(size, size)
    for i in range(size):
        result[i][i] = 1.0
    return result


def transpose(values):
    rows, cols = shape(values)
    return [[values[i][j] for i in range(rows)] for j in range(cols)]


def dot(left, right):
    if len(left) != len(right):
        raise NumericalError("dot dimensions differ")
    result = math.fsum(a * b for a, b in zip(left, right))
    if not math.isfinite(result):
        raise NumericalError("dot product overflow")
    return result


def norm(values):
    return math.hypot(*values)


def add(left, right):
    if len(left) != len(right):
        raise NumericalError("vector dimensions differ")
    return [a + b for a, b in zip(left, right)]


def subtract(left, right):
    if len(left) != len(right):
        raise NumericalError("vector dimensions differ")
    return [a - b for a, b in zip(left, right)]


def scale(values, factor):
    return [x * factor for x in values]


def outer(left, right):
    return [[a * b for b in right] for a in left]


def matvec(values, coefficients):
    rows, cols = shape(values)
    if cols != len(coefficients):
        raise NumericalError("matrix/vector dimensions differ")
    return [dot(row, coefficients) for row in values]


def matmul(left, right):
    m, inner = shape(left)
    rows, n = shape(right)
    if inner != rows:
        raise NumericalError("matrix product dimensions differ")
    columns = transpose(right)
    return [[dot(row, column) for column in columns] for row in left]


def diagonal(values):
    result = zeros(len(values), len(values))
    for i, value in enumerate(values):
        result[i][i] = value
    return result


def trace(values):
    n, m = shape(values)
    if n != m:
        raise NumericalError("trace requires a square matrix")
    return math.fsum(values[i][i] for i in range(n))


def symmetrize(values):
    n, m = shape(values)
    if n != m:
        raise NumericalError("symmetrization requires a square matrix")
    return [[(values[i][j] + values[j][i]) / 2 for j in range(n)] for i in range(n)]


def solve(values, rhs, tolerance=1e-12):
    """Scaled partial-pivot elimination; singular systems are explicitly rejected."""
    values = matrix(values)
    n, m = shape(values)
    rhs = vector(rhs)
    if n != m or len(rhs) != n or n == 0:
        raise NumericalError("linear solve requires a nonempty square system")
    if tolerance <= 0:
        raise NumericalError("pivot tolerance must be positive")
    work = [list(row) + [value] for row, value in zip(values, rhs)]
    scales = [max(abs(x) for x in row) for row in values]
    if any(x == 0 for x in scales):
        raise NumericalError("singular linear system")
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(work[i][k]) / scales[i])
        if abs(work[pivot][k]) <= tolerance * scales[pivot]:
            raise NumericalError("singular or ill-conditioned linear system")
        work[k], work[pivot] = work[pivot], work[k]
        scales[k], scales[pivot] = scales[pivot], scales[k]
        for i in range(k + 1, n):
            factor = work[i][k] / work[k][k]
            work[i][k] = 0.0
            for j in range(k + 1, n + 1):
                work[i][j] -= factor * work[k][j]
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        result[i] = (work[i][n] - math.fsum(work[i][j] * result[j] for j in range(i + 1, n))) / work[i][i]
    if any(not math.isfinite(x) for x in result):
        raise NumericalError("nonfinite solution")
    return result


def inverse(values, tolerance=1e-12):
    n, m = shape(values)
    if n != m or n == 0:
        raise NumericalError("inverse requires a nonempty square matrix")
    basis = identity(n)
    return transpose([solve(values, row, tolerance) for row in basis])


def cholesky(values, tolerance=1e-12):
    values = matrix(values)
    n, m = shape(values)
    if n != m or n == 0:
        raise NumericalError("Cholesky requires a square matrix")
    bound = max(max(abs(x) for x in row) for row in values)
    for i in range(n):
        for j in range(i):
            if abs(values[i][j] - values[j][i]) > tolerance * max(1.0, bound):
                raise NumericalError("matrix is not symmetric")
    lower = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            remainder = values[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if remainder <= tolerance * max(bound, 1e-300):
                    raise NumericalError("matrix is not positive definite")
                lower[i][j] = math.sqrt(remainder)
            else:
                lower[i][j] = remainder / lower[j][j]
    return lower


def squared_distance(left, right):
    return dot(subtract(left, right), subtract(left, right))


def weighted_gram(values, weights):
    n, p = shape(values)
    if len(weights) != n or any(w < 0 or not math.isfinite(w) for w in weights):
        raise NumericalError("invalid Gram weights")
    columns = transpose(values)
    result = zeros(p, p)
    for i in range(p):
        for j in range(i, p):
            value = math.fsum(w * x * y for w, x, y in zip(weights, columns[i], columns[j]))
            result[i][j] = result[j][i] = value
    return result


def weighted_rhs(values, target, weights):
    n, p = shape(values)
    if len(target) != n or len(weights) != n:
        raise NumericalError("weighted system populations differ")
    return [math.fsum(w * row[j] * y for row, w, y in zip(values, weights, target)) for j in range(p)]


def residuals(values, target, coefficients):
    return subtract(target, matvec(values, coefficients))


def quadratic(values, coefficients):
    return dot(coefficients, matvec(values, coefficients))


def condition_indicator(values):
    """Infinity-norm condition number; intended for the small fitted systems."""
    inv = inverse(values)
    left = max(math.fsum(abs(x) for x in row) for row in values)
    right = max(math.fsum(abs(x) for x in row) for row in inv)
    return left * right
