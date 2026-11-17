class MatrixError(ValueError):
    """Invalid data or incompatible numerical state."""


class SchemaError(MatrixError):
    pass


class IdentityError(MatrixError):
    pass


class NumericalError(MatrixError):
    pass


class NotFitted(MatrixError):
    pass


class StateError(MatrixError):
    pass
