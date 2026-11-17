import ast
import math
import operator
from datetime import datetime

from .errors import NumericalError, SchemaError
from .schema import Field, finite


BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {"abs", "sqrt", "log", "log1p", "exp", "minimum", "maximum", "coalesce", "seconds_between"}


class Expression:
    """Small numeric expression language. Field references use stable field ids."""

    def __init__(self, text, max_nodes=128):
        if not isinstance(text, str) or len(text) > 4096:
            raise SchemaError("expression must be bounded text")
        self.text = text
        try:
            self.tree = ast.parse(text, mode="eval").body
        except SyntaxError as error:
            raise SchemaError("invalid feature expression") from error
        nodes = list(ast.walk(self.tree))
        if len(nodes) > max_nodes:
            raise SchemaError("expression is too complex")
        self.dependencies = set()
        self._validate(self.tree)

    def _validate(self, node):
        if isinstance(node, ast.Constant):
            if node.value is not None:
                finite(node.value, "expression literal")
            return
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            self._validate(node.left)
            self._validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            self._validate(node.operand)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords:
                raise SchemaError("expression calls do not accept keywords")
            name = node.func.id
            if name == "field":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                    raise SchemaError("field requires one literal id")
                key = node.args[0].value
                if not isinstance(key, str) or not key:
                    raise SchemaError("field requires a nonempty string id")
                self.dependencies.add(key)
                return
            if name not in FUNCTIONS:
                raise SchemaError("unsupported expression function")
            arity = 2 if name in {"minimum", "maximum", "coalesce", "seconds_between"} else 1
            if len(node.args) != arity:
                raise SchemaError("expression function argument count mismatch")
            for child in node.args:
                self._validate(child)
            return
        raise SchemaError("expression syntax is not allowed")

    def _evaluate(self, node, values):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, values)
            return None if value is None else UNARY[type(node.op)](value)
        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left, values)
            right = self._evaluate(node.right, values)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise NumericalError("expression exponent exceeds safe bound")
            return BINARY[type(node.op)](finite(left), finite(right))
        name = node.func.id
        if name == "field":
            return values[node.args[0].value]
        left = self._evaluate(node.args[0], values)
        if name == "coalesce":
            return self._evaluate(node.args[1], values) if left is None else left
        if left is None:
            return None
        if name in {"minimum", "maximum", "seconds_between"}:
            right = self._evaluate(node.args[1], values)
            if right is None:
                return None
            if name == "seconds_between":
                return (datetime.fromisoformat(left) - datetime.fromisoformat(right)).total_seconds()
            return min(left, right) if name == "minimum" else max(left, right)
        value = finite(left)
        methods = {"abs": abs, "sqrt": math.sqrt, "log": math.log,
                   "log1p": math.log1p, "exp": math.exp}
        return methods[name](value)

    def evaluate(self, values):
        if not self.dependencies <= values.keys():
            raise SchemaError("expression field dependency is absent")
        try:
            result = self._evaluate(self.tree, values)
            return None if result is None else finite(result, "expression result")
        except (ArithmeticError, ValueError, TypeError) as error:
            raise NumericalError(f"feature expression failed: {self.text}") from error


class Derivations:
    def __init__(self, definitions):
        self.definitions = tuple(dict(item) for item in definitions)
        self.expressions = {}
        self.fields = {}
        for item in self.definitions:
            field = Field.restore(item["field"])
            if field.id in self.fields:
                raise SchemaError("duplicate derived field identity")
            if field.kind != "number":
                raise SchemaError("derived expressions produce numeric fields")
            self.fields[field.id] = field
            self.expressions[field.id] = Expression(item["expression"])

    def order(self, schema):
        available = set(schema.ids())
        if available & self.fields.keys():
            raise SchemaError("derived identity collides with input")
        pending = list(self.fields)
        result = []
        while pending:
            ready = [key for key in pending if self.expressions[key].dependencies <= available]
            if not ready:
                raise SchemaError("derived features contain a cycle or absent dependency")
            for key in ready:
                available.add(key)
                pending.remove(key)
                result.append(key)
        return tuple(result)

    def transform(self, table):
        for key in self.order(table.schema):
            expression = self.expressions[key]
            values = []
            for sample in table.samples:
                context = dict(zip(table.schema.ids(), sample.values))
                values.append(expression.evaluate(context))
            table = table.with_column(self.fields[key], values)
        return table

    def state(self):
        return [dict(item) for item in self.definitions]

    @classmethod
    def restore(cls, state):
        return cls(state)
