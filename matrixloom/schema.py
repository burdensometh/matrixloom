from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math

from .errors import SchemaError


KINDS = {"number", "integer", "category", "boolean", "timestamp"}
ROLES = {"feature", "target", "group", "time", "auxiliary"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finite(value, label="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise SchemaError(f"{label} exceeds floating-point range") from error
    if not math.isfinite(result):
        raise SchemaError(f"{label} must be finite")
    return result


def timestamp(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise SchemaError("invalid ISO timestamp") from error
    else:
        raise SchemaError("timestamp must be ISO text or datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaError("timestamp requires an explicit timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Field:
    id: str
    name: str
    kind: str = "number"
    nullable: bool = True
    role: str = "feature"
    unit: str | None = None
    categories: tuple = ()

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise SchemaError("field id must be nonempty text")
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError("field name must be nonempty text")
        if self.kind not in KINDS or self.role not in ROLES:
            raise SchemaError("unsupported field kind or role")
        if type(self.nullable) is not bool:
            raise SchemaError("nullable must be boolean")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise SchemaError("unit must be nonempty text")
        object.__setattr__(self, "categories", tuple(self.categories))
        if any(not isinstance(x, str) for x in self.categories):
            raise SchemaError("category vocabulary must contain text")
        if len(set(self.categories)) != len(self.categories):
            raise SchemaError("duplicate category vocabulary entries")
        if self.categories and self.kind != "category":
            raise SchemaError("only category fields have a vocabulary")

    def parse(self, value):
        if value is None:
            if not self.nullable:
                raise SchemaError(f"{self.id} does not allow missing values")
            return None
        if self.kind == "number":
            return finite(value, self.id)
        if self.kind == "integer":
            if type(value) is int:
                return value
            result = finite(value, self.id)
            if not result.is_integer():
                raise SchemaError(f"{self.id} requires an integer")
            return int(result)
        if self.kind == "boolean":
            if type(value) is not bool:
                raise SchemaError(f"{self.id} requires a boolean")
            return value
        if self.kind == "timestamp":
            return timestamp(value)
        if not isinstance(value, str):
            raise SchemaError(f"{self.id} requires category text")
        if self.categories and value not in self.categories:
            raise SchemaError(f"{self.id} has an undeclared category")
        return value

    def parse_text(self, text, missing_tokens=("",)):
        if text in missing_tokens:
            return self.parse(None)
        if self.kind in {"number", "integer"}:
            try:
                value = Decimal(text)
            except (InvalidOperation, TypeError) as error:
                raise SchemaError(f"{self.id} contains invalid numeric text") from error
            if not value.is_finite():
                raise SchemaError(f"{self.id} contains nonfinite numeric text")
            if self.kind == "integer":
                if value != value.to_integral_value():
                    raise SchemaError(f"{self.id} requires integer text")
                return int(value)
            return self.parse(float(value))
        if self.kind == "boolean":
            if text not in {"true", "false"}:
                raise SchemaError(f"{self.id} requires true or false")
            return text == "true"
        return self.parse(text)

    def renamed(self, name):
        return replace(self, name=name)

    def state(self):
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "nullable": self.nullable, "role": self.role, "unit": self.unit,
                "categories": list(self.categories)}

    @classmethod
    def restore(cls, state):
        return cls(**state)


class Schema:
    def __init__(self, fields):
        self.fields = tuple(fields)
        if not self.fields:
            raise SchemaError("schema needs at least one field")
        if not all(isinstance(f, Field) for f in self.fields):
            raise SchemaError("schema entries must be Fields")
        self._ids = {f.id: i for i, f in enumerate(self.fields)}
        self._names = {f.name: i for i, f in enumerate(self.fields)}
        if len(self._ids) != len(self.fields) or len(self._names) != len(self.fields):
            raise SchemaError("field ids and display names must each be unique")

    def index(self, field_id):
        try:
            return self._ids[field_id]
        except KeyError as error:
            raise SchemaError(f"unknown field id {field_id}") from error

    def field(self, field_id):
        return self.fields[self.index(field_id)]

    def ids(self, role=None):
        return tuple(f.id for f in self.fields if role is None or f.role == role)

    def select(self, ids):
        return Schema(self.field(key) for key in ids)

    def without(self, ids):
        excluded = set(ids)
        for key in excluded:
            self.index(key)
        return Schema(f for f in self.fields if f.id not in excluded)

    def rename(self, names):
        for key in names:
            self.index(key)
        return Schema(f.renamed(names.get(f.id, f.name)) for f in self.fields)

    def validate_row(self, values):
        if len(values) != len(self.fields):
            raise SchemaError("row width differs from schema")
        return tuple(f.parse(value) for f, value in zip(self.fields, values))

    def parse_record(self, record, by_name=False):
        keys = tuple(f.name if by_name else f.id for f in self.fields)
        unknown = set(record) - set(keys)
        if unknown:
            raise SchemaError(f"undeclared fields: {sorted(unknown)}")
        return self.validate_row([record.get(key) for key in keys])

    def compatible(self, other, allow_reorder=True):
        if not isinstance(other, Schema):
            raise SchemaError("expected a Schema")
        if set(self.ids()) != set(other.ids()):
            raise SchemaError("schema field identities differ")
        if not allow_reorder and self.ids() != other.ids():
            raise SchemaError("schema field order differs")
        for field in self.fields:
            peer = other.field(field.id)
            if (field.kind, field.unit, field.role) != (peer.kind, peer.unit, peer.role):
                raise SchemaError(f"incompatible field semantics for {field.id}")
        return True

    def state(self):
        return [field.state() for field in self.fields]

    def digest(self):
        return fingerprint(self.state())

    @classmethod
    def restore(cls, state):
        return cls(Field.restore(item) for item in state)
