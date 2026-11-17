from dataclasses import replace

from .errors import SchemaError
from .schema import Schema, finite
from .table import Sample, Table


UNITS = {
    "m": ("length", 1.0, 0.0), "cm": ("length", .01, 0.0), "mm": ("length", .001, 0.0),
    "s": ("duration", 1.0, 0.0), "ms": ("duration", .001, 0.0), "min": ("duration", 60.0, 0.0),
    "kg": ("mass", 1.0, 0.0), "g": ("mass", .001, 0.0),
    "K": ("temperature", 1.0, 0.0), "degC": ("temperature", 1.0, 273.15),
    "1": ("dimensionless", 1.0, 0.0), "%": ("dimensionless", .01, 0.0),
}


def conversion(source, target):
    if source not in UNITS or target not in UNITS:
        raise SchemaError("unsupported measurement unit")
    a, scale_a, offset_a = UNITS[source]
    b, scale_b, offset_b = UNITS[target]
    if a != b:
        raise SchemaError("incompatible unit dimensions")
    return scale_a / scale_b, (offset_a - offset_b) / scale_b


def convert_values(values, source, target):
    scale, offset = conversion(source, target)
    return tuple(None if value is None else finite(value) * scale + offset for value in values)


def normalize(table, targets):
    fields = list(table.schema.fields)
    columns = {}
    for key, unit in targets.items():
        field = table.schema.field(key)
        if field.kind not in {"number", "integer"} or field.unit is None:
            raise SchemaError("unit conversion requires a declared numeric unit")
        columns[key] = convert_values(table.column(key), field.unit, unit)
        fields[table.schema.index(key)] = replace(field, kind="number", unit=unit)
    values = []
    for i, sample in enumerate(table.samples):
        row = [columns[field.id][i] if field.id in columns else sample.values[j]
               for j, field in enumerate(table.schema.fields)]
        values.append(Sample(sample.id, tuple(row), sample.weight))
    return Table(Schema(fields), values)
