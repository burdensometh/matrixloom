import csv
import io
import json
from dataclasses import dataclass

from .errors import IdentityError, SchemaError
from .schema import Field, Schema, canonical, fingerprint, finite


@dataclass(frozen=True)
class Sample:
    id: str
    values: tuple
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise IdentityError("sample id must be nonempty text")
        object.__setattr__(self, "values", tuple(self.values))
        weight = finite(self.weight, "sample weight")
        if weight < 0:
            raise SchemaError("sample weight cannot be negative")
        object.__setattr__(self, "weight", weight)


class Table:
    def __init__(self, schema, samples):
        if not isinstance(schema, Schema):
            raise SchemaError("Table requires a Schema")
        self.schema = schema
        checked = []
        seen = set()
        for sample in samples:
            if not isinstance(sample, Sample):
                raise SchemaError("table entries must be Samples")
            if sample.id in seen:
                raise IdentityError(f"duplicate sample id {sample.id}")
            seen.add(sample.id)
            values = schema.validate_row(sample.values)
            checked.append(Sample(sample.id, values, sample.weight))
        self.samples = tuple(checked)
        self._positions = {sample.id: i for i, sample in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def ids(self):
        return tuple(sample.id for sample in self.samples)

    def weights(self):
        return tuple(sample.weight for sample in self.samples)

    def sample(self, sample_id):
        try:
            return self.samples[self._positions[sample_id]]
        except KeyError as error:
            raise IdentityError(f"unknown sample id {sample_id}") from error

    def column(self, field_id):
        position = self.schema.index(field_id)
        return tuple(sample.values[position] for sample in self.samples)

    def records(self, include_weight=False):
        result = []
        for sample in self.samples:
            record = dict(zip(self.schema.ids(), sample.values))
            record["_sample_id"] = sample.id
            if include_weight:
                record["_weight"] = sample.weight
            result.append(record)
        return result

    def take(self, sample_ids):
        return Table(self.schema, (self.sample(key) for key in sample_ids))

    def select(self, field_ids):
        ids = tuple(field_ids)
        positions = [self.schema.index(key) for key in ids]
        schema = self.schema.select(ids)
        return Table(schema, [Sample(sample.id, tuple(sample.values[i] for i in positions), sample.weight)
                              for sample in self.samples])

    def filter(self, field_id, predicate):
        position = self.schema.index(field_id)
        return Table(self.schema, [sample for sample in self.samples if predicate(sample.values[position])])

    def drop_missing(self, field_ids):
        positions = [self.schema.index(key) for key in field_ids]
        return Table(self.schema, [sample for sample in self.samples
                                  if all(sample.values[i] is not None for i in positions)])

    def rename(self, names):
        return Table(self.schema.rename(names), self.samples)

    def append(self, other):
        self.schema.compatible(other.schema)
        ordered = other.select(self.schema.ids())
        return Table(self.schema, self.samples + ordered.samples)

    def with_column(self, field, values):
        if field.id in self.schema.ids():
            raise SchemaError("new column identity already exists")
        values = tuple(values)
        if len(values) != len(self):
            raise IdentityError("new column population differs")
        schema = Schema(self.schema.fields + (field,))
        return Table(schema, [Sample(sample.id, sample.values + (value,), sample.weight)
                              for sample, value in zip(self.samples, values)])

    def replace_column(self, field_id, values):
        position = self.schema.index(field_id)
        values = tuple(values)
        if len(values) != len(self):
            raise IdentityError("replacement column population differs")
        samples = []
        for sample, value in zip(self.samples, values):
            row = list(sample.values)
            row[position] = value
            samples.append(Sample(sample.id, tuple(row), sample.weight))
        return Table(self.schema, samples)

    def join_lookup(self, other, left_key, right_key, fields, prefix="lookup/"):
        left_index = self.schema.index(left_key)
        right_index = other.schema.index(right_key)
        lookup = {}
        for sample in other.samples:
            key = sample.values[right_index]
            if key is None:
                raise IdentityError("lookup key cannot be missing")
            if key in lookup:
                raise IdentityError("lookup key must be unique")
            lookup[key] = sample
        selected = tuple(other.schema.field(key) for key in fields)
        additions = tuple(Field(prefix + f.id, prefix + f.name, f.kind, True, f.role, f.unit, f.categories)
                          for f in selected)
        schema = Schema(self.schema.fields + additions)
        positions = [other.schema.index(f.id) for f in selected]
        result = []
        for sample in self.samples:
            peer = lookup.get(sample.values[left_index])
            extra = tuple(peer.values[i] if peer else None for i in positions)
            result.append(Sample(sample.id, sample.values + extra, sample.weight))
        return Table(schema, result)

    def digest(self):
        return fingerprint(self.state())

    def state(self):
        return {"schema": self.schema.state(), "samples": [
            {"id": s.id, "values": list(s.values), "weight": s.weight} for s in self.samples]}

    @classmethod
    def restore(cls, state):
        return cls(Schema.restore(state["schema"]), [Sample(s["id"], tuple(s["values"]), s["weight"])
                                                   for s in state["samples"]])

    @classmethod
    def from_records(cls, schema, records, id_key="_sample_id", weight_key="_weight"):
        samples = []
        for record in records:
            record = dict(record)
            if id_key not in record:
                raise IdentityError("record lacks sample identity")
            sample_id = record.pop(id_key)
            weight = record.pop(weight_key, 1.0)
            samples.append(Sample(sample_id, schema.parse_record(record), weight))
        return cls(schema, samples)

    def to_jsonl(self):
        return "".join(canonical(record) + "\n" for record in self.records(True))

    @classmethod
    def from_jsonl(cls, schema, text):
        records = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as error:
                raise SchemaError(f"invalid JSON record at line {number}") from error
            if not isinstance(value, dict):
                raise SchemaError(f"line {number} must contain an object")
            records.append(value)
        return cls.from_records(schema, records)

    def to_csv(self):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["_sample_id", "_weight", *self.schema.ids()])
        for sample in self.samples:
            row = []
            for field, value in zip(self.schema.fields, sample.values):
                if value is None:
                    row.append("\\N")
                elif field.kind == "boolean":
                    row.append("true" if value else "false")
                else:
                    text = str(value)
                    row.append("\\" + text if text.startswith("\\") else text)
            writer.writerow([sample.id, sample.weight, *row])
        return stream.getvalue()

    @classmethod
    def from_csv(cls, schema, text):
        reader = csv.DictReader(io.StringIO(text, newline=""))
        expected = {"_sample_id", "_weight", *schema.ids()}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise SchemaError("CSV header does not match declared identities")
        if len(reader.fieldnames) != len(expected):
            raise SchemaError("duplicate CSV headers")
        samples = []
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise SchemaError("CSV row width mismatch")
            try:
                weight = float(record["_weight"])
            except ValueError as error:
                raise SchemaError("invalid sample weight") from error
            values = []
            for field in schema.fields:
                text = record[field.id]
                if text == "\\N":
                    values.append(field.parse(None))
                else:
                    if text.startswith("\\\\"):
                        text = text[1:]
                    values.append(field.parse_text(text, missing_tokens=()))
            samples.append(Sample(record["_sample_id"], tuple(values), weight))
        return cls(schema, samples)
