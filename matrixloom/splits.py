from dataclasses import dataclass
import random

from .errors import IdentityError, SchemaError
from .schema import timestamp


@dataclass(frozen=True)
class Split:
    train: tuple
    test: tuple
    name: str

    def __post_init__(self):
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "test", tuple(self.test))
        if not self.train or not self.test:
            raise IdentityError("both split populations must be nonempty")
        if len(set(self.train)) != len(self.train) or len(set(self.test)) != len(self.test):
            raise IdentityError("duplicate split sample identity")
        if set(self.train) & set(self.test):
            raise IdentityError("training and testing populations overlap")

    def validate(self, table, exhaustive=False):
        selected = set(self.train) | set(self.test)
        if not selected <= set(table.ids()):
            raise IdentityError("split contains unknown samples")
        if exhaustive and selected != set(table.ids()):
            raise IdentityError("split does not exhaust the dataset")
        return True

    def apply(self, table):
        self.validate(table)
        return table.take(self.train), table.take(self.test)

    def state(self):
        return {"train": list(self.train), "test": list(self.test), "name": self.name}

    @classmethod
    def restore(cls, state):
        return cls(state["train"], state["test"], state["name"])


def random_split(table, test_fraction=.2, seed=0):
    if not 0 < test_fraction < 1:
        raise SchemaError("test fraction must lie between zero and one")
    if len(table) < 2:
        raise IdentityError("holdout requires at least two samples")
    ids = sorted(table.ids())
    random.Random(seed).shuffle(ids)
    count = min(len(ids) - 1, max(1, round(len(ids) * test_fraction)))
    return Split(ids[count:], ids[:count], f"random-{seed}")


def kfold(table, folds=5, seed=0):
    if type(folds) is not int or not 2 <= folds <= len(table):
        raise SchemaError("fold count must be between two and sample count")
    ids = sorted(table.ids())
    random.Random(seed).shuffle(ids)
    result = []
    for fold in range(folds):
        test = tuple(ids[fold::folds])
        test_set = set(test)
        train = tuple(key for key in ids if key not in test_set)
        result.append(Split(train, test, f"fold-{fold}"))
    return tuple(result)


def time_split(table, field_id, cutoff):
    field = table.schema.field(field_id)
    if field.kind != "timestamp":
        raise SchemaError("temporal split needs a timestamp field")
    cutoff = timestamp(cutoff)
    values = table.column(field_id)
    if any(value is None for value in values):
        raise SchemaError("temporal split cannot silently place missing times")
    train = [key for key, value in zip(table.ids(), values) if value < cutoff]
    test = [key for key, value in zip(table.ids(), values) if value >= cutoff]
    return Split(train, test, "time-" + cutoff)


def stratified_split(table, field_id, test_fraction=.2, seed=0):
    if not 0 < test_fraction < 1:
        raise SchemaError("test fraction must lie between zero and one")
    groups = {}
    for key, value in zip(table.ids(), table.column(field_id)):
        if value is None:
            raise SchemaError("stratification labels cannot be missing")
        groups.setdefault(value, []).append(key)
    train = []
    test = []
    generator = random.Random(seed)
    for value in sorted(groups, key=str):
        ids = sorted(groups[value])
        if len(ids) < 2:
            raise IdentityError("each stratum needs training and test support")
        generator.shuffle(ids)
        count = min(len(ids) - 1, max(1, round(len(ids) * test_fraction)))
        test.extend(ids[:count])
        train.extend(ids[count:])
    return Split(train, test, "stratified-" + str(seed))
