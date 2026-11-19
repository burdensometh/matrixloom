from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

from .errors import IdentityError, StateError
from .schema import Schema, canonical, fingerprint
from .table import Sample, Table
from .pipeline import restore_bundle


DDL = """
CREATE TABLE IF NOT EXISTS datasets (
  name TEXT NOT NULL,
  revision INTEGER NOT NULL,
  schema_json TEXT NOT NULL,
  digest TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(name,revision)
);
CREATE TABLE IF NOT EXISTS samples (
  dataset_name TEXT NOT NULL,
  revision INTEGER NOT NULL,
  position INTEGER NOT NULL,
  sample_id TEXT NOT NULL,
  weight REAL NOT NULL,
  values_json TEXT NOT NULL,
  PRIMARY KEY(dataset_name,revision,sample_id),
  UNIQUE(dataset_name,revision,position),
  FOREIGN KEY(dataset_name,revision) REFERENCES datasets(name,revision)
);
CREATE TABLE IF NOT EXISTS bundles (
  name TEXT NOT NULL,
  revision INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  digest TEXT NOT NULL,
  dataset_name TEXT,
  dataset_revision INTEGER,
  created_at TEXT NOT NULL,
  PRIMARY KEY(name,revision),
  FOREIGN KEY(dataset_name,dataset_revision) REFERENCES datasets(name,revision)
);
CREATE TABLE IF NOT EXISTS evaluations (
  run_id TEXT PRIMARY KEY,
  bundle_name TEXT NOT NULL,
  bundle_revision INTEGER NOT NULL,
  dataset_name TEXT NOT NULL,
  dataset_revision INTEGER NOT NULL,
  population_digest TEXT NOT NULL,
  predictions_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  digest TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(bundle_name,bundle_revision) REFERENCES bundles(name,revision),
  FOREIGN KEY(dataset_name,dataset_revision) REFERENCES datasets(name,revision)
);
CREATE TABLE IF NOT EXISTS comparisons (
  comparison_id TEXT PRIMARY KEY,
  dataset_name TEXT NOT NULL,
  dataset_revision INTEGER NOT NULL,
  left_name TEXT NOT NULL,
  left_revision INTEGER NOT NULL,
  right_name TEXT NOT NULL,
  right_revision INTEGER NOT NULL,
  metric TEXT NOT NULL,
  group_field TEXT NOT NULL,
  population_digest TEXT NOT NULL,
  requests_json TEXT NOT NULL,
  result_json TEXT NOT NULL,
  digest TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(dataset_name,dataset_revision) REFERENCES datasets(name,revision),
  FOREIGN KEY(left_name,left_revision) REFERENCES bundles(name,revision),
  FOREIGN KEY(right_name,right_revision) REFERENCES bundles(name,revision)
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise IdentityError("storage name must be 1..200 nonblank characters")
    return value


class Store:
    """Immutable revisions of actual sample data, fitted numerical states, and evaluations."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        try:
            connection.executescript(DDL)
        finally:
            connection.close()

    def connect(self):
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _latest(self, connection, table, key):
        if table not in {"datasets", "bundles"}:
            raise StateError("invalid revision namespace")
        row = connection.execute(f"SELECT max(revision) AS revision FROM {table} WHERE name=?", (key,)).fetchone()
        return row["revision"] or 0

    def _revision(self, connection, table, key, expected):
        latest = self._latest(connection, table, key)
        if expected is not None and (type(expected) is not int or expected != latest):
            raise StateError("revision precondition no longer holds")
        return latest + 1

    def save_table(self, key, table, expected_revision=None):
        name(key)
        digest = table.digest()
        schema = canonical(table.schema.state())
        with self.transaction() as connection:
            revision = self._revision(connection, "datasets", key, expected_revision)
            connection.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?)",
                               (key, revision, schema, digest, len(table), now()))
            connection.executemany("INSERT INTO samples VALUES(?,?,?,?,?,?)",
                [(key, revision, i, sample.id, sample.weight, canonical(sample.values))
                 for i, sample in enumerate(table.samples)])
        return {"name": key, "revision": revision, "digest": digest, "rows": len(table)}

    def _dataset(self, connection, key, revision):
        name(key)
        if revision is None:
            revision = self._latest(connection, "datasets", key)
        row = connection.execute("SELECT * FROM datasets WHERE name=? AND revision=?", (key, revision)).fetchone()
        if row is None:
            raise IdentityError("dataset revision does not exist")
        return row

    def table_info(self, key, revision=None):
        connection = self.connect()
        try:
            row = self._dataset(connection, key, revision)
            return {"name": row["name"], "revision": row["revision"], "digest": row["digest"],
                    "rows": row["row_count"], "schema": json.loads(row["schema_json"]), "created_at": row["created_at"]}
        finally:
            connection.close()

    def load_table(self, key, revision=None):
        connection = self.connect()
        try:
            info = self._dataset(connection, key, revision)
            rows = connection.execute("SELECT * FROM samples WHERE dataset_name=? AND revision=? ORDER BY position",
                                      (key, info["revision"])).fetchall()
            table = Table(Schema.restore(json.loads(info["schema_json"])),
                          [Sample(row["sample_id"], tuple(json.loads(row["values_json"])), row["weight"]) for row in rows])
            if len(table) != info["row_count"] or table.digest() != info["digest"]:
                raise StateError("stored dataset integrity check failed")
            return table
        finally:
            connection.close()

    def batches(self, key, revision=None, size=256):
        if type(size) is not int or size < 1:
            raise StateError("batch size must be positive")
        connection = self.connect()
        try:
            info = self._dataset(connection, key, revision)
            schema = Schema.restore(json.loads(info["schema_json"]))
            cursor = connection.execute("SELECT * FROM samples WHERE dataset_name=? AND revision=? ORDER BY position",
                                        (key, info["revision"]))
            count = 0
            while True:
                rows = cursor.fetchmany(size)
                if not rows:
                    break
                count += len(rows)
                yield Table(schema, [Sample(row["sample_id"], tuple(json.loads(row["values_json"])), row["weight"])
                                     for row in rows])
            if count != info["row_count"]:
                raise StateError("stored dataset row count mismatch")
        finally:
            connection.close()

    def save_bundle(self, key, bundle, dataset=None, expected_revision=None):
        name(key)
        state = bundle.state()
        payload = canonical(state)
        digest = fingerprint(state)
        dataset_name, dataset_revision = (None, None) if dataset is None else dataset
        with self.transaction() as connection:
            if dataset_name is not None:
                info = self._dataset(connection, dataset_name, dataset_revision)
                if info["digest"] != state["training_digest"]:
                    raise StateError("bundle training data differs from bound dataset revision")
                dataset_revision = info["revision"]
            revision = self._revision(connection, "bundles", key, expected_revision)
            connection.execute("INSERT INTO bundles VALUES(?,?,?,?,?,?,?,?)",
                (key, revision, state["kind"], payload, digest, dataset_name, dataset_revision, now()))
        return {"name": key, "revision": revision, "digest": digest, "kind": state["kind"]}

    def _bundle(self, connection, key, revision):
        name(key)
        if revision is None:
            revision = self._latest(connection, "bundles", key)
        row = connection.execute("SELECT * FROM bundles WHERE name=? AND revision=?", (key, revision)).fetchone()
        if row is None:
            raise IdentityError("model bundle revision does not exist")
        state = json.loads(row["payload"])
        if fingerprint(state) != row["digest"]:
            raise StateError("stored bundle integrity check failed")
        return row, state

    def load_bundle(self, key, revision=None):
        connection = self.connect()
        try:
            _, state = self._bundle(connection, key, revision)
            return restore_bundle(state)
        finally:
            connection.close()

    def bundle_info(self, key, revision=None):
        connection = self.connect()
        try:
            row, state = self._bundle(connection, key, revision)
            return {"name": row["name"], "revision": row["revision"], "digest": row["digest"],
                    "kind": row["kind"], "dataset_name": row["dataset_name"],
                    "dataset_revision": row["dataset_revision"], "format_version": state["version"]}
        finally:
            connection.close()

    def save_evaluation(self, run_id, bundle_ref, dataset_ref, predictions, metrics):
        name(run_id)
        bundle_name, bundle_revision = bundle_ref
        dataset_name, dataset_revision = dataset_ref
        with self.transaction() as connection:
            bundle, _ = self._bundle(connection, bundle_name, bundle_revision)
            dataset = self._dataset(connection, dataset_name, dataset_revision)
            ids = {row[0] for row in connection.execute(
                "SELECT sample_id FROM samples WHERE dataset_name=? AND revision=?", (dataset_name, dataset["revision"]))}
            if ids != set(predictions):
                raise IdentityError("evaluation predictions differ from bound dataset population")
            population_digest = fingerprint(sorted(ids))
            payload = {"bundle": [bundle_name, bundle["revision"]], "dataset": [dataset_name, dataset["revision"]],
                       "population_digest": population_digest, "predictions": predictions, "metrics": metrics}
            digest = fingerprint(payload)
            old = connection.execute("SELECT digest FROM evaluations WHERE run_id=?", (run_id,)).fetchone()
            if old is not None:
                if old["digest"] != digest:
                    raise StateError("evaluation id already binds a different result")
                return {"run_id": run_id, "replayed": True, "digest": digest}
            connection.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, bundle_name, bundle["revision"], dataset_name, dataset["revision"], population_digest,
                 canonical(predictions), canonical(metrics), digest, now()))
        return {"run_id": run_id, "replayed": False, "digest": digest}

    def evaluation(self, run_id):
        connection = self.connect()
        try:
            row = connection.execute("SELECT * FROM evaluations WHERE run_id=?", (name(run_id),)).fetchone()
            if row is None:
                raise IdentityError("evaluation does not exist")
            payload = {"bundle": [row["bundle_name"], row["bundle_revision"]],
                       "dataset": [row["dataset_name"], row["dataset_revision"]],
                       "population_digest": row["population_digest"],
                       "predictions": json.loads(row["predictions_json"]), "metrics": json.loads(row["metrics_json"])}
            if fingerprint(payload) != row["digest"]:
                raise StateError("stored evaluation integrity check failed")
            return {"run_id": run_id, "digest": row["digest"], **payload}
        finally:
            connection.close()

    def save_comparison(self, comparison_id, dataset_ref, left_ref, right_ref, group_field,
                        metric, requests, result, population):
        """Persist a grouped paired comparison with its seed and interval convention.

        Replaying the same identifier with identical request and result payloads is
        idempotent; any drift in the resampling configuration or computed statistics is
        rejected so a stored interval can always be reproduced from its own records.
        """
        name(comparison_id)
        dataset_name, dataset_revision = dataset_ref
        left_name, left_revision = left_ref
        right_name, right_revision = right_ref
        name(group_field)
        population = tuple(population)
        if len(set(population)) != len(population):
            raise IdentityError("comparison population contains duplicate sample identities")
        with self.transaction() as connection:
            dataset = self._dataset(connection, dataset_name, dataset_revision)
            left, left_state = self._bundle(connection, left_name, left_revision)
            right, right_state = self._bundle(connection, right_name, right_revision)
            # Both models must target the same declared field; differing targets would
            # make the paired difference meaningless even if sample identities align.
            if left_state.get("target") != right_state.get("target"):
                raise IdentityError("compared models do not share a target field")
            target_field = left_state["target"]
            dataset_schema = Schema.restore(json.loads(dataset["schema_json"]))
            if target_field not in dataset_schema.ids():
                raise IdentityError("comparison dataset lacks the compared target field")
            target_definition = dataset_schema.field(target_field).state()
            # A model bound to a training dataset pins the target's declared version;
            # a later change of kind, unit, or role must not be silently compared.
            for label, bundle in (("left", left), ("right", right)):
                if bundle["dataset_name"] is None:
                    continue
                training = self._dataset(connection, bundle["dataset_name"], bundle["dataset_revision"])
                training_schema = Schema.restore(json.loads(training["schema_json"]))
                if target_field not in training_schema.ids():
                    raise IdentityError(f"{label} model target is absent from its training dataset")
                if training_schema.field(target_field).state() != target_definition:
                    raise IdentityError(f"{label} model target version differs from the comparison dataset")
            ids = {row[0] for row in connection.execute(
                "SELECT sample_id FROM samples WHERE dataset_name=? AND revision=?",
                (dataset_name, dataset["revision"]))}
            if set(population) != ids:
                raise IdentityError("comparison population differs from the bound dataset")
            population_digest = fingerprint(sorted(ids))
            requests = dict(requests)
            payload = {"dataset": [dataset_name, dataset["revision"]],
                       "left": [left_name, left["revision"]], "right": [right_name, right["revision"]],
                       "metric": metric, "group_field": group_field,
                       "population_digest": population_digest,
                       "requests": requests, "result": result}
            digest = fingerprint(payload)
            old = connection.execute(
                "SELECT digest FROM comparisons WHERE comparison_id=?", (comparison_id,)).fetchone()
            if old is not None:
                if old["digest"] != digest:
                    raise StateError("comparison id already binds a different result")
                return {"comparison_id": comparison_id, "replayed": True, "digest": digest}
            connection.execute("INSERT INTO comparisons VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (comparison_id, dataset_name, dataset["revision"], left_name, left["revision"],
                 right_name, right["revision"], metric, group_field, population_digest,
                 canonical(requests), canonical(result), digest, now()))
        return {"comparison_id": comparison_id, "replayed": False, "digest": digest}

    def comparison(self, comparison_id):
        connection = self.connect()
        try:
            row = connection.execute("SELECT * FROM comparisons WHERE comparison_id=?",
                                     (name(comparison_id),)).fetchone()
            if row is None:
                raise IdentityError("comparison does not exist")
            payload = {"dataset": [row["dataset_name"], row["dataset_revision"]],
                       "left": [row["left_name"], row["left_revision"]],
                       "right": [row["right_name"], row["right_revision"]],
                       "metric": row["metric"], "group_field": row["group_field"],
                       "population_digest": row["population_digest"],
                       "requests": json.loads(row["requests_json"]),
                       "result": json.loads(row["result_json"])}
            if fingerprint(payload) != row["digest"]:
                raise StateError("stored comparison integrity check failed")
            return {"comparison_id": comparison_id, "digest": row["digest"], **payload}
        finally:
            connection.close()

    def export_bundle(self, key, revision=None):
        connection = self.connect()
        try:
            _, state = self._bundle(connection, key, revision)
            return canonical({"format": "matrixloom-bundle", "version": 1,
                              "payload": state, "sha256": fingerprint(state)}) + "\n"
        finally:
            connection.close()

    def import_bundle(self, key, text, expected_revision=None):
        envelope = json.loads(text)
        if envelope.get("format") != "matrixloom-bundle" or envelope.get("version") != 1:
            raise StateError("unsupported bundle envelope")
        if envelope.get("sha256") != fingerprint(envelope["payload"]):
            raise StateError("bundle envelope checksum mismatch")
        return self.save_bundle(key, restore_bundle(envelope["payload"]), expected_revision=expected_revision)

    def revisions(self, key, kind="dataset"):
        table = {"dataset": "datasets", "bundle": "bundles"}.get(kind)
        if table is None:
            raise StateError("invalid revision kind")
        connection = self.connect()
        try:
            rows = connection.execute(f"SELECT name,revision,digest,created_at FROM {table} WHERE name=? ORDER BY revision", (name(key),))
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def select_samples(self, key, sample_ids, revision=None):
        sample_ids = tuple(sample_ids)
        if len(set(sample_ids)) != len(sample_ids):
            raise IdentityError("requested sample identities must be unique")
        connection = self.connect()
        try:
            info = self._dataset(connection, key, revision)
            found = {}
            for start in range(0, len(sample_ids), 128):
                chunk = sample_ids[start:start + 128]
                placeholders = ",".join("?" for _ in chunk)
                query = "SELECT * FROM samples WHERE dataset_name=? AND revision=? AND sample_id IN (" + placeholders + ")"
                for row in connection.execute(query, (key, info["revision"], *chunk)):
                    found[row["sample_id"]] = Sample(row["sample_id"], tuple(json.loads(row["values_json"])), row["weight"])
            if set(found) != set(sample_ids):
                raise IdentityError("requested sample does not exist in the pinned revision")
            return Table(Schema.restore(json.loads(info["schema_json"])), [found[x] for x in sample_ids])
        finally:
            connection.close()
