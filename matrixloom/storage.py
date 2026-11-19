from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

from .errors import IdentityError, StateError
from .schema import Schema, canonical, fingerprint
from .statistics import CovarianceAccumulator
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
CREATE TABLE IF NOT EXISTS covariance_states (
  name TEXT NOT NULL,
  revision INTEGER NOT NULL,
  payload TEXT NOT NULL,
  digest TEXT NOT NULL,
  dataset_name TEXT,
  dataset_revision INTEGER,
  created_at TEXT NOT NULL,
  PRIMARY KEY(name,revision),
  FOREIGN KEY(dataset_name,dataset_revision) REFERENCES datasets(name,revision)
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
        if table not in {"datasets", "bundles", "covariance_states"}:
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

    def save_covariance(self, key, accumulator, dataset=None, expected_revision=None):
        name(key)
        if not isinstance(accumulator, CovarianceAccumulator):
            raise StateError("covariance persistence requires a CovarianceAccumulator")
        state = accumulator.state()
        payload = canonical(state)
        digest = fingerprint(state)
        dataset_name, dataset_revision = (None, None) if dataset is None else dataset
        with self.transaction() as connection:
            if dataset_name is not None:
                info = self._dataset(connection, dataset_name, dataset_revision)
                if accumulator.training_digest != info["digest"]:
                    raise StateError("covariance state differs from bound dataset revision")
                dataset_revision = info["revision"]
            revision = self._revision(connection, "covariance_states", key, expected_revision)
            connection.execute("INSERT INTO covariance_states VALUES(?,?,?,?,?,?,?)",
                               (key, revision, payload, digest, dataset_name, dataset_revision, now()))
        return {"name": key, "revision": revision, "digest": digest,
                "features": state["features"], "pairs": len(state["pairs"]), "ddof": state["ddof"]}

    def _covariance(self, connection, key, revision):
        name(key)
        if revision is None:
            revision = self._latest(connection, "covariance_states", key)
        row = connection.execute("SELECT * FROM covariance_states WHERE name=? AND revision=?",
                                 (key, revision)).fetchone()
        if row is None:
            raise IdentityError("covariance state revision does not exist")
        state = json.loads(row["payload"])
        if fingerprint(state) != row["digest"]:
            raise StateError("stored covariance integrity check failed")
        return row, state

    def load_covariance(self, key, revision=None):
        connection = self.connect()
        try:
            _, state = self._covariance(connection, key, revision)
            return CovarianceAccumulator.restore(state)
        finally:
            connection.close()

    def covariance_info(self, key, revision=None):
        connection = self.connect()
        try:
            row, state = self._covariance(connection, key, revision)
            return {"name": row["name"], "revision": row["revision"], "digest": row["digest"],
                    "dataset_name": row["dataset_name"], "dataset_revision": row["dataset_revision"],
                    "features": [feature["id"] for feature in state["features"]],
                    "ddof": state["ddof"], "created_at": row["created_at"]}
        finally:
            connection.close()

    def revisions(self, key, kind="dataset"):
        table = {"dataset": "datasets", "bundle": "bundles", "covariance": "covariance_states"}.get(kind)
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
