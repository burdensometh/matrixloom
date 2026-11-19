from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

from . import blocks
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
  format_version INTEGER NOT NULL DEFAULT 1,
  block_rows INTEGER,
  published INTEGER NOT NULL DEFAULT 1,
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
CREATE TABLE IF NOT EXISTS sample_blocks (
  dataset_name TEXT NOT NULL,
  revision INTEGER NOT NULL,
  block_index INTEGER NOT NULL,
  payload TEXT NOT NULL,
  digest TEXT NOT NULL,
  PRIMARY KEY(dataset_name,revision,block_index),
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
  format_version INTEGER NOT NULL DEFAULT 1,
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
"""

# Columns added after the version 1 layout.  Opening an older store upgrades
# it in place; version 1 rows keep reading as published single-row inventories.
UPGRADE_COLUMNS = {
    "datasets": {"format_version": "INTEGER NOT NULL DEFAULT 1",
                 "block_rows": "INTEGER",
                 "published": "INTEGER NOT NULL DEFAULT 1"},
    "bundles": {"format_version": "INTEGER NOT NULL DEFAULT 1"},
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise IdentityError("storage name must be 1..200 nonblank characters")
    return value


class Store:
    """Immutable revisions of actual sample data, fitted numerical states, and evaluations.

    New datasets and bundles are written in format version 2 (column blocks for
    samples, column blocks for a neighbor model's embedded training matrix).
    Version 1 single-row inventories remain readable and are never rewritten.
    """

    def __init__(self, path, block_rows=blocks.DEFAULT_BLOCK_ROWS, read_only=False):
        self.path = Path(path)
        self.block_rows = block_rows
        self.read_only = bool(read_only)
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = self.connect()
        except sqlite3.OperationalError as error:
            raise StateError("inventory cannot be opened") from error
        try:
            if not self.read_only:
                connection.executescript(DDL)
                self._upgrade(connection)
        finally:
            connection.close()

    def _upgrade(self, connection):
        for table, columns in UPGRADE_COLUMNS.items():
            existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            for column, declaration in columns.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @classmethod
    def legacy(cls, path):
        """Open an existing inventory without altering it, for version 1 reads."""
        return cls(path, read_only=True)

    def connect(self):
        if self.read_only:
            connection = sqlite3.connect(self._read_uri(), uri=True, isolation_level=None, timeout=15)
        else:
            connection = sqlite3.connect(self.path, isolation_level=None, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _read_uri(self):
        from urllib.parse import quote
        return "file:" + quote(str(self.path.resolve())) + "?mode=ro"

    @staticmethod
    def _value(row, key, default=None):
        return row[key] if key in row.keys() else default

    @staticmethod
    def _columns(connection, table):
        return {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}

    def _published_clause(self, connection):
        # A never-upgraded read-only inventory predates publishing; every row is final.
        return " AND published=1" if "published" in self._columns(connection, "datasets") else ""

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            try:
                connection.rollback()
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            connection.close()

    def _latest(self, connection, table, key):
        if table not in {"datasets", "bundles"}:
            raise StateError("invalid revision namespace")
        if table == "datasets":
            row = connection.execute(
                "SELECT max(revision) AS revision FROM datasets WHERE name=?" + self._published_clause(connection),
                (key,)).fetchone()
        else:
            row = connection.execute(f"SELECT max(revision) AS revision FROM {table} WHERE name=?", (key,)).fetchone()
        return row["revision"] or 0

    def _revision(self, connection, table, key, expected):
        latest = self._latest(connection, table, key)
        if expected is not None and (type(expected) is not int or expected != latest):
            raise StateError("revision precondition no longer holds")
        return latest + 1

    # ------------------------------------------------------------------
    # Dataset writing

    def save_table(self, key, table, expected_revision=None, block_rows=None):
        name(key)
        block_rows = self.block_rows if block_rows is None else block_rows
        if type(block_rows) is not int or block_rows < 1:
            raise StateError("column block size must be positive")
        digest = table.digest()
        schema = canonical(table.schema.state())
        encoded = blocks.encode_table(table, block_rows)
        with self.transaction() as connection:
            revision = self._revision(connection, "datasets", key, expected_revision)
            stale = connection.execute(
                "SELECT published FROM datasets WHERE name=? AND revision=?", (key, revision)).fetchone()
            if stale is not None:
                if stale["published"]:
                    raise StateError("dataset revision is already published")
                # Resume an interrupted write: only unpublished block rows exist.
                connection.execute("DELETE FROM sample_blocks WHERE dataset_name=? AND revision=?",
                                   (key, revision))
                connection.execute("DELETE FROM datasets WHERE name=? AND revision=?", (key, revision))
            connection.execute(
                "INSERT INTO datasets(name,revision,schema_json,digest,row_count,created_at,"
                "format_version,block_rows,published) VALUES(?,?,?,?,?,?,?,?,0)",
                (key, revision, schema, digest, len(table), now(), blocks.FORMAT_VERSION, block_rows))
        # Blocks are committed one at a time so an interrupted write leaves a
        # resumable draft that is invisible to readers.
        for block in encoded["blocks"]:
            with self.transaction() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO sample_blocks VALUES(?,?,?,?,?)",
                    (key, revision, block["index"], canonical(block), block["digest"]))
        with self.transaction() as connection:
            self._publish(connection, key, revision, table, block_rows)
        return {"name": key, "revision": revision, "digest": digest, "rows": len(table),
                "format_version": blocks.FORMAT_VERSION}

    def _publish(self, connection, key, revision, table, block_rows):
        info = connection.execute("SELECT * FROM datasets WHERE name=? AND revision=?", (key, revision)).fetchone()
        if info is None or info["published"]:
            raise StateError("dataset revision cannot be published")
        stored = self._decode_info(info, connection)
        if len(stored) != len(table) or stored.digest() != table.digest():
            raise StateError("stored column blocks differ from the source population")
        connection.execute("UPDATE datasets SET published=1 WHERE name=? AND revision=?", (key, revision))

    def _stored_envelope(self, connection, info):
        if self._value(info, "format_version", 1) != blocks.FORMAT_VERSION:
            raise StateError("dataset revision is not column-block encoded")
        rows = connection.execute(
            "SELECT payload FROM sample_blocks WHERE dataset_name=? AND revision=? ORDER BY block_index",
            (info["name"], info["revision"])).fetchall()
        payloads = [json.loads(row["payload"]) for row in rows]
        if any(block["index"] != i for i, block in enumerate(payloads)):
            raise StateError("stored column block order is invalid")
        return {"format": blocks.TABLE_FORMAT, "version": blocks.FORMAT_VERSION,
                "rows": info["row_count"], "block_rows": self._value(info, "block_rows"),
                "fields": json.loads(info["schema_json"]), "blocks": payloads}

    def _decode_info(self, info, connection):
        if self._value(info, "format_version", 1) == 1:
            rows = connection.execute(
                "SELECT * FROM samples WHERE dataset_name=? AND revision=? ORDER BY position",
                (info["name"], info["revision"])).fetchall()
            return Table(Schema.restore(json.loads(info["schema_json"])),
                         [Sample(row["sample_id"], tuple(json.loads(row["values_json"])), row["weight"])
                          for row in rows])
        return blocks.decode_table(self._stored_envelope(connection, info))

    # ------------------------------------------------------------------
    # Dataset reading

    def _dataset(self, connection, key, revision):
        name(key)
        if revision is None:
            revision = self._latest(connection, "datasets", key)
        row = connection.execute(
            "SELECT * FROM datasets WHERE name=? AND revision=?" + self._published_clause(connection),
            (key, revision)).fetchone()
        if row is None:
            raise IdentityError("dataset revision does not exist")
        return row

    def table_info(self, key, revision=None):
        connection = self.connect()
        try:
            row = self._dataset(connection, key, revision)
            return {"name": row["name"], "revision": row["revision"], "digest": row["digest"],
                    "rows": row["row_count"], "schema": json.loads(row["schema_json"]),
                    "created_at": row["created_at"],
                    "format_version": self._value(row, "format_version", 1),
                    "block_rows": self._value(row, "block_rows")}
        finally:
            connection.close()

    def load_table(self, key, revision=None):
        connection = self.connect()
        try:
            info = self._dataset(connection, key, revision)
            table = self._decode_info(info, connection)
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
            count = 0
            if self._value(info, "format_version", 1) == 1:
                cursor = connection.execute(
                    "SELECT * FROM samples WHERE dataset_name=? AND revision=? ORDER BY position",
                    (key, info["revision"]))
                while True:
                    rows = cursor.fetchmany(size)
                    if not rows:
                        break
                    count += len(rows)
                    yield Table(schema, [Sample(row["sample_id"], tuple(json.loads(row["values_json"])),
                                               row["weight"]) for row in rows])
            else:
                buffer = []
                for _, samples in blocks.decode_table_blocks(self._stored_envelope(connection, info)):
                    buffer.extend(samples)
                    while len(buffer) >= size:
                        chunk, buffer = buffer[:size], buffer[size:]
                        count += len(chunk)
                        yield Table(schema, chunk)
                if buffer:
                    count += len(buffer)
                    yield Table(schema, buffer)
            if count != info["row_count"]:
                raise StateError("stored dataset row count mismatch")
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
            if self._value(info, "format_version", 1) == 1:
                for start in range(0, len(sample_ids), 128):
                    chunk = sample_ids[start:start + 128]
                    placeholders = ",".join("?" for _ in chunk)
                    query = ("SELECT * FROM samples WHERE dataset_name=? AND revision=? "
                             "AND sample_id IN (" + placeholders + ")")
                    for row in connection.execute(query, (key, info["revision"], *chunk)):
                        found[row["sample_id"]] = Sample(
                            row["sample_id"], tuple(json.loads(row["values_json"])), row["weight"])
            else:
                wanted = set(sample_ids)
                for _, samples in blocks.decode_table_blocks(self._stored_envelope(connection, info)):
                    for sample in samples:
                        if sample.id in wanted:
                            found[sample.id] = sample
            if set(found) != set(sample_ids):
                raise IdentityError("requested sample does not exist in the pinned revision")
            return Table(Schema.restore(json.loads(info["schema_json"])), [found[x] for x in sample_ids])
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Bundle writing and reading

    def save_bundle(self, key, bundle, dataset=None, expected_revision=None, block_rows=None):
        name(key)
        block_rows = self.block_rows if block_rows is None else block_rows
        state = bundle.state()
        logical = blocks.bundle_logical_state(state)
        payload = canonical(blocks.bundle_to_v2(logical, block_rows))
        digest = fingerprint(state)
        dataset_name, dataset_revision = (None, None) if dataset is None else dataset
        with self.transaction() as connection:
            if dataset_name is not None:
                info = self._dataset(connection, dataset_name, dataset_revision)
                if info["digest"] != state["training_digest"]:
                    raise StateError("bundle training data differs from bound dataset revision")
                dataset_revision = info["revision"]
            revision = self._revision(connection, "bundles", key, expected_revision)
            connection.execute(
                "INSERT INTO bundles(name,revision,kind,payload,digest,dataset_name,dataset_revision,"
                "created_at,format_version) VALUES(?,?,?,?,?,?,?,?,?)",
                (key, revision, state["kind"], payload, digest, dataset_name, dataset_revision,
                 now(), blocks.FORMAT_VERSION))
        return {"name": key, "revision": revision, "digest": digest, "kind": state["kind"],
                "format_version": blocks.FORMAT_VERSION}

    def _bundle(self, connection, key, revision):
        name(key)
        if revision is None:
            revision = self._latest(connection, "bundles", key)
        row = connection.execute("SELECT * FROM bundles WHERE name=? AND revision=?", (key, revision)).fetchone()
        if row is None:
            raise IdentityError("model bundle revision does not exist")
        try:
            stored = json.loads(row["payload"])
            state = blocks.bundle_logical_state(stored)
        except (TypeError, ValueError, StateError) as error:
            raise StateError("stored bundle payload is unreadable") from error
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
                    "dataset_revision": row["dataset_revision"],
                    "format_version": self._value(row, "format_version", 1),
                    "logical_version": state["version"]}
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Evaluations

    def save_evaluation(self, run_id, bundle_ref, dataset_ref, predictions, metrics):
        name(run_id)
        bundle_name, bundle_revision = bundle_ref
        dataset_name, dataset_revision = dataset_ref
        with self.transaction() as connection:
            bundle, _ = self._bundle(connection, bundle_name, bundle_revision)
            dataset = self._dataset(connection, dataset_name, dataset_revision)
            table = self._decode_info(dataset, connection)
            ids = set(table.ids())
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

    # ------------------------------------------------------------------
    # Bundle envelopes

    def export_bundle(self, key, revision=None):
        connection = self.connect()
        try:
            row, state = self._bundle(connection, key, revision)
            stored = json.loads(row["payload"])
            return canonical({"format": "matrixloom-bundle", "version": blocks.FORMAT_VERSION,
                              "payload": stored, "sha256": fingerprint(stored),
                              "logical_sha256": fingerprint(state)}) + "\n"
        finally:
            connection.close()

    def import_bundle(self, key, text, expected_revision=None):
        envelope = json.loads(text)
        if envelope.get("format") != "matrixloom-bundle":
            raise StateError("unsupported bundle envelope")
        version = envelope.get("version")
        if version == 1:
            payload = envelope["payload"]
            if envelope.get("sha256") != fingerprint(payload):
                raise StateError("bundle envelope checksum mismatch")
            state = payload
        elif version == blocks.FORMAT_VERSION:
            stored = envelope.get("payload")
            if envelope.get("sha256") != fingerprint(stored):
                raise StateError("bundle envelope checksum mismatch")
            state = blocks.bundle_logical_state(stored)
            if envelope.get("logical_sha256") is not None and envelope["logical_sha256"] != fingerprint(state):
                raise StateError("bundle envelope logical checksum mismatch")
        else:
            raise StateError("unsupported bundle envelope")
        return self.save_bundle(key, restore_bundle(state), expected_revision=expected_revision)

    # ------------------------------------------------------------------
    # Inventory inspection

    def revisions(self, key, kind="dataset"):
        table = {"dataset": "datasets", "bundle": "bundles"}.get(kind)
        if table is None:
            raise StateError("invalid revision kind")
        connection = self.connect()
        try:
            version = "format_version" if "format_version" in self._columns(connection, table) else "1 AS format_version"
            if table == "datasets":
                rows = connection.execute(
                    f"SELECT name,revision,digest,created_at,{version} FROM datasets "
                    "WHERE name=?" + self._published_clause(connection) + " ORDER BY revision", (name(key),))
            else:
                rows = connection.execute(
                    f"SELECT name,revision,digest,created_at,{version} FROM bundles "
                    "WHERE name=? ORDER BY revision", (name(key),))
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def drafts(self, key=None):
        """List unpublished, resumable dataset revisions."""
        connection = self.connect()
        try:
            if "published" not in self._columns(connection, "datasets"):
                return []
            if key is None:
                rows = connection.execute(
                    "SELECT name,revision,digest,row_count,created_at FROM datasets "
                    "WHERE published=0 ORDER BY name,revision")
            else:
                rows = connection.execute(
                    "SELECT name,revision,digest,row_count,created_at FROM datasets "
                    "WHERE published=0 AND name=? ORDER BY revision", (name(key),))
            return [dict(row) for row in rows]
        finally:
            connection.close()
