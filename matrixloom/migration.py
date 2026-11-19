"""Migration of a version 1 inventory to version 2 column blocks.

The migration never refits anything.  Every source table revision is decoded
with the version 1 reader, re-encoded as version 2 column blocks, decoded
again, and compared against the original logical state (schema, sample order,
weights, and every value).  Every bundle's embedded training matrix is
converted the same way and its logical digest is preserved.

Publishing is transactional: a dataset revision flips to column blocks and its
legacy sample rows are removed in a single SQLite transaction, and a bundle
payload is replaced in a single transaction.  An interrupted migration leaves
unpublished draft blocks that readers never consult, so the version 1 revision
stays fully readable and a rerun resumes the conversion.
"""

from pathlib import Path
import json
import sqlite3

from . import blocks
from .errors import IdentityError, StateError
from .schema import fingerprint
from .storage import Store, canonical


def _value(row, key, default=None):
    return row[key] if key in row.keys() else default


def _verify_table(source, converted):
    if source.schema.state() != converted.schema.state():
        raise StateError("migrated table schema differs from its source")
    if source.ids() != converted.ids():
        raise StateError("migrated table sample order differs from its source")
    if source.weights() != converted.weights():
        raise StateError("migrated table weights differ from their source")
    if source.digest() != converted.digest():
        raise StateError("migrated table digest differs from its source")


def _verify_bundle(source, converted):
    if fingerprint(source) != fingerprint(converted):
        raise StateError("migrated bundle logical state differs from its source")


def migrate_table(store, key, revision, block_rows=None):
    """Convert one dataset revision to column blocks in place, atomically."""
    if store.read_only:
        raise StateError("migration requires a writable inventory")
    block_rows = store.block_rows if block_rows is None else block_rows
    if type(block_rows) is not int or block_rows < 1:
        raise StateError("column block size must be positive")
    connection = store.connect()
    try:
        row = connection.execute("SELECT * FROM datasets WHERE name=? AND revision=?", (key, revision)).fetchone()
        if row is None:
            raise IdentityError("dataset revision does not exist")
        if _value(row, "format_version", 1) != 1:
            return {"name": key, "revision": revision, "digest": row["digest"],
                    "rows": row["row_count"], "format_version": "already-v2"}
        source = store._decode_info(row, connection)
    finally:
        connection.close()

    encoded = blocks.encode_table(source, block_rows)
    # Stage blocks first.  Readers of a version 1 revision never look at
    # sample_blocks, and a version 2 revision is only visible once the final
    # transaction below flips format_version and removes the legacy rows.
    # Rewriting the same block index makes an interrupted run resumable.
    for block in encoded["blocks"]:
        with store.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO sample_blocks VALUES(?,?,?,?,?)",
                (key, revision, block["index"], canonical(block), block["digest"]))
    with store.transaction() as connection:
        current = connection.execute("SELECT * FROM datasets WHERE name=? AND revision=?",
                                     (key, revision)).fetchone()
        if current is None or _value(current, "format_version", 1) != 1:
            raise StateError("dataset revision changed during migration")
        if current["digest"] != source.digest():
            raise StateError("dataset digest changed during migration")
        draft = dict(current)
        draft.update({"format_version": blocks.FORMAT_VERSION, "block_rows": block_rows, "published": 1})
        draft.pop("row_count_original", None)
        converted = store._decode_info(draft, connection)
        _verify_table(source, converted)
        connection.execute(
            "UPDATE datasets SET format_version=?, block_rows=?, published=1 WHERE name=? AND revision=?",
            (blocks.FORMAT_VERSION, block_rows, key, revision))
        connection.execute("DELETE FROM samples WHERE dataset_name=? AND revision=?", (key, revision))
    return {"name": key, "revision": revision, "digest": source.digest(), "rows": len(source),
            "format_version": blocks.FORMAT_VERSION}


def migrate_bundle(store, key, revision, block_rows=None):
    """Convert a stored bundle payload's training matrix to column blocks."""
    if store.read_only:
        raise StateError("migration requires a writable inventory")
    block_rows = store.block_rows if block_rows is None else block_rows
    connection = store.connect()
    try:
        row = connection.execute("SELECT * FROM bundles WHERE name=? AND revision=?", (key, revision)).fetchone()
        if row is None:
            raise IdentityError("model bundle revision does not exist")
        original = json.loads(row["payload"])
        original_logical = blocks.bundle_logical_state(original)
        if fingerprint(original_logical) != row["digest"]:
            raise StateError("stored bundle integrity check failed")
        if original.get("version") == blocks.FORMAT_VERSION:
            return {"name": key, "revision": revision, "digest": row["digest"],
                    "kind": original.get("kind"), "format_version": "already-v2"}
    finally:
        connection.close()
    converted = blocks.bundle_logical_state(blocks.bundle_to_v2(original_logical, block_rows))
    _verify_bundle(original_logical, converted)
    payload = blocks.bundle_to_v2(original_logical, block_rows)
    with store.transaction() as connection:
        current = connection.execute("SELECT payload,digest FROM bundles WHERE name=? AND revision=?",
                                     (key, revision)).fetchone()
        if current is None or current["digest"] != row["digest"]:
            raise StateError("bundle revision changed during migration")
        connection.execute(
            "UPDATE bundles SET payload=?, format_version=? WHERE name=? AND revision=?",
            (canonical(payload), blocks.FORMAT_VERSION, key, revision))
    return {"name": key, "revision": revision, "digest": row["digest"],
            "kind": original.get("kind"), "format_version": blocks.FORMAT_VERSION}


def _dataset_revisions(store):
    connection = store.connect()
    try:
        return [dict(row) for row in connection.execute("SELECT name,revision FROM datasets ORDER BY name,revision")]
    finally:
        connection.close()


def _bundle_revisions(store):
    connection = store.connect()
    try:
        return [dict(row) for row in connection.execute("SELECT name,revision FROM bundles ORDER BY name,revision")]
    finally:
        connection.close()


def _evaluation_ids(store):
    connection = store.connect()
    try:
        return [row["run_id"] for row in connection.execute("SELECT run_id FROM evaluations ORDER BY run_id")]
    finally:
        connection.close()


def migrate_inventory(path, destination=None, block_rows=None):
    """Migrate every version 1 revision of an inventory to version 2.

    ``destination`` is a database path.  When it differs from ``path`` the
    source is copied first, so the original version 1 inventory is untouched.
    The returned manifest lists each conversion and its verification.
    """
    source_path = Path(path)
    destination = source_path if destination is None else Path(destination)
    if not source_path.exists():
        raise IdentityError("source inventory does not exist")
    if destination.resolve() == source_path.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        store = Store(destination, block_rows=block_rows or blocks.DEFAULT_BLOCK_ROWS)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_inventory(source_path, destination)
        store = Store(destination, block_rows=block_rows or blocks.DEFAULT_BLOCK_ROWS)

    manifest = {"source": str(source_path), "store": str(destination),
                "tables": [], "bundles": [], "evaluations": []}
    for item in _dataset_revisions(store):
        manifest["tables"].append(migrate_table(store, item["name"], item["revision"], block_rows))
    for item in _bundle_revisions(store):
        manifest["bundles"].append(migrate_bundle(store, item["name"], item["revision"], block_rows))
    for run_id in _evaluation_ids(store):
        stored = store.evaluation(run_id)
        manifest["evaluations"].append({"run_id": run_id, "digest": stored["digest"]})
    return manifest


def verify_inventory(legacy_path, migrated_path):
    """Prove that a migrated inventory reproduces the legacy population and values.

    Bundles are used exactly as stored (no fitting) to predict over their bound
    dataset revisions, and every historical evaluation is compared field by
    field.  Returns a report; any discrepancy raises ``StateError``.
    """
    legacy = Store.legacy(legacy_path)
    migrated = Store(migrated_path, read_only=True)
    if Path(legacy_path).resolve() == Path(migrated_path).resolve():
        raise StateError("verification needs an unmigrated legacy inventory and a migrated one")
    report = {"tables": [], "bundles": [], "evaluations": []}
    legacy_connection = legacy.connect()
    try:
        table_rows = [dict(row) for row in legacy_connection.execute(
            "SELECT name,revision,digest,row_count FROM datasets ORDER BY name,revision")]
    finally:
        legacy_connection.close()
    for item in table_rows:
        before = legacy.load_table(item["name"], item["revision"])
        after = migrated.load_table(item["name"], item["revision"])
        _verify_table(before, after)
        if before.digest() != item["digest"] or after.digest() != item["digest"]:
            raise StateError("migrated table no longer matches its recorded digest")
        if len(after) != item["row_count"]:
            raise StateError("migrated table row count changed")
        report["tables"].append({"name": item["name"], "revision": item["revision"],
                                 "digest": after.digest(), "rows": len(after)})
    legacy_connection = legacy.connect()
    try:
        bundle_rows = [dict(row) for row in legacy_connection.execute(
            "SELECT name,revision,digest,kind,dataset_name,dataset_revision FROM bundles "
            "ORDER BY name,revision")]
    finally:
        legacy_connection.close()
    for item in bundle_rows:
        source = legacy.load_bundle(item["name"], item["revision"])
        target = migrated.load_bundle(item["name"], item["revision"])
        source_digest = fingerprint(source.state())
        target_digest = fingerprint(target.state())
        if source_digest != target_digest or target_digest != item["digest"]:
            raise StateError("migrated bundle digest changed")
        if item["dataset_name"] is not None:
            before_table = legacy.load_table(item["dataset_name"], item["dataset_revision"])
            after_table = migrated.load_table(item["dataset_name"], item["dataset_revision"])
            before = source.predict(before_table)
            after = target.predict(after_table)
            if before != after:
                raise StateError("migrated bundle predictions differ from the legacy model")
        else:
            before = after = None
        report["bundles"].append({"name": item["name"], "revision": item["revision"],
                                  "kind": item["kind"], "digest": target_digest,
                                  "predictions_checked": before is not None})
    for run_id in _evaluation_ids(legacy):
        original = legacy.evaluation(run_id)
        converted = migrated.evaluation(run_id)
        if original != converted:
            raise StateError("migrated evaluation differs from its legacy record")
        report["evaluations"].append({"run_id": run_id, "digest": converted["digest"]})
    return report


def migrate_bundle_files(source_directory, destination_directory, block_rows=None):
    """Convert exported version 1 bundle envelopes into version 2 envelopes.

    Each output is re-verified against the source envelope's logical digest.
    """
    source_directory = Path(source_directory)
    destination_directory = Path(destination_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(source_directory.glob("*.json")):
        envelope = json.loads(path.read_text())
        if envelope.get("format") != "matrixloom-bundle":
            continue
        if envelope.get("version") != 1:
            continue
        payload = envelope["payload"]
        if envelope.get("sha256") != fingerprint(payload):
            raise StateError(f"source envelope checksum mismatch for {path.name}")
        stored = blocks.bundle_to_v2(payload, block_rows or blocks.DEFAULT_BLOCK_ROWS)
        logical = blocks.bundle_logical_state(stored)
        _verify_bundle(payload, logical)
        output = {"format": "matrixloom-bundle", "version": blocks.FORMAT_VERSION,
                  "payload": stored, "sha256": fingerprint(stored),
                  "logical_sha256": fingerprint(logical)}
        (destination_directory / path.name).write_text(canonical(output) + "\n")
        manifest.append({"file": path.name, "logical_digest": fingerprint(logical),
                         "stored_digest": fingerprint(stored)})
    return manifest


def _copy_inventory(source, destination):
    if destination.exists():
        destination.unlink()
    from urllib.parse import quote
    source_connection = sqlite3.connect("file:" + quote(str(Path(source).resolve())) + "?mode=ro", uri=True)
    try:
        destination_connection = sqlite3.connect(destination)
        try:
            source_connection.backup(destination_connection)
        finally:
            destination_connection.close()
    finally:
        source_connection.close()


def main(argv=None):
    """Migrate an inventory (or exported bundle directory) to version 2.

    ``matrixloom-migrate SOURCE [DESTINATION]`` converts every version 1 dataset
    revision and bundle in a SQLite inventory, then re-verifies that reopening
    reproduces the legacy population and predictions.  A directory containing
    exported bundle envelopes is converted file by file.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="matrixloom-migrate", description=__doc__)
    parser.add_argument("source", help="version 1 inventory or exported bundle directory")
    parser.add_argument("destination", nargs="?", default=None,
                        help="output inventory or directory (default: migrate in place)")
    parser.add_argument("--block-rows", type=int, default=blocks.DEFAULT_BLOCK_ROWS,
                        help="rows per column block")
    options = parser.parse_args(argv)
    source = Path(options.source)
    if source.is_dir():
        destination = Path(options.destination) if options.destination else source / "v2"
        manifest = migrate_bundle_files(source, destination, options.block_rows)
        print(json.dumps({"bundles": manifest}, indent=2, sort_keys=True))
        return 0
    if options.destination is None:
        # Snapshot the original first so verification compares against the
        # legacy inventory rather than the file just rewritten in place.
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "legacy.sqlite"
            _copy_inventory(source, snapshot)
            manifest = migrate_inventory(source, None, options.block_rows)
            verification = verify_inventory(snapshot, source)
    else:
        manifest = migrate_inventory(source, options.destination, options.block_rows)
        verification = verify_inventory(source, options.destination)
    print(json.dumps({"manifest": manifest, "verification": verification}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
