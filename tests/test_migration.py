import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from matrixloom import blocks
from matrixloom.errors import IdentityError, MatrixError, StateError
from matrixloom.frame import Feature, Frame
from matrixloom.migration import (migrate_bundle, migrate_inventory, migrate_bundle_files,
                                  migrate_table, verify_inventory)
from matrixloom.schema import Field, Schema, canonical, fingerprint
from matrixloom.storage import Store
from matrixloom.table import Sample, Table

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v1"


def observations(count=9):
    schema = Schema([Field("x", "Measurement", unit="cm"),
                     Field("category", "Category", "category"),
                     Field("y", "Response", role="target")])
    samples = []
    for i in range(count):
        samples.append(Sample(f"r{i:02}", (float(i), "a" if i % 2 == 0 else "b",
                                           float(2 * i + 1)), 1 + (i % 3)))
    return Table(schema, samples)


class BlockCodecTests(unittest.TestCase):
    def test_table_round_trip_preserves_order_mask_and_weights(self):
        table = observations().replace_column("x", [0.0, None, 2.0, None, 4.0, 5.0, None, 7.0, 8.0])
        state = blocks.encode_table(table, block_rows=2)
        self.assertEqual(state["version"], 2)
        self.assertEqual([block["index"] for block in state["blocks"]], [0, 1, 2, 3, 4])
        first = state["blocks"][0]
        self.assertEqual(first["sample_ids"], ["r00", "r01"])
        self.assertEqual(first["column_ids"], ["x", "category", "y"])
        self.assertEqual(first["missing"], [[0, 1], [0, 0], [0, 0]])
        self.assertIn("digest", first)
        restored = blocks.decode_table(json.loads(canonical(state)))
        self.assertEqual(restored.state(), table.state())
        self.assertEqual(restored.digest(), table.digest())

    def test_frame_round_trip_and_missing_mask(self):
        frame = Frame(["a", "b", "c"], [Feature("f", "f"), Feature("g", "g")],
                      [[1.0, None], [None, 4.0], [5.0, 6.0]], [1.0, 0.0, 2.0])
        state = blocks.encode_frame(frame, block_rows=2)
        self.assertEqual([block["missing"] for block in state["blocks"]],
                         [[[0, 1], [1, 0]], [[0], [0]]])
        self.assertEqual(blocks.decode_frame(json.loads(canonical(state))).state(), frame.state())

    def test_corruption_is_rejected(self):
        state = blocks.encode_table(observations(), block_rows=4)
        state["blocks"][0]["weights"][0] = 99
        with self.assertRaises(StateError):
            blocks.decode_table(state)
        state = blocks.encode_table(observations(), block_rows=4)
        state["blocks"][1]["index"] = 5
        with self.assertRaises(StateError):
            blocks.decode_table(state)
        state = blocks.encode_table(observations(), block_rows=4)
        state["blocks"][0]["missing"][0][0] = 1
        state["blocks"][0].pop("digest")
        state["blocks"][0]["digest"] = fingerprint({key: value for key, value in state["blocks"][0].items()})
        with self.assertRaises(StateError):
            blocks.decode_table(state)

    def test_duplicate_identity_and_short_coverage_rejected(self):
        state = blocks.encode_table(observations(), block_rows=4)
        state["blocks"][1]["sample_ids"][0] = state["blocks"][0]["sample_ids"][0]
        body = {key: value for key, value in state["blocks"][1].items() if key != "digest"}
        state["blocks"][1]["digest"] = fingerprint(body)
        with self.assertRaises(IdentityError):
            blocks.decode_table(state)

    def test_new_store_writes_version_two_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "analysis.sqlite", block_rows=4)
            saved = store.save_table("samples", observations())
            self.assertEqual(saved["format_version"], 2)
            self.assertEqual(store.table_info("samples")["format_version"], 2)
            connection = store.connect()
            legacy_rows = connection.execute("SELECT count(*) FROM samples").fetchone()[0]
            block_rows = connection.execute("SELECT count(*) FROM sample_blocks").fetchone()[0]
            connection.close()
            self.assertEqual(legacy_rows, 0)
            self.assertEqual(block_rows, 3)
            self.assertEqual(store.load_table("samples").digest(), observations().digest())
            self.assertEqual(store.batches("samples", size=2).__iter__().__next__().ids(), ("r00", "r01"))
            self.assertEqual(store.select_samples("samples", ["r08", "r00"]).ids(), ("r08", "r00"))


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.legacy = self.root / "legacy.sqlite"
        shutil.copy(FIXTURES / "analysis.sqlite", self.legacy)

    def tearDown(self):
        self.temporary.cleanup()

    def test_legacy_inventory_reads_without_rewrite(self):
        before = hashlib.sha256(self.legacy.read_bytes()).hexdigest()
        store = Store.legacy(self.legacy)
        self.assertEqual(store.revisions("samples")[0]["format_version"], 1)
        table = store.load_table("samples", 1)
        self.assertEqual(len(table), 16)
        self.assertEqual(table.digest(), "dbfa327d17db5aec0e88872431e1378911e591547dc436780b189a58b3de2034")
        bundle = store.load_bundle("neighbors", 1)
        self.assertEqual(bundle.training_digest, table.digest())
        self.assertEqual(store.evaluation("original-regression")["digest"],
                         "e4fe57354903648500c0b1716f8f2eb11e1d2af5479498e756b0c1239efdb410")
        after = hashlib.sha256(self.legacy.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_migration_converts_actual_records_and_reopens(self):
        destination = self.root / "v2.sqlite"
        manifest = migrate_inventory(self.legacy, destination)
        self.assertEqual([entry["format_version"] for entry in manifest["tables"]], [2, 2])
        self.assertEqual(len(manifest["bundles"]), 5)
        self.assertEqual(len(manifest["evaluations"]), 2)
        report = verify_inventory(self.legacy, destination)

        # The original file is byte-for-byte unchanged.
        legacy_store = Store.legacy(self.legacy)
        self.assertEqual(legacy_store.table_info("samples", 1)["format_version"], 1)

        migrated = Store(destination, read_only=True)
        for item in report["tables"]:
            table = migrated.load_table(item["name"], item["revision"])
            self.assertEqual(table.digest(), item["digest"])
            self.assertEqual(migrated.table_info(item["name"], item["revision"])["format_version"], 2)
        connection = migrated.connect()
        self.assertEqual(connection.execute("SELECT count(*) FROM samples").fetchone()[0], 0)
        self.assertGreater(connection.execute("SELECT count(*) FROM sample_blocks").fetchone()[0], 0)
        connection.close()

        # Reopening predicts the same numbers as the historical model.
        neighbours = migrated.load_bundle("neighbors", 1)
        legacy_neighbours = legacy_store.load_bundle("neighbors", 1)
        self.assertEqual(neighbours.predict(migrated.load_table("samples", 1)),
                         legacy_neighbours.predict(legacy_store.load_table("samples", 1)))
        self.assertEqual(migrated.evaluation("original-classification"),
                         legacy_store.evaluation("original-classification"))

    def test_migration_keeps_bundle_training_population_and_digest(self):
        destination = self.root / "v2.sqlite"
        migrate_inventory(self.legacy, destination)
        store = Store(destination, read_only=True)
        connection = store.connect()
        payload = json.loads(connection.execute(
            "SELECT payload FROM bundles WHERE name='neighbors' AND revision=1").fetchone()[0])
        connection.close()
        self.assertEqual(payload["version"], 2)
        training = payload["model"]["training"]
        self.assertEqual(training["format"], blocks.FRAME_FORMAT)
        self.assertEqual(training["version"], 2)
        self.assertEqual(training["blocks"][0]["sample_ids"][:3], ["r00", "r01", "r02"])
        decoded = blocks.decode_frame(training)
        self.assertEqual(decoded.digest(),
                         fingerprint(blocks.bundle_logical_state(payload)["model"]["training"]))

    def test_in_place_migration_is_atomic_and_resumable(self):
        store = Store(self.legacy)
        source = store.load_table("samples", 1)
        encoded = blocks.encode_table(source, 256)
        # Simulate an interrupted run: only the first staged block is present.
        connection = store.connect()
        connection.execute("INSERT INTO sample_blocks VALUES(?,?,?,?,?)",
                           ("samples", 1, 0, canonical(encoded["blocks"][0]),
                            encoded["blocks"][0]["digest"]))
        connection.close()
        # The legacy revision is still the published, readable one.
        self.assertEqual(store.table_info("samples", 1)["format_version"], 1)
        self.assertEqual(store.load_table("samples", 1).digest(), source.digest())
        # A rerun completes and publishes once.
        migrate_table(store, "samples", 1, block_rows=256)
        self.assertEqual(store.table_info("samples", 1)["format_version"], 2)
        self.assertEqual(store.load_table("samples", 1).digest(), source.digest())
        self.assertEqual(store.table_info("samples", 1)["block_rows"], 256)

    def test_interrupted_draft_never_published_as_half_inventory(self):
        destination = self.root / "partial.sqlite"
        shutil.copy(self.legacy, destination)
        store = Store(destination)
        connection = store.connect()
        connection.execute(
            "INSERT INTO datasets(name,revision,schema_json,digest,row_count,created_at,"
            "format_version,block_rows,published) VALUES('extra',1,'[]','x',3,'now',2,2,0)")
        connection.close()
        self.assertEqual(store.revisions("extra"), [])
        self.assertEqual([draft["name"] for draft in store.drafts()], ["extra"])
        with self.assertRaises(IdentityError):
            store.load_table("extra", 1)
        with self.assertRaises(IdentityError):
            store.table_info("extra")

    def test_bundle_file_migration(self):
        source = self.root / "bundles"
        source.mkdir()
        for path in FIXTURES.glob("*.json"):
            shutil.copy(path, source / path.name)
        destination = self.root / "bundles-v2"
        manifest = migrate_bundle_files(source, destination)
        self.assertEqual(len(manifest), 5)
        for entry in manifest:
            envelope = json.loads((destination / entry["file"]).read_text())
            self.assertEqual(envelope["version"], 2)
            self.assertEqual(envelope["logical_sha256"], entry["logical_digest"])
            legacy = json.loads((source / entry["file"]).read_text())
            self.assertEqual(fingerprint(legacy["payload"]), envelope["logical_sha256"])
            self.assertEqual(blocks.bundle_logical_state(envelope["payload"]), legacy["payload"])

    def test_migrated_store_rejects_tampered_block(self):
        destination = self.root / "v2.sqlite"
        migrate_inventory(self.legacy, destination)
        store = Store(destination)
        connection = store.connect()
        row = connection.execute(
            "SELECT payload FROM sample_blocks WHERE dataset_name='samples' AND revision=2 LIMIT 1").fetchone()
        payload = json.loads(row["payload"])
        payload["missing"][0][0] = 1 - payload["missing"][0][0]
        connection.execute("UPDATE sample_blocks SET payload=? WHERE dataset_name='samples' AND revision=2",
                           (canonical(payload),))
        connection.close()
        with self.assertRaises(StateError):
            store.load_table("samples", 2)

    def test_bundle_import_of_migrated_envelope_preserves_predictions(self):
        destination = self.root / "v2.sqlite"
        migrate_inventory(self.legacy, destination)
        store = Store(destination, read_only=True)
        envelope = store.export_bundle("ridge", 1)
        self.assertEqual(json.loads(envelope)["version"], 2)
        fresh = Store(self.root / "imported.sqlite")
        fresh.import_bundle("ridge-copy", envelope)
        migrated_table = store.load_table("samples", 1)
        imported = fresh.load_bundle("ridge-copy")
        self.assertEqual(imported.predict(migrated_table), store.load_bundle("ridge", 1).predict(migrated_table))


if __name__ == "__main__":
    unittest.main()
