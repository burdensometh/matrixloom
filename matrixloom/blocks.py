"""Versioned column-block encoding for sample tables and fitted training matrices.

Format version 2 stores a population as independent, checksummed row blocks.
Every block explicitly records its sample order, the identity of every column,
the per-column missing mask, and the sample weights.  The legacy version 1
format stored one JSON array per sample row; it remains readable but is never
produced by new writes.

Column orientation keeps each block's columns contiguous, while the per-column
``missing`` mask makes an absent observation explicit instead of relying on a
JSON ``null`` sentinel inside a row of mixed types.
"""

import copy

from .errors import IdentityError, SchemaError, StateError
from .frame import Feature, Frame
from .schema import Schema, fingerprint
from .table import Sample, Table


TABLE_FORMAT = "matrixloom-table-blocks"
FRAME_FORMAT = "matrixloom-frame-blocks"
FORMAT_VERSION = 2
DEFAULT_BLOCK_ROWS = 256


# ---------------------------------------------------------------------------
# Encoding


def _make_block(index, first_row, ids, weights, column_ids, columns, missing):
    body = {"index": index, "first_row": first_row, "sample_ids": list(ids),
            "weights": list(weights), "column_ids": list(column_ids),
            "columns": [list(column) for column in columns],
            "missing": [list(mask) for mask in missing]}
    body["digest"] = fingerprint(body)
    return body


def _column_blocks(ids, weights, values, column_ids, block_rows):
    width = len(column_ids)
    blocks = []
    for index, start in enumerate(range(0, len(ids), block_rows)):
        stop = min(start + block_rows, len(ids))
        selected = values[start:stop]
        columns = [[row[j] for row in selected] for j in range(width)]
        missing = [[1 if row[j] is None else 0 for row in selected] for j in range(width)]
        blocks.append(_make_block(index, start, ids[start:stop], weights[start:stop],
                                  column_ids, columns, missing))
    return blocks


def _check_block_size(block_rows):
    if type(block_rows) is not int or block_rows < 1:
        raise SchemaError("block size must be a positive integer")


def encode_table(table, block_rows=DEFAULT_BLOCK_ROWS):
    """Encode a table as a version 2 column-block structure."""
    _check_block_size(block_rows)
    ids = list(table.ids())
    weights = list(table.weights())
    values = [sample.values for sample in table.samples]
    return {"format": TABLE_FORMAT, "version": FORMAT_VERSION, "rows": len(table),
            "block_rows": block_rows, "fields": table.schema.state(),
            "blocks": _column_blocks(ids, weights, values, table.schema.ids(), block_rows)}


def encode_frame(frame, block_rows=DEFAULT_BLOCK_ROWS):
    """Encode a fitted numeric matrix as version 2 column blocks."""
    _check_block_size(block_rows)
    return {"format": FRAME_FORMAT, "version": FORMAT_VERSION, "rows": len(frame),
            "block_rows": block_rows, "features": [f.state() for f in frame.features],
            "blocks": _column_blocks(list(frame.ids), list(frame.weights), list(frame.values),
                                     frame.feature_ids(), block_rows)}


# ---------------------------------------------------------------------------
# Decoding and validation


def _envelope(state, expected_format):
    if not isinstance(state, dict):
        raise StateError("column-block state must be an object")
    if state.get("format") != expected_format or state.get("version") != FORMAT_VERSION:
        raise StateError("unsupported column-block format")
    rows = state.get("rows")
    block_rows = state.get("block_rows")
    blocks = state.get("blocks")
    if type(rows) is not int or rows < 0:
        raise StateError("column-block row count is invalid")
    if type(block_rows) is not int or block_rows < 1:
        raise StateError("column-block size is invalid")
    if not isinstance(blocks, list):
        raise StateError("column-block list is malformed")
    return rows, block_rows, blocks


def _verify_block(block, width, expected_columns, block_rows, expected_index, first_row):
    """Validate one column block and return its ordered identities."""
    if not isinstance(block, dict) or "digest" not in block:
        raise StateError("column block lacks a checksum")
    body = {key: value for key, value in block.items() if key != "digest"}
    if fingerprint(body) != block["digest"]:
        raise StateError("column block checksum mismatch")
    if block.get("index") != expected_index or block.get("first_row") != first_row:
        raise StateError("column block position is inconsistent")
    ids = block.get("sample_ids")
    weights = block.get("weights")
    columns = block.get("columns")
    missing = block.get("missing")
    if not isinstance(ids, list) or not isinstance(weights, list):
        raise StateError("column block population is malformed")
    if tuple(block.get("column_ids") or ()) != tuple(expected_columns):
        raise StateError("column block feature identity does not match its header")
    if not isinstance(columns, list) or not isinstance(missing, list):
        raise StateError("column block columns are malformed")
    if len(columns) != width or len(missing) != width:
        raise StateError("column block width differs from its feature header")
    count = len(ids)
    if count < 1 or count > block_rows or len(weights) != count:
        raise StateError("column block row count is invalid")
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != count:
        raise StateError("column block sample identities are invalid")
    if any(not isinstance(column, list) or len(column) != count for column in columns):
        raise StateError("column block width differs from its population")
    if any(not isinstance(mask, list) or len(mask) != count for mask in missing):
        raise StateError("column block missing mask differs from its population")
    for mask in missing:
        for flag in mask:
            if type(flag) is not int or flag not in (0, 1):
                raise StateError("column block missing mask must contain 0 or 1")
    for column, mask in zip(columns, missing):
        for value, flag in zip(column, mask):
            if flag == 1 and value is not None:
                raise StateError("column block marks an observed value as missing")
            if flag == 0 and value is None:
                raise StateError("column block marks a missing value as observed")
    return ids, weights


def _block_rows(block, count, width):
    columns = block["columns"]
    missing = block["missing"]
    return [tuple(None if missing[j][i] else columns[j][i] for j in range(width))
            for i in range(count)]


def _decode(state, expected_format, column_ids, width):
    total, block_rows, blocks = _envelope(state, expected_format)
    if total == 0:
        if blocks:
            raise StateError("empty column-block population carries blocks")
        return [], [], []
    if not blocks:
        raise StateError("column-block population lacks blocks")
    ids = []
    weights = []
    rows = []
    seen = set()
    first_row = 0
    for expected_index, block in enumerate(blocks):
        block_ids, block_weights = _verify_block(block, width, column_ids, block_rows,
                                                 expected_index, first_row)
        last = expected_index + 1 == len(blocks)
        if not last and len(block_ids) != block_rows:
            raise StateError("non-final column block is not full")
        if len(ids) + len(block_ids) > total:
            raise StateError("column blocks exceed the declared population")
        for sample_id in block_ids:
            if sample_id in seen:
                raise IdentityError("column blocks repeat a sample identity")
            seen.add(sample_id)
        first_row += len(block_ids)
        ids.extend(block_ids)
        weights.extend(block_weights)
        rows.extend(_block_rows(block, len(block_ids), width))
    if len(ids) != total:
        raise StateError("column blocks do not cover the declared population")
    return ids, weights, rows


def decode_table(state):
    """Restore a table from validated version 2 column blocks."""
    if not isinstance(state, dict):
        raise StateError("column-block table must be an object")
    schema = Schema.restore(state.get("fields"))
    ids, weights, rows = _decode(state, TABLE_FORMAT, schema.ids(), len(schema.fields))
    return Table(schema, [Sample(key, row, weight) for key, row, weight in zip(ids, rows, weights)])


def decode_frame(state):
    """Restore a fitted numeric matrix from validated version 2 column blocks."""
    if not isinstance(state, dict):
        raise StateError("column-block matrix must be an object")
    features = tuple(Feature(**item) for item in state.get("features") or ())
    column_ids = tuple(feature.id for feature in features)
    if len(set(column_ids)) != len(column_ids):
        raise StateError("column-block feature identity is not unique")
    ids, weights, rows = _decode(state, FRAME_FORMAT, column_ids, len(features))
    return Frame(ids, features, rows, weights)


def decode_table_blocks(state):
    """Yield ``(block_index, samples)`` for validated table blocks.

    Used by streaming consumers that must not materialise the whole population.
    """
    if not isinstance(state, dict):
        raise StateError("column-block table must be an object")
    schema = Schema.restore(state.get("fields"))
    total, block_rows, blocks = _envelope(state, TABLE_FORMAT)
    width = len(schema.fields)
    first_row = 0
    seen = set()
    for expected_index, block in enumerate(blocks):
        ids, weights = _verify_block(block, width, schema.ids(), block_rows, expected_index, first_row)
        last = expected_index + 1 == len(blocks)
        if not last and len(ids) != block_rows:
            raise StateError("non-final column block is not full")
        if first_row + len(ids) > total:
            raise StateError("column blocks exceed the declared population")
        for sample_id in ids:
            if sample_id in seen:
                raise IdentityError("column blocks repeat a sample identity")
            seen.add(sample_id)
        first_row += len(ids)
        rows = _block_rows(block, len(ids), width)
        yield expected_index, [Sample(key, row, weight) for key, row, weight in zip(ids, rows, weights)]
    if first_row != total:
        raise StateError("column blocks do not cover the declared population")


# ---------------------------------------------------------------------------
# Bundle payload conversion (version 1 logical state <-> version 2 storage)


def is_frame_block_state(state):
    return isinstance(state, dict) and state.get("format") == FRAME_FORMAT


def frame_logical_state(state):
    """Return the logical frame state, decoding column blocks when present."""
    if is_frame_block_state(state):
        return decode_frame(state).state()
    if not isinstance(state, dict):
        raise StateError("frame state must be an object")
    return state


def bundle_to_v2(state, block_rows=DEFAULT_BLOCK_ROWS):
    """Convert a logical version 1 bundle payload into its version 2 storage form."""
    if not isinstance(state, dict):
        raise StateError("bundle state must be an object")
    version = state.get("version")
    if version == FORMAT_VERSION:
        return copy.deepcopy(state)
    if version != 1:
        raise StateError("unsupported bundle state version")
    payload = copy.deepcopy(state)
    payload["version"] = FORMAT_VERSION
    model = payload.get("model")
    if isinstance(model, dict) and "training" in model:
        training = model["training"]
        if not is_frame_block_state(training):
            model["training"] = encode_frame(Frame.restore(training), block_rows)
    return payload


def bundle_logical_state(payload):
    """Return the version 1 logical state of a stored bundle payload.

    Digests identify content, not encoding, so they continue to be computed
    over this canonical logical form even after the payload has moved to
    version 2 column blocks.
    """
    if not isinstance(payload, dict):
        raise StateError("bundle payload must be an object")
    version = payload.get("version")
    if version == 1:
        return payload
    if version != FORMAT_VERSION:
        raise StateError("unsupported bundle payload version")
    logical = copy.deepcopy(payload)
    logical["version"] = 1
    model = logical.get("model")
    if isinstance(model, dict) and "training" in model:
        model["training"] = frame_logical_state(model["training"])
    return logical
