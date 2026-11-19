import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matrixloom import Analysis, Field, Schema, Table
from matrixloom.schema import canonical
from matrixloom.table import Sample


def main(home):
    schema = Schema([Field("measurement", "Measurement", unit="cm"),
                     Field("context", "Context"),
                     Field("batch", "Batch", "category", role="group"),
                     Field("response", "Response", role="target")])
    samples = [Sample(f"sample-{i}", (float(i), float(i % 4),
                                      "alpha" if i < 6 else "beta", 0.5 * i * i + 1))
               for i in range(12)]
    app = Analysis(home)
    app.ingest("observations", Table(schema, samples), expected_revision=0)

    nodes = [
        {"id": "raw", "kind": "source", "fields": ["measurement", "context"],
         "unknown": "zero"},
        {"id": "standardized", "kind": "transform", "inputs": ["raw"],
         "transform": {"kind": "standardize"}},
        {"id": "expansion", "kind": "transform", "inputs": ["standardized"],
         "transform": {"kind": "polynomial", "degree": 2, "interactions_only": True}},
        {"id": "interaction", "kind": "transform", "inputs": ["expansion"],
         "transform": {"kind": "select",
                       "ids": ["product:" + canonical(["measurement", "context"])]}},
        {"id": "features", "kind": "merge", "inputs": ["standardized", "interaction"]},
    ]
    graph = {"kind": "graph", "nodes": nodes, "outputs": ["features"]}
    trained = app.train("linear", "observations", graph, {"kind": "ridge", "alpha": .1}, "response")

    reopened = Analysis(home)
    frame = reopened.store.load_bundle("linear").encoded(
        reopened.store.load_table("observations"))
    print(json.dumps({"model": trained,
                      "feature_ids": list(frame.feature_ids()),
                      "predictions": reopened.predict("linear", "observations")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
