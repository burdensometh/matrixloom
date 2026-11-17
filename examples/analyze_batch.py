import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matrixloom import Analysis, Field, Schema, Table
from matrixloom.table import Sample


def main(home):
    schema = Schema([Field("measurement", "Measurement", unit="cm"),
                     Field("batch", "Batch", "category", role="group"),
                     Field("response", "Response", role="target")])
    samples = [Sample(f"sample-{i}", (float(i), "alpha" if i < 6 else "beta", float(2 * i + 1)))
               for i in range(12)]
    app = Analysis(home)
    app.ingest("observations", Table(schema, samples), expected_revision=0)
    spec = {"fields": ["measurement"], "transforms": [{"kind": "standardize"}]}
    fitted = app.train("linear", "observations", spec, {"kind": "ridge", "alpha": .1}, "response")
    reopened = Analysis(home)
    result = reopened.evaluate("reopened-evaluation", "linear", "observations", group_field="batch")
    print(json.dumps({"model": fitted, "evaluation": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
