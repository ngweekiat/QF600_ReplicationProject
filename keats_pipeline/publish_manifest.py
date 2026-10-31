"""Record checksums of the acquired inputs without reading or publishing credentials."""
from pathlib import Path
import hashlib
import json

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def main():
    files = []
    for path in sorted((ROOT / "data/inputs").glob("*.parquet")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        parquet = pq.ParquetFile(path)
        files.append({"file": path.name, "bytes": path.stat().st_size,
                      "rows": parquet.metadata.num_rows, "columns": parquet.schema.names,
                      "sha256": digest.hexdigest()})
    manifest = {"acquired_date": "2026-10-31", "files": files, "normal_runner_downloads_enabled": False,
                "source_missing_values_preserved": True}
    (ROOT / "data/inputs/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {len(files)} Parquet inputs; {sum(f['bytes'] for f in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
