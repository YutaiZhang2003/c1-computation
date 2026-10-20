"""Refresh the source manifest after intentional source edits (maintainers only).

Never called by reproduce.py. Input and reference manifests are not changed.
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]


def main():
    for index_path in sorted((ROOT / "code").rglob("CODE_INDEX.json")):
        index = json.loads(index_path.read_text())
        for entry in index["sources"]:
            current_hash = hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest()
            if entry["sha256"] != current_hash:
                entry.setdefault("original_sha256", entry["sha256"])
                entry["sha256"] = current_hash
        index_path.write_text(json.dumps(index, indent=2, allow_nan=False)+"\n")
        with index_path.with_suffix(".tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["path", "group", "purpose", "sha256"],
                delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(index["sources"])
    path = ROOT / "SOURCE_MANIFEST.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    files = []
    for source in sorted(ROOT.rglob("*")):
        relative = source.relative_to(ROOT)
        if (not source.is_file() or relative.parts[0] in
                {"data", ".git", ".venv", "output", "source_packages"}
                or "__pycache__" in relative.parts
                or source.name in {"SOURCE_MANIFEST.json", ".DS_Store"}
                or source.suffix in {".pyc", ".log"}):
            continue
        contents = source.read_bytes()
        if source.suffix == ".py":
            compile(contents, str(source), "exec")
        files.append(dict(path=str(relative), bytes=len(contents),
                          sha256=hashlib.sha256(contents).hexdigest()))
    previous.update(format="c1-computation-source-v1", files=files)
    path.write_text(json.dumps(previous, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(files)} source/configuration/documentation files.")


if __name__ == "__main__":
    main()
