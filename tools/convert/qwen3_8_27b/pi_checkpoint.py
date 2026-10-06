"""Pinned Pi checkpoint provenance and non-copying native-converter view.

The finetune uses the existing registered Qwen3.8 graph and groupwise recipe.
Only source identity/frontend resources differ. This never borrows base weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("pi_source.json")


def verify_and_prepare(source: Path, view: Path) -> Path:
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if source.resolve() == view.resolve():
        raise ValueError("conversion view must differ from the downloaded source")
    records = []
    for entry in manifest["files"]:
        path = source / entry["path"]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"source size mismatch: {entry['path']}")
        digest = hashlib.sha256() if entry["sha256"] else hashlib.sha1()
        if not entry["sha256"]:
            digest.update(f"blob {entry['size']}\0".encode())
        before = path.stat()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != (entry["sha256"] or entry["git_blob"]):
            raise ValueError(f"publisher digest mismatch: {entry['path']}")
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError("source changed during verification")
        records.append({"path": entry["path"], "digest": actual, "size": after.st_size,
                        "mtime_ns": after.st_mtime_ns})
        print(f"verified {entry['path']}", flush=True)
    view.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        dest = view / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = (source / entry["path"]).resolve()
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != target:
                raise ValueError(f"conversion view would overwrite {dest}")
        else:
            dest.symlink_to(target)
    processor = json.loads((source / "processor_config.json").read_text())
    video = json.dumps(processor["video_processor"], indent=2) + "\n"
    video_path = view / "video_preprocessor_config.json"
    if video_path.exists() and video_path.read_text() != video:
        raise ValueError("derived video resource would overwrite an existing file")
    video_path.write_text(video)
    receipt = {"repository": manifest["repository"], "revision": manifest["revision"],
               "source": str(source.resolve()), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "files": records, "derived_video_sha256": hashlib.sha256(video.encode()).hexdigest()}
    (view / "pi-source-verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return view


def validate_view(view: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    receipt = json.loads((view / "pi-source-verified.json").read_text())
    if receipt.get("manifest_sha256") != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
        raise ValueError("Pi checkpoint verification receipt does not match the pinned manifest")
    if receipt.get("repository") != manifest["repository"] or receipt.get("revision") != manifest["revision"]:
        raise ValueError("Pi checkpoint identity mismatch")
    records = {item["path"]: item for item in receipt["files"]}
    if set(records) != {item["path"] for item in manifest["files"]}:
        raise ValueError("Pi checkpoint receipt is incomplete")
    for item in manifest["files"]:
        path = view / item["path"]
        record = records[item["path"]]
        stat = path.stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != record["mtime_ns"] or record["digest"] != (item["sha256"] or item["git_blob"]):
            raise ValueError(f"verified Pi source changed: {item['path']}")
        # Small frontend metadata is checked again directly, not just by timestamps.
        if not item["sha256"]:
            data = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if actual != item["git_blob"]:
                raise ValueError(f"Pi metadata changed: {item['path']}")
    video = (view / "video_preprocessor_config.json").read_bytes()
    processor = json.loads((view / "processor_config.json").read_text())
    if json.loads(video) != processor["video_processor"] or hashlib.sha256(video).hexdigest() != receipt["derived_video_sha256"]:
        raise ValueError("derived video profile changed")
    return {"repository": manifest["repository"], "revision": manifest["revision"], "profile": "pi"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("view", type=Path)
    args = parser.parse_args()
    verify_and_prepare(args.source, args.view)


if __name__ == "__main__":
    main()
