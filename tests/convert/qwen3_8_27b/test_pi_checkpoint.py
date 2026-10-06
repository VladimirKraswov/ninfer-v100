from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.convert.qwen3_8_27b import pi_checkpoint


def checkpoint(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    files = {"model.safetensors": b"pinned weight bytes", "processor_config.json": b'{"video_processor":{"fps":2}}'}
    entries = []
    for name, data in files.items():
        (source / name).write_bytes(data)
        large = name.endswith("safetensors")
        entries.append({"path": name, "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest() if large else None,
                        "git_blob": None if large else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repository": "test/pinned", "revision": "fixed", "files": entries}))
    return source, tmp_path / "view", manifest


def test_verified_view_retains_source_and_checks_derived_frontend(tmp_path: Path) -> None:
    source, view, manifest = checkpoint(tmp_path)
    with patch.object(pi_checkpoint, "MANIFEST", manifest):
        pi_checkpoint.verify_and_prepare(source, view)
        assert pi_checkpoint.validate_view(view)["revision"] == "fixed"
        assert (view / "model.safetensors").resolve() == source / "model.safetensors"
        pi_checkpoint.verify_and_prepare(source, view)  # resumable, no replacement of source weights
        (view / "video_preprocessor_config.json").write_text('{"fps":9}')
        with pytest.raises(ValueError, match="derived video"):
            pi_checkpoint.validate_view(view)


def test_publisher_mismatch_and_changed_source_block_conversion(tmp_path: Path) -> None:
    source, view, manifest = checkpoint(tmp_path)
    with patch.object(pi_checkpoint, "MANIFEST", manifest):
        pi_checkpoint.verify_and_prepare(source, view)
        (source / "model.safetensors").write_bytes(b"altered weight bytes")
        with pytest.raises(ValueError, match="source changed"):
            pi_checkpoint.validate_view(view)
        with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
            pi_checkpoint.verify_and_prepare(source, tmp_path / "bad-view")
        assert not (tmp_path / "bad-view").exists()


def test_preparation_never_overwrites_existing_view_files(tmp_path: Path) -> None:
    source, view, manifest = checkpoint(tmp_path)
    view.mkdir()
    (view / "model.safetensors").write_bytes(b"keep owner file")
    with patch.object(pi_checkpoint, "MANIFEST", manifest):
        with pytest.raises(ValueError, match="would overwrite"):
            pi_checkpoint.verify_and_prepare(source, view)
    assert (view / "model.safetensors").read_bytes() == b"keep owner file"
