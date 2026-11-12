"""Network only: fetch raw FlyWire files into data/raw/ and check their md5."""
import hashlib
import shutil
import urllib.request
from pathlib import Path

from modularfly.config import DataConfig, RemoteFile


def file_md5(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def check_md5(path: Path, expected: str, chunk_bytes: int) -> None:
    actual = file_md5(path, chunk_bytes)
    if actual != expected:
        raise ValueError(f"md5 mismatch for {path}: expected {expected}, got {actual}")


def fetch(url: str, dest: Path, chunk_bytes: int) -> None:
    partial = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, chunk_bytes)
    partial.replace(dest)


def ensure_file(remote: RemoteFile, raw_dir: Path, chunk_bytes: int) -> Path:
    """Download unless already present; either way the md5 must match."""
    dest = Path(raw_dir) / remote.relative_path
    if not dest.exists():
        fetch(remote.url, dest, chunk_bytes)
    check_md5(dest, remote.md5, chunk_bytes)
    return dest


def ensure_flywire_files(cfg: DataConfig) -> tuple[Path, Path]:
    connections = ensure_file(cfg.connections, cfg.raw_dir, cfg.download_chunk_bytes)
    annotations = ensure_file(cfg.annotations, cfg.raw_dir, cfg.download_chunk_bytes)
    return connections, annotations
