"""CLI: download the FlyWire v783 connections and annotations into data/raw/ (md5-checked)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # TODO(debt-001)

from modularfly.config import DataConfig
from modularfly.data.download import ensure_flywire_files


def main() -> None:
    for path in ensure_flywire_files(DataConfig()):
        print(f"ok  {path}")


if __name__ == "__main__":
    main()
