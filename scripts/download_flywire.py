"""CLI: download the FlyWire v783 connections and annotations into data/raw/ (md5-checked)."""

from modularfly.config import DataConfig
from modularfly.data.download import ensure_flywire_files


def main() -> None:
    for path in ensure_flywire_files(DataConfig()):
        print(f"ok  {path}")


if __name__ == "__main__":
    main()
