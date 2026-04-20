from __future__ import annotations

import argparse
import sys

from .demo import main as demo_main


def _parse_root_args(argv: list[str]) -> tuple[str | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.add_parser("parametric")
    subparsers.add_parser("mutation")
    parsed, remaining = parser.parse_known_args(argv)
    return parsed.mode, remaining


def main() -> None:
    mode, remaining = _parse_root_args(sys.argv[1:])
    forwarded = ["autoresearch.demo"]
    if mode is not None:
        forwarded.extend(["--mode", mode])
    forwarded.extend(remaining)
    original_argv = sys.argv
    try:
        sys.argv = forwarded
        demo_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
