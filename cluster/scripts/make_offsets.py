from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HTCondor index intervals.")
    parser.add_argument("--chunks", default="artifacts/rag/chunks_hierarchical.jsonl")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    chunk_path = Path(args.chunks)
    total = count_lines(chunk_path)
    for offset in range(0, total, args.limit):
        interval_limit = min(args.limit, total - offset)
        print(f"{offset}, {interval_limit}, {args.batch_size}")
    return 0


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    raise SystemExit(main())
