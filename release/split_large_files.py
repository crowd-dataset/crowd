"""Split CSV/JSONL files over GitHub's size limit into parts, and join them back.

    python split_large_files.py split mapping_metadata.csv crowd_index.jsonl
    python split_large_files.py join mapping_metadata.csv crowd_index.jsonl

A file of at most MAX_BYTES is left whole (stale parts are removed). A larger one becomes
NAME_part1.EXT, NAME_part2.EXT, ... each under MAX_BYTES; CSV parts repeat the header.
"""
import csv
import os
import sys
from pathlib import Path

MAX_BYTES = int(os.environ.get("SPLIT_MAX_BYTES", 99 * 1000 * 1000))  # GitHub refuses files over 100 MB
csv.field_size_limit(sys.maxsize)


def part_path(path, n):
    """NAME_partN.EXT for path NAME.EXT."""
    p = Path(path)
    return p.with_name(f"{p.stem}_part{n}{p.suffix}")


def existing_parts(path):
    """The parts of path that exist, in order, up to the first missing number."""
    parts = []
    while (p := part_path(path, len(parts) + 1)).exists():
        parts.append(p)
    return parts


def _remove_parts_from(path, n):
    """Delete parts n, n + 1, ... of path (stale ones from an earlier, larger split)."""
    while part_path(path, n).exists():
        part_path(path, n).unlink()
        n += 1


def _records(path):
    """Yield (header, raw_record) pairs; a CSV record may span lines inside quotes."""
    with open(path, encoding="utf-8", newline="") as f:
        if Path(path).suffix != ".csv":
            for line in f:
                yield b"", line.encode("utf-8")
            return
        header = f.readline().encode("utf-8")
        buf = ""
        for line in f:
            buf += line
            if buf.count('"') % 2 == 0:  # quotes balanced: the record is complete
                yield header, buf.encode("utf-8")
                buf = ""
        if buf:
            yield header, buf.encode("utf-8")


def split(path, max_bytes=MAX_BYTES):
    """Return the files to commit for path: itself, or its parts."""
    if os.path.getsize(path) <= max_bytes:
        _remove_parts_from(path, 1)
        return [Path(path)]
    n, out, size = 0, None, 0
    for header, rec in _records(path):
        if out is None or size + len(rec) > max_bytes:
            if out:
                out.close()
            n += 1
            out = open(part_path(path, n), "wb")
            out.write(header)
            size = len(header)
        out.write(rec)
        size += len(rec)
    if out:
        out.close()
    _remove_parts_from(path, n + 1)
    return existing_parts(path)


def join(path):
    """Rebuild path from its parts when they are newer than it (or it is missing)."""
    parts = existing_parts(path)
    if not parts or (Path(path).exists()
                     and os.path.getmtime(path) >= max(os.path.getmtime(p) for p in parts)):
        return False
    tmp = Path(f"{path}.tmp")
    with open(tmp, "wb") as out:
        for i, p in enumerate(parts):
            with open(p, "rb") as f:
                if i and Path(path).suffix == ".csv":
                    f.readline()  # header already written by the first part
                out.write(f.read())
    os.replace(tmp, path)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("split", "join"):
        sys.exit(__doc__)
    for name in sys.argv[2:]:
        if sys.argv[1] == "join":
            print(f"{name}: {'joined from parts' if join(name) else 'nothing to join'}")
        else:
            print(f"{name}: " + ", ".join(str(p) for p in split(name)))
