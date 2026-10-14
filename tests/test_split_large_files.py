import os

from release import split_large_files as s


def test_round_trip(tmp_path):
    csv_file = tmp_path / "meta.csv"
    csv_file.write_text('id,text\n' + ''.join(f'{i},"line one\nline ""two"" {i}"\n' for i in range(500)),
                        encoding="utf-8")
    jsonl_file = tmp_path / "index.jsonl"
    jsonl_file.write_text(''.join(f'{{"i": {i}}}\n' for i in range(500)), encoding="utf-8")
    for f in (csv_file, jsonl_file):
        original = f.read_bytes()
        parts = s.split(f, max_bytes=2000)
        assert len(parts) > 1 and all(os.path.getsize(p) <= 2000 for p in parts)
        if f.suffix == ".csv":
            assert all(p.read_text(encoding="utf-8").startswith("id,text\n") for p in parts)
        f.unlink()
        assert s.join(f) and f.read_bytes() == original
        assert s.split(f) == [f] and not s.existing_parts(f)
