from __future__ import annotations

import csv
from pathlib import Path


# Update this path if the report is stored elsewhere.
UNMATCHED_CSV = Path(
    "_output/unmatched_bbox_csv_files.csv"
)


def delete_unmatched_files(report_path: Path) -> None:
    """Delete files listed in the report's file_path column."""
    report_path = report_path.expanduser().resolve()

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Unmatched-files report was not found: {report_path}"
        )

    deleted_count = 0
    missing_count = 0
    failed_count = 0

    with report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames or "file_path" not in reader.fieldnames:
            raise ValueError(
                f"{report_path} does not contain a 'file_path' column."
            )

        for row_number, row in enumerate(reader, start=2):
            raw_path = str(row.get("file_path") or "").strip()

            if not raw_path:
                print(f"Skipped empty path at CSV row {row_number}")
                continue

            file_path = Path(raw_path).expanduser()

            try:
                if not file_path.exists():
                    missing_count += 1
                    print(f"Already missing: {file_path}")
                    continue

                if not file_path.is_file():
                    failed_count += 1
                    print(f"Skipped because it is not a file: {file_path}")
                    continue

                file_path.unlink()
                deleted_count += 1
                print(f"Deleted: {file_path}")

            except OSError as error:
                failed_count += 1
                print(f"Could not delete {file_path}: {error}")

    print("\nCompleted")
    print(f"Deleted files: {deleted_count:,}")
    print(f"Already missing: {missing_count:,}")
    print(f"Failed or skipped: {failed_count:,}")


def main() -> None:
    delete_unmatched_files(UNMATCHED_CSV)


if __name__ == "__main__":
    main()