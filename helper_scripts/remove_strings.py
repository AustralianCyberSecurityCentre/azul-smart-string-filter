import os
import shutil
from pathlib import Path

filename = input("Enter filename: ").strip()
file_path = Path(filename)

targets = set()

print("Enter strings to remove, one at a time.")
print("Press Enter on an empty prompt when finished.")

while True:
    value = input("String: ")

    if not value:
        break

    targets.add(value.strip())

if not targets:
    print("No strings entered.")
    raise SystemExit(1)

if not file_path.is_file():
    print(f"File not found: {file_path}")
    raise SystemExit(1)

backup_path = file_path.with_name(file_path.name + ".bak")
temporary_path = file_path.with_name(file_path.name + ".tmp")

shutil.copy2(file_path, backup_path)

removed_lines = []

try:
    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="surrogateescape",
        newline="",
    ) as source, open(
        temporary_path,
        "w",
        encoding="utf-8",
        errors="surrogateescape",
        newline="",
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if line.strip() in targets:
                removed_lines.append((line_number, line.strip()))
                continue

            destination.write(line)

    os.replace(temporary_path, file_path)

except Exception:
    if temporary_path.exists():
        temporary_path.unlink()
    raise

print(f"Removed {len(removed_lines)} matching lines.")
print(f"Backup created: {backup_path}")

for line_number, value in removed_lines:
    print(f"Removed line {line_number}: {value!r}")