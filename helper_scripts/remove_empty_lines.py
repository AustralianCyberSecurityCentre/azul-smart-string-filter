"""Remove empty and whitespace-only lines from a text file."""

import argparse
import os
import tempfile
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Remove lines containing only whitespace, including spaces and tabs. "
            "All nonblank lines are preserved exactly."
        )
    )
    parser.add_argument("input_file", type=Path, help="File to process")
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="File to write (omit when using --in-place)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Safely replace the input file with the cleaned version",
    )
    return parser.parse_args()


def remove_blank_lines(input_file: Path, output_file: Path) -> tuple[int, int]:
    """Copy nonblank lines to output_file and return (kept, removed)."""
    kept = 0
    removed = 0

    with input_file.open(
        "r", encoding="utf-8", errors="surrogateescape", newline=""
    ) as source, output_file.open(
        "w", encoding="utf-8", errors="surrogateescape", newline=""
    ) as destination:
        for line in source:
            if line.strip():
                destination.write(line)
                kept += 1
            else:
                removed += 1

    return kept, removed


def main() -> None:
    """Remove blank lines and report the result."""
    arguments = parse_arguments()
    input_file = arguments.input_file
    output_file = arguments.output_file

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if arguments.in_place and output_file is not None:
        raise ValueError("Do not provide an output file when using --in-place")

    if not arguments.in_place and output_file is None:
        raise ValueError("Provide an output file or use --in-place")

    if output_file is not None and input_file.resolve() == output_file.resolve():
        raise ValueError("Use --in-place when the input and output paths are the same")

    if arguments.in_place:
        temporary_handle, temporary_name = tempfile.mkstemp(
            prefix=f".{input_file.name}.",
            suffix=".tmp",
            dir=input_file.parent,
        )
        os.close(temporary_handle)
        temporary_file = Path(temporary_name)

        try:
            kept, removed = remove_blank_lines(input_file, temporary_file)
            os.replace(temporary_file, input_file)
        finally:
            if temporary_file.exists():
                temporary_file.unlink()

        destination = input_file
    else:
        assert output_file is not None
        kept, removed = remove_blank_lines(input_file, output_file)
        destination = output_file

    print(f"Kept lines: {kept}")
    print(f"Blank lines removed: {removed}")
    print(f"Output written to: {destination}")


if __name__ == "__main__":
    main()
