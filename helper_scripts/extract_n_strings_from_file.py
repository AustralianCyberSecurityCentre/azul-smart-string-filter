"""Copy the first requested number of nonblank strings from a text file."""

import argparse
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy the first NUMBER nonblank lines from an input file to an "
            "output file. Each nonblank line is treated as one string."
        )
    )
    parser.add_argument("input_file", type=Path, help="File containing the strings")
    parser.add_argument("output_file", type=Path, help="File to write")
    parser.add_argument("number", type=int, help="Number of strings to copy")
    return parser.parse_args()


def extract_first_strings(
    input_file: Path,
    output_file: Path,
    number: int,
) -> int:
    """Write up to number nonblank lines and return the amount written."""
    written = 0

    with input_file.open(
        "r", encoding="utf-8", errors="surrogateescape", newline=""
    ) as source, output_file.open(
        "w", encoding="utf-8", errors="surrogateescape", newline=""
    ) as destination:
        for line in source:
            if not line.strip():
                continue

            destination.write(line)
            written += 1

            if written == number:
                break

    return written


def main() -> None:
    """Validate the arguments, extract the strings, and report the result."""
    arguments = parse_arguments()

    if not arguments.input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {arguments.input_file}")

    if arguments.number < 1:
        raise ValueError("Number of strings must be at least 1")

    if arguments.input_file.resolve() == arguments.output_file.resolve():
        raise ValueError("The input and output files must be different")

    written = extract_first_strings(
        arguments.input_file,
        arguments.output_file,
        arguments.number,
    )

    print(f"Requested strings: {arguments.number}")
    print(f"Strings written: {written}")
    print(f"Output written to: {arguments.output_file}")

    if written < arguments.number:
        print("The input file contained fewer nonblank strings than requested.")


if __name__ == "__main__":
    main()
