"""Extract a random selection of unique strings from a text file."""

import argparse
import random
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Randomly select NUMBER unique, nonblank strings from an input "
            "file and save them to an output file."
        )
    )
    parser.add_argument("input_file", type=Path, help="File containing the strings")
    parser.add_argument("output_file", type=Path, help="File to write")
    parser.add_argument("number", type=int, help="Number of unique strings to extract")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for producing the same selection again",
    )
    return parser.parse_args()


def load_unique_strings(input_file: Path) -> list[str]:
    """Load unique nonblank strings while preserving their original text."""
    unique_strings: list[str] = []
    seen: set[str] = set()

    with input_file.open(
        "r", encoding="utf-8", errors="surrogateescape", newline=""
    ) as source:
        for line in source:
            string = line.rstrip("\r\n")

            if not string.strip() or string in seen:
                continue

            seen.add(string)
            unique_strings.append(string)

    return unique_strings


def main() -> None:
    """Select unique strings randomly and write them to the output file."""
    arguments = parse_arguments()

    if not arguments.input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {arguments.input_file}")

    if arguments.number < 1:
        raise ValueError("Number of strings must be at least 1")

    if arguments.input_file.resolve() == arguments.output_file.resolve():
        raise ValueError("The input and output files must be different")

    unique_strings = load_unique_strings(arguments.input_file)

    if arguments.number > len(unique_strings):
        raise ValueError(
            f"Requested {arguments.number} strings, but the input contains only "
            f"{len(unique_strings)} unique nonblank strings"
        )

    random_generator = random.Random(arguments.seed)
    selected_strings = random_generator.sample(unique_strings, arguments.number)

    with arguments.output_file.open(
        "w", encoding="utf-8", errors="surrogateescape", newline="\n"
    ) as destination:
        for string in selected_strings:
            destination.write(string + "\n")

    print(f"Unique strings available: {len(unique_strings)}")
    print(f"Random strings written: {len(selected_strings)}")
    print(f"Output written to: {arguments.output_file}")


if __name__ == "__main__":
    main()
