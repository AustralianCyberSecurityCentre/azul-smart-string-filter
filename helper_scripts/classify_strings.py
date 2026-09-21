"""Extract strings from a binary and classify them with SmartStringFilter."""

import argparse
import csv
import sys
from pathlib import Path

import binary2strings as b2s

from azul_smart_string_filter.lib import SmartStringFilter


# The training code assigns 1 to good strings and 0 to bad strings.
GOOD_LABEL = 1
BAD_LABEL = 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract strings from a binary and classify them with the selected "
            "win/elf AI model."
        )
    )
    parser.add_argument("binary_file", type=Path, help="Binary file to inspect")
    parser.add_argument("model_type", help='Model type, such as "win" or "elf"')
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional TSV output file. Results are printed to stdout when omitted.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=4,
        help="Minimum extracted string length (default: 4)",
    )
    return parser.parse_args()


def extract_strings(binary_file: Path, min_chars: int) -> list[str]:
    """Extract UTF-8 and wide strings from a binary using binary2strings."""
    if not binary_file.is_file():
        raise FileNotFoundError(f"Binary file not found: {binary_file}")

    if min_chars < 1:
        raise ValueError("Minimum string length must be at least 1")

    with binary_file.open("rb") as file:
        binary_data = file.read()

    extracted = b2s.extract_all_strings(
        binary_data,
        min_chars=min_chars,
        only_interesting=False,
    )

    # binary2strings returns:
    # (string, encoding, (start_offset, end_offset), is_interesting)
    return [string for string, _encoding, _span, _is_interesting in extracted]


def classify_strings(
    strings: list[str],
    model_type: str,
) -> list[bool]:
    """Classify every extracted string in one model call."""
    string_filter = SmartStringFilter()
    return string_filter.find_legible_strings(strings, model_type=model_type)


def write_results(
    strings: list[str],
    predictions: list[bool],
    output_file: Path | None,
) -> None:
    """Write each string beside its raw model label and classification."""
    destination = (
        output_file.open("w", encoding="utf-8", newline="")
        if output_file
        else sys.stdout
    )

    try:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["string", "prediction", "classification"])

        for string, prediction in zip(strings, predictions, strict=True):
            label = int(prediction)

            if label == GOOD_LABEL:
                classification = "GOOD"
            elif label == BAD_LABEL:
                classification = "BAD"
            else:
                classification = "UNKNOWN"

            writer.writerow([string, label, classification])
    finally:
        if output_file:
            destination.close()


def main() -> None:
    """Extract binary strings, run the model, and output aligned results."""
    arguments = parse_arguments()
    strings = extract_strings(arguments.binary_file, arguments.min_chars)

    if not strings:
        raise ValueError(f"No strings found in {arguments.binary_file}")

    print(
        f"Extracted {len(strings)} strings. Starting classification...",
        file=sys.stderr,
        flush=True,
    )

    predictions = classify_strings(
        strings,
        arguments.model_type,
    )
    write_results(strings, predictions, arguments.output)

    if arguments.output:
        print(
            f"Wrote {len(predictions)} classifications to {arguments.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
