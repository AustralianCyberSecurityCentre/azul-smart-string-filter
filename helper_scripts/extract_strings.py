from pathlib import Path
from cart import unpack_stream
from io import BytesIO
from binary2strings import extract_all_strings


INPUT_FOLDER = "azul_files_elf"
OUTPUT_FILE = "all_strings_elf.txt"

input_path = Path(INPUT_FOLDER)

with open(OUTPUT_FILE, "w", encoding="utf-8", errors="ignore") as out_f:

    for file_path in sorted(input_path.iterdir()):

        if not file_path.is_file():
            continue

        print(f"Processing: {file_path.name}")

        try:
            # De-CART
            cart_buffer = BytesIO()

            with open(file_path, "rb") as infile:
                unpack_stream(infile, cart_buffer)

            binary_data = cart_buffer.getvalue()

            # Extract strings
            strings = extract_all_strings(binary_data)

            # Write results
            out_f.write("=" * 80 + "\n")
            out_f.write(f"FILE: {file_path.name}\n")
            out_f.write("=" * 80 + "\n\n")

            for s in strings:
                out_f.write(f"{s[0]}\n")

            out_f.write("\n\n")

        except Exception as e:
            print(f"Failed: {file_path.name} -> {e}")

print(f"\nFinished. Results written to {OUTPUT_FILE}")