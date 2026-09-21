input_file = "good_elf.txt"
output_file = "good_elf_deduped.txt"

seen = set()

with open(input_file, "r", encoding="utf-8", errors="ignore") as infile, \
     open(output_file, "w", encoding="utf-8", newline="\n") as outfile:

    for line in infile:
        line = line.strip()

        if not line:
            continue

        if line not in seen:
            seen.add(line)
            outfile.write(line + "\n")

print(f"Removed duplicates. Output written to: {output_file}")
print(f"Unique lines: {len(seen)}")