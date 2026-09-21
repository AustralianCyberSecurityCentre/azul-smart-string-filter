filename = input("Enter filename: ").strip()
search_string = input("Enter string to find (without quotes): ").strip()

try:
    found = False

    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip() == search_string:
                print(f"Found on line {line_number}: {line.rstrip()!r}")
                found = True

    if not found:
        print(f"{search_string!r} was not found.")

except FileNotFoundError:
    print(f"File not found: {filename}")
except OSError as error:
    print(f"Unable to read file: {error}")