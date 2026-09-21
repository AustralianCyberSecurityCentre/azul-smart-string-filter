import re
from wordfreq import zipf_frequency

URL_RE = re.compile(
    r'https?://|www\.|[a-zA-Z0-9-]+\.(com|net|org|io|biz)',
    re.I
)

FILE_RE = re.compile(
    r'.*\.(dll|exe|sys|bat|cmd|ps1|vbs|js)$',
    re.I
)


def contains_english_word(s):
    words = re.findall(r"[A-Za-z]{4,}", s.lower())

    for word in words:
        # Higher = more common English word
        if zipf_frequency(word, "en") >= 3.0:
            return True

    return False


def is_human_readable(s):
    s = s.strip()

    if len(s) < 4:
        return False

    # Always keep obvious useful strings
    if URL_RE.search(s):
        return True

    if FILE_RE.search(s):
        return True

    alpha = sum(c.isalpha() for c in s)

    if alpha < 4:
        return False

    vowels = sum(
        c.lower() in "aeiou"
        for c in s
        if c.isalpha()
    )

    vowel_ratio = vowels / alpha

    if vowel_ratio < 0.20:
        return False

    # Must contain at least one English word
    if not contains_english_word(s):
        return False

    # Throw away opcode-looking junk
    if re.fullmatch(r"[A-Za-z0-9|$@;:+!%&_\\-]+", s):
        if len(s) < 20:
            return False

    return True


input_file = "all_strings_elf.txt"

good_count = 0
bad_count = 0

# change the input/ouput filenames to whatever you require
with open(input_file, "r", encoding="utf-8", errors="ignore") as infile, \
     open("good_elf.txt", "w", encoding="utf-8") as good, \
     open("bad_elf.txt", "w", encoding="utf-8") as bad:

    for line in infile:
        line = line.rstrip("\r\n")

        if is_human_readable(line):
            good.write(line + "\n")
            good_count += 1
        else:
            bad.write(line + "\n")
            bad_count += 1

print(f"Human-readable strings: {good_count}")
print(f"Other strings: {bad_count}")