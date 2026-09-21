from pathlib import Path

good = {
    line.strip()
    for line in Path("azul_smart_string_filter/good_win.txt").read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()
    if line.strip()
}

bad = {
    line.strip()
    for line in Path("azul_smart_string_filter/bad_win.txt").read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()
    if line.strip()
}

for value in sorted(good & bad):
    print(repr(value))
PY