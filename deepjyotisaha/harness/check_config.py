"""Check a G5 config: every setting must have a value and a source.

Usage:
    python harness/check_config.py [configs/config_v0.yaml] [--list]

For the config, the configs it inherits from and its side files (lr_schedule.yaml, data_mix.yaml), every
setting is put in one of three groups:
    ERROR    no value, or no "# source:" comment            -> must be fixed
    WAITING  the value is "open" and the source says who gives it and when (the size values TSAI gives
             on 31 Oct)                                    -> allowed in config v0
    OK       a value and a source
A section header (a key with settings under it) needs neither. The exit code is 1 when there are errors,
so the check can run before every commit. --list prints every setting, not only errors and waiting ones.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configlib import OPEN, load_settings_chain, read_settings, side_file_paths  # noqa: E402


def classify(s, parents):
    if not s.raw:
        # a key with settings under it is a section; a key with nothing under it has no value
        return ("SECTION", "") if s.path in parents else ("ERROR", "no value")
    problems = []
    if s.value is None or (isinstance(s.value, str) and not s.value.strip()):
        problems.append("no value")
    if not s.source:
        problems.append("no source")
    if problems:
        return "ERROR", " and ".join(problems)
    if s.value == OPEN:
        return "WAITING", s.source
    return "OK", ""


def check(path, show_all=False):
    path = Path(path).resolve()
    files = [path] + side_file_paths(path)
    counts = {"OK": 0, "WAITING": 0, "ERROR": 0}
    seen = set()
    rows = []
    for f in files:
        settings = load_settings_chain(f) if f == path else read_settings(f)
        parents = {s.path[:i] for s in settings for i in range(1, len(s.path))}
        # with inherits, a setting overridden by a later file counts once: the value that applies
        last = {s.path: i for i, s in enumerate(settings)}
        settings = [s for i, s in enumerate(settings) if last[s.path] == i]
        for s in settings:
            kind, why = classify(s, parents)
            if kind == "SECTION":
                continue
            key = (s.file, s.path)
            if key in seen:
                continue
            seen.add(key)
            counts[kind] += 1
            rows.append((kind, s, why))
    print(f"Checking {path.name}" + (f" (+ {', '.join(f.name for f in files[1:])})" if files[1:] else ""))
    for kind, s, why in rows:
        if show_all or kind != "OK":
            where = f"{s.file.name}:{s.line}"
            print(f"  {kind:7s} {'.'.join(s.path):40s} {where:24s} {why}")
    print(f"Result: {counts['OK']} ok, {counts['WAITING']} waiting (open, with a source), {counts['ERROR']} errors")
    return counts["ERROR"]


def main():
    ap = argparse.ArgumentParser(description="Every setting needs a value and a # source: comment.")
    ap.add_argument("config", nargs="?", default=str(Path(__file__).resolve().parents[1] / "configs" / "config_v0.yaml"))
    ap.add_argument("--list", action="store_true", help="print every setting, not only errors and waiting ones")
    a = ap.parse_args()
    sys.exit(1 if check(a.config, a.list) else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
