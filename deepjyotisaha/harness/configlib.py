"""Reading the G5 config files.

The configs are YAML with one rule added: every setting carries a "# source:" comment on its own line.
PyYAML drops comments, so this module reads each file twice: once line by line to find every setting, its
raw value and its source comment, and once through yaml.safe_load for the values themselves.

Two more features:
  inherits:  a config may start from another one ("inherits: ../config_v0.yaml") and override some settings,
             which is how the tiny test configs reuse every default and source of config v0.
  side files: optimizer.lr_schedule_file and training.data_mix_file are loaded into the result under
             "lr_schedule" and "data_mix", as V4 kept its schedule and data mix in separate files.

Paths inside a config (settings whose name ends in _file, _dir, _pack, _text or is "path") are resolved
relative to the file that sets them.
"""
import copy
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

OPEN = "open"
PATH_SUFFIXES = ("_file", "_dir", "_pack", "_text")
SIDE_FILES = {("optimizer", "lr_schedule_file"): "lr_schedule", ("training", "data_mix_file"): "data_mix"}

_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+):(?P<rest>.*)$")


@dataclass
class Setting:
    path: tuple          # ("model", "hidden_size")
    raw: str             # the value as written, "" for a section header
    value: object        # parsed with yaml.safe_load
    source: str          # text after "# source:", "" when missing
    file: Path
    line: int


def _split_comment(rest):
    """'  2048   # source: S14' -> ('2048', '# source: S14'). A '#' inside quotes or brackets is kept."""
    quote, depth = None, 0
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "#" and depth == 0 and (i == 0 or rest[i - 1].isspace()):
            return rest[:i].strip(), rest[i:].strip()
    return rest.strip(), ""


def read_settings(path):
    """Every setting in one file, in order, as Setting objects. Section headers have raw == ""."""
    path = Path(path)
    out, stack = [], []  # stack of (indent, key)
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KEY_LINE.match(line)
        if not m:
            raise ValueError(f"{path}:{n}: cannot read this line (one 'key: value  # source: ...' per line): {line!r}")
        indent = len(m.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        raw, comment = _split_comment(m.group("rest"))
        keypath = tuple(k for _, k in stack) + (m.group("key"),)
        source = ""
        sm = re.match(r"#\s*source:\s*(.*)", comment)
        if sm:
            source = sm.group(1).strip()
        value = yaml.safe_load(raw) if raw else None
        out.append(Setting(keypath, raw, value, source, path, n))
        if not raw:
            stack.append((indent, m.group("key")))
    return out


def _is_path_key(key):
    return key == "path" or key.endswith(PATH_SUFFIXES)


def _set(tree, keypath, value):
    for k in keypath[:-1]:
        tree = tree.setdefault(k, {})
    tree[keypath[-1]] = value


def _get(tree, keypath, default=None):
    for k in keypath:
        if not isinstance(tree, dict) or k not in tree:
            return default
        tree = tree[k]
    return tree


def load_settings_chain(path):
    """All settings of a config and of the configs it inherits from, base first, as (settings, files)."""
    path = Path(path).resolve()
    own = read_settings(path)
    inherit = [s for s in own if s.path == ("inherits",)]
    chain = []
    if inherit:
        chain = load_settings_chain((path.parent / inherit[0].value).resolve())
    return chain + [s for s in own if s.path != ("inherits",)]


def load_config(path):
    """The merged config as a plain dict, with paths resolved and side files loaded."""
    tree = {}
    for s in load_settings_chain(path):
        if not s.raw:
            continue
        value = s.value
        if isinstance(value, str) and value != OPEN and _is_path_key(s.path[-1]):
            value = str((s.file.parent / value).resolve())
        _set(tree, s.path, value)
    for keypath, name in SIDE_FILES.items():
        side = _get(tree, keypath)
        if side and side != OPEN:
            tree[name] = load_config(side)
    return tree


def side_file_paths(path):
    """The side files a config refers to (resolved), for the config check."""
    cfg = {}
    for s in load_settings_chain(path):
        if s.raw and s.path in SIDE_FILES:
            cfg[s.path] = (s.file.parent / s.value).resolve()
    return list(cfg.values())


def require(cfg, *keypath):
    """A setting that must have a real value to train; 'open' or missing stops with a clear message."""
    value = _get(cfg, keypath)
    if value is None or value == OPEN:
        raise SystemExit(f"Setting {'.'.join(keypath)} is {value!r}. A training config must give it a value "
                         f"(the tiny configs in configs/tiny/ do).")
    return value


def get(cfg, *keypath, default=None):
    return _get(cfg, keypath, default)


def deep_copy(cfg):
    return copy.deepcopy(cfg)
