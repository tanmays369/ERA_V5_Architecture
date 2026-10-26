"""Report settings with no value or no inline '# source:' comment.

Requires PyYAML. Values are ordinary YAML defaults, not {default, source} wrappers.
Default config resolves relative to this file, independent of working directory.
Size nulls are deliberately reported, not silently exempted.
Exit 0: no missing fields. Exit 1: missing values/sources. Exit 2: invalid config.
"""
import argparse
import json
import pathlib
import re
import sys
import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

SOURCE = re.compile(r'^\s*#\s*source\s*:\s*(.*?)\s*$', re.IGNORECASE)


def audit_text(text):
    tree = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(tree, MappingNode):
        raise ValueError('Config must be a YAML mapping')
    lines = text.splitlines()
    rows = []

    def source_for(key, value):
        # Search after the parsed YAML value, never inside a quoted string.
        if value.end_mark.line == key.start_mark.line:
            tail = lines[key.start_mark.line][value.end_mark.column:]
        else:
            # Container settings require a source beside their key, not on a child.
            tail = lines[key.start_mark.line][key.end_mark.column:]
            tail = tail[tail.find('#'):] if '#' in tail else ''
        match = SOURCE.fullmatch(tail)
        return match.group(1).strip() if match else ''

    def walk(node, prefix='', ancestors=frozenset()):
        if id(node) in ancestors:
            raise ValueError('Recursive YAML aliases are not supported')
        if not isinstance(node, MappingNode):
            raise ValueError('Internal traversal error')
        seen = set()
        for key, value in node.value:
            if not isinstance(key, ScalarNode) or key.tag != 'tag:yaml.org,2002:str':
                raise ValueError('Setting keys must be strings; YAML merges are not supported')
            if key.value in seen:
                raise ValueError('Duplicate setting: ' + (prefix + '.' if prefix else '') + key.value)
            seen.add(key.value)
            name = prefix + '.' + key.value if prefix else key.value
            if isinstance(value, MappingNode) and value.value:
                walk(value, name, ancestors | {id(node)})
                continue
            source = source_for(key, value)
            missing_value = (
                isinstance(value, ScalarNode) and (
                    value.tag == 'tag:yaml.org,2002:null' or
                    value.tag == 'tag:yaml.org,2002:str' and not value.value.strip()
                ) or isinstance(value, (MappingNode, SequenceNode)) and not value.value
            )
            issues = []
            if missing_value:
                issues.append('MISSING_VALUE')
            if not source:
                issues.append('MISSING_SOURCE')
            if isinstance(value, SequenceNode):
                # A list is one setting. Null/empty list entries are also missing values.
                if any(isinstance(x, ScalarNode) and (x.tag == 'tag:yaml.org,2002:null' or not x.value.strip()) for x in value.value):
                    if 'MISSING_VALUE' not in issues:
                        issues.append('MISSING_VALUE')
            rows.append(dict(setting=name, line=key.start_mark.line + 1,
                             source=source, issues=issues))

    walk(tree)
    if not rows:
        raise ValueError('Config contains no settings')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / 'configs/config_v0.yaml')
    parser.add_argument('--json', action='store_true', help='Print all setting statuses as JSON')
    args = parser.parse_args(argv)
    try:
        rows = audit_text(args.config.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print('CONFIG_ERROR: ' + str(exc), file=sys.stderr)
        return 2
    missing = [row for row in rows if row['issues']]
    if args.json:
        print(json.dumps(dict(config=str(args.config), total_settings=len(rows),
                              settings_with_issues=len(missing), settings=rows), indent=2))
    else:
        for row in missing:
            print(f"{row['setting']} (line {row['line']}): {', '.join(row['issues'])}")
        print(f'Checked {len(rows)} settings; {len(missing)} settings with missing values or sources.')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
