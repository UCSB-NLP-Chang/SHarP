#!/usr/bin/env python3
"""Fetch and verify an immutable upstream checkout with the release patch applied."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(dest, *args):
    return subprocess.check_output(['git', '-C', str(dest), *args], text=True).strip()


def main():
    pins = json.loads((ROOT / 'configs/upstreams.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', choices=list(pins))
    a = p.parse_args()
    pin, dest = pins[a.name], ROOT / 'external' / a.name
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        subprocess.run(['git', 'clone', '--no-checkout', pin['url'], str(dest)], check=True)
        git(dest, 'checkout', '--detach', pin['commit'])
    if git(dest, 'rev-parse', 'HEAD') != pin['commit']:
        raise SystemExit('Existing upstream revision differs; preserve it and use a fresh checkout.')
    dirty = git(dest, 'status', '--porcelain', '--untracked-files=no', '--ignore-submodules=all')
    if pin.get('patch') and not dirty:
        git(dest, 'apply', '--check', str(ROOT / pin['patch']))
        git(dest, 'apply', str(ROOT / pin['patch']))
    expected = pin.get('patched_files', {})
    changed = set(git(dest, 'diff', 'HEAD', '--name-only', '--ignore-submodules=all').splitlines())
    if changed != set(expected):
        raise SystemExit(f'Unexpected upstream changes in {dest}; refusing to overwrite them.')
    for rel, digest in expected.items():
        if hashlib.sha256((dest / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f'Patched file differs: {dest / rel}')
    if pin.get('submodule'):
        sub = pin['submodule']
        subdest = dest / sub['path']
        if (subdest / '.git').exists():
            if git(subdest, 'rev-parse', 'HEAD') != sub['commit'] or git(subdest, 'status', '--porcelain', '--untracked-files=no'):
                raise SystemExit('Existing SDK checkout differs; preserve it and use a fresh checkout.')
        git(dest, 'submodule', 'update', '--init', '--recursive', sub['path'])
        if git(dest / sub['path'], 'rev-parse', 'HEAD') != sub['commit']:
            raise SystemExit('SDK submodule revision differs')
    print(f"{a.name}: {pin['commit']} (release patches verified)")


if __name__ == '__main__':
    main()
