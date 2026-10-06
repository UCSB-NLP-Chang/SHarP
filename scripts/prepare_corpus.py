#!/usr/bin/env python3
"""Copy the exact frozen LitQA2 corpus from a user-supplied PDF pool."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf-pool', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation'], required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory for the selected PDF bytes')
    a = p.parse_args()
    expected = set(json.loads((ROOT/f'configs/splits/litqa2_{a.split}.json').read_text())['corpus_document_sha256'])
    found = {}
    for file in a.pdf_pool.rglob('*'):
        if file.is_file() and file.suffix.lower() == '.pdf':
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in expected:
                found.setdefault(digest, file)
    missing = expected - found.keys()
    if missing:
        p.error(f'Missing {len(missing)} of {len(expected)} frozen PDFs; no output created. Missing hashes: {sorted(missing)}')
    a.output.mkdir(parents=True, exist_ok=False)
    for digest, src in sorted(found.items()):
        shutil.copyfile(src, a.output/f'{digest}.pdf')
    print(f'Prepared {len(found)} PDFs with exact frozen content hashes: {a.output}')


if __name__ == '__main__':
    main()
