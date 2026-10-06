#!/usr/bin/env python3
"""Acquire benchmark inputs and verify the frozen SHarP task sets before inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'configs/data_sources.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_ids(name):
    data = json.loads((ROOT / f'configs/splits/{name}.json').read_text())
    return {str(item) for values in data.values() for item in values}


def require_ids(wanted, available, label):
    missing = wanted - available
    if missing:
        raise ValueError(f'{label}: missing {len(missing)} frozen task IDs: {sorted(missing)}')


def check_bundled(target):
    """The data is tracked in the same pinned Git commit as the harness."""
    up = ROOT / 'external' / target
    pin = json.loads((ROOT / 'configs/upstreams.json').read_text())[target]['commit']
    prefixes = (['TauBench/data/tau2/domains/airline', 'TauBench/data/tau2/domains/retail',
                 'TauBench/data/tau2/user_simulator'] if target == 'life' else
                ['dataset/deepplanning_shopping/data', *[f'dataset/deepplanning_shopping/database_level{i}' for i in (1, 2, 3)]])
    tracked = subprocess.check_output(['git', '-C', str(up), 'ls-tree', '-r', '--name-only', pin, '--', *prefixes], text=True).splitlines()
    if not tracked:
        raise ValueError(f'No tracked {target} dataset files; rerun setup.sh {target}')
    # Unlike checking only directory existence, this catches removed/modified records and databases.
    subprocess.run(['git', '-C', str(up), 'diff', '--exit-code', pin, '--', *prefixes], check=True, stdout=subprocess.DEVNULL)
    for name in tracked:
        if not (up / name).is_file():
            raise ValueError(f'Missing tracked dataset file: {up / name}')
    if target == 'life':
        counts = {}
        for domain in ('airline', 'retail'):
            rows = json.loads((up / f'TauBench/data/tau2/domains/{domain}/tasks.json').read_text())
            wanted = frozen_ids(f'tau2_{domain}')
            require_ids(wanted, {str(r['id']) for r in rows}, domain)
            counts[domain] = len(wanted)
    else:
        base = up / 'dataset/deepplanning_shopping'
        available = set()
        for level in (1, 2, 3):
            for row in json.loads((base / f'data/level_{level}_query_meta.json').read_text()):
                case = base / f'database_level{level}/case_{row["id"]}'
                if all((case / name).is_file() for name in ('products.jsonl', 'user_info.json', 'cart.json', 'validation_cases.json')):
                    available.add(f'level{level}_case{row["id"]}')
        wanted = frozen_ids('deepplanning_shopping')
        require_ids(wanted, available, 'Shopping')
        counts = {'shopping': len(wanted)}
    return {'source_commit': pin, 'verified_files': len(tracked), 'frozen_tasks': counts}


def download_verified(url, dest, expected):
    if dest.exists():
        if sha256(dest) != expected:
            raise ValueError(f'Existing file differs from pinned source: {dest}; preserve it and use a new output directory')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    try:
        with urlopen(Request(url, headers={'User-Agent': 'SHarP-data/1.0'}), timeout=60) as response, part.open('wb') as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        if sha256(part) != expected:
            raise ValueError(f'Checksum mismatch from {url}; no dataset file installed')
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def prepare_litqa_records(output, check_only):
    source = json.loads(SOURCES.read_text())['litqa2_records']
    dest = output / 'litqa-v2-public.jsonl'
    if not check_only:
        download_verified(source['url'], dest, source['sha256'])
    if not dest.is_file() or sha256(dest) != source['sha256']:
        raise ValueError('Missing or changed LitQA2 records; run prepare_data.py litqa2')
    rows = [json.loads(line) for line in dest.read_text().splitlines() if line.strip()]
    ids = {str(r['id']) for r in rows}
    if len(ids) != len(rows):
        raise ValueError('Duplicate LitQA2 IDs')
    counts = {}
    for split in ('train', 'validation'):
        wanted = set(json.loads((ROOT / f'configs/splits/litqa2_{split}.json').read_text())['question_ids'])
        require_ids(wanted, ids, f'LitQA2 {split}')
        counts[split] = len(wanted)
    return {'records': str(dest), 'sha256': source['sha256'], 'frozen_tasks': counts}


def check_gaia(root):
    """Validate all selected questions and attachments, using only the local snapshot."""
    import pyarrow.parquet as pq
    metadata = root / '2023/validation/metadata.parquet'
    if not metadata.is_file():
        raise ValueError('GAIA is not prepared. Run: python scripts/prepare_data.py gaia')
    rows = pq.read_table(metadata).to_pylist()
    records = {str(r['task_id']): r for r in rows}
    if len(records) != len(rows):
        raise ValueError('Duplicate GAIA IDs')
    wanted = frozen_ids('gaia')
    require_ids(wanted, set(records), 'GAIA')
    attachments = set()
    for task in wanted:
        name = records[task].get('file_name')
        if name:
            attachment = (root / '2023/validation' / name).resolve()
            if not attachment.is_relative_to(root.resolve()) or not attachment.is_file() or not attachment.stat().st_size:
                raise ValueError(f'Missing GAIA attachment for {task}: {name}; rerun prepare_data.py gaia')
            attachments.add(str(attachment.relative_to(root.resolve())))
    return {'frozen_tasks': len(wanted), 'attachments': len(attachments),
            'files_sha256': {n: sha256(root / n) for n in ['2023/validation/metadata.parquet', *sorted(attachments)]}}


def prepare_gaia(root, check_only):
    source = json.loads(SOURCES.read_text())['gaia']
    if not check_only:
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(source['repo_id'], repo_type='dataset', revision=source['revision'],
                              allow_patterns=['README.md', '2023/validation/*'], local_dir=str(root))
        except Exception as exc:
            raise RuntimeError('GAIA download failed. Accept the dataset terms at '
                               'https://huggingface.co/datasets/gaia-benchmark/GAIA, then run `hf auth login` '
                               'with a read token from that same approved account and retry. '
                               f'Underlying error: {type(exc).__name__}') from exc
    result = check_gaia(root)
    marker = root / 'sharp-data.json'
    if check_only:
        if not marker.is_file():
            raise ValueError('Missing GAIA snapshot verification record; run prepare_data.py gaia')
        prior = json.loads(marker.read_text())
        if prior.get('revision') != source['revision'] or prior.get('files_sha256') != result['files_sha256']:
            raise ValueError('GAIA snapshot changed since preparation; rerun prepare_data.py gaia')
    else:
        marker.write_text(json.dumps(dict(result, revision=source['revision']), indent=2) + '\n')
        marker.chmod(0o600)
    return dict(result, revision=source['revision'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('target', choices=['life', 'shopping', 'litqa2', 'gaia'])
    p.add_argument('--check-only', action='store_true', help='Validate existing files without network access')
    p.add_argument('--output', type=Path, help='Download directory for litqa2 or gaia')
    a = p.parse_args()
    try:
        if a.target in ('life', 'shopping'):
            if a.output:
                p.error('--output only applies to litqa2 and gaia')
            if not a.check_only:
                subprocess.run([sys.executable, str(ROOT / 'scripts/fetch_upstream.py'), a.target], check=True)
            result = check_bundled(a.target)
        elif a.target == 'litqa2':
            result = prepare_litqa_records(a.output or ROOT / 'outputs/data/litqa2', a.check_only)
        else:
            result = prepare_gaia((a.output or ROOT / 'external/gaia/benchmarks/gaia/data').resolve(), a.check_only)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(1, f'{exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
