#!/usr/bin/env python3
"""Download the exact frozen LitQA2 PDFs from recorded public source URLs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
MAX_PDF = 100 * 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def valid_pdf(path, expected):
    if not path.is_file():
        return False
    with path.open('rb') as handle:
        if handle.read(5) != b'%PDF-':
            return False
    return digest(path) == expected


def copy_bounded(source, target, limit):
    size = 0
    while block := source.read(1024 * 1024):
        size += len(block)
        if size > limit:
            raise ValueError(f'Download exceeds {limit} bytes')
        target.write(block)


def fetch_candidate(candidate, dest, expected, timeout):
    """Never extract archive paths onto disk or install unverified PDF bytes."""
    part = dest.with_suffix('.part')
    archive_path = dest.with_suffix('.archive.part')
    try:
        request = Request(candidate['url'], headers={'User-Agent': 'SHarP-corpus/1.0'})
        with urlopen(request, timeout=timeout) as response:
            if candidate.get('archive_member'):
                with archive_path.open('wb') as handle:
                    copy_bounded(response, handle, 2 * MAX_PDF)
            else:
                with part.open('wb') as handle:
                    copy_bounded(response, handle, MAX_PDF)
        if candidate.get('archive_member'):
            with tarfile.open(archive_path, 'r:*') as archive:
                member = archive.getmember(candidate['archive_member'])
                if not member.isfile() or not 0 < member.size <= MAX_PDF:
                    raise ValueError('Expected a bounded regular PDF archive member')
                with archive.extractfile(member) as source, part.open('wb') as handle:
                    copy_bounded(source, handle, MAX_PDF)
        if not valid_pdf(part, expected):
            raise ValueError(f'Source returned different bytes or a non-PDF response; expected {expected}, received {digest(part)}')
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)


def obtain(row, pool, check_only, timeout, retries):
    dest = pool / (row['sha256'] + '.pdf')
    result = {'sha256': row['sha256'], 'doi': row['doi'], 'title': row.get('title'),
              'version': row.get('version'), 'landing_page_url': row.get('landing_page_url'),
              'sources': row['sources']}
    if valid_pdf(dest, row['sha256']):
        return dict(result, status='verified')
    if dest.exists():
        return dict(result, status='failed', errors=[f'Existing file has wrong bytes: {dest}; preserve it outside this pool before retrying'])
    if check_only:
        return dict(result, status='missing')
    errors = []
    for candidate in row['sources']:
        if candidate.get('bundle'):
            errors.append('Licensed supplement was not imported; see the bundle error in the acquisition report')
            continue
        if candidate.get('access') == 'browser':
            errors.append(f'{candidate["url"]}: use the article page in a browser and import the downloaded PDF')
            continue
        for attempt in range(retries):
            try:
                fetch_candidate(candidate, dest, row['sha256'], timeout)
                return dict(result, status='downloaded', downloaded_from=candidate['url'])
            except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
                errors.append(f'{candidate["url"]}: {type(exc).__name__}: {exc}')
                if isinstance(exc, HTTPError) and exc.code == 429:
                    errors.append('Rate limited; retry later according to the provider Retry-After header: '
                                  + str(exc.headers.get('Retry-After', 'not specified')))
                    return dict(result, status='failed', errors=errors)
                if attempt + 1 < retries:
                    time.sleep(attempt + 1)
    return dict(result, status='failed', errors=errors)


def import_pdfs(directory, rows, pool):
    """Copy matching PDFs without modifying the user's download directory."""
    if not directory.is_dir():
        raise ValueError(f'Import directory does not exist: {directory}')
    expected = {row['sha256'] for row in rows}
    imported, unmatched = [], []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.pdf':
            continue
        key = digest(path)
        if key not in expected or not valid_pdf(path, key):
            unmatched.append({'file': str(path), 'sha256': key})
            continue
        dest = pool / (key + '.pdf')
        if dest.exists():
            if not valid_pdf(dest, key):
                raise ValueError(f'Existing pool file has wrong bytes: {dest}; preserved without overwrite')
        else:
            part = dest.with_suffix('.import.part')
            try:
                shutil.copyfile(path, part)
                if not valid_pdf(part, key):
                    raise ValueError(f'Import source changed while copying: {path}')
                part.replace(dest)
            finally:
                part.unlink(missing_ok=True)
        imported.append(key)
    return {'matched': sorted(set(imported)), 'unmatched': unmatched}


def prepare_bundles(manifest, rows, pool, timeout, local_bundle=None):
    """Fetch each licensed supplement once and import only verified members."""
    reports = []
    for bundle in manifest.get('pdf_bundles', []):
        needed = [(row, source) for row in rows for source in row['sources']
                  if source.get('bundle') == bundle['id'] and not valid_pdf(pool / (row['sha256'] + '.pdf'), row['sha256'])]
        if not needed:
            continue
        cache = pool / '.downloads'; cache.mkdir(exist_ok=True)
        archive_path = local_bundle or cache / bundle['filename']
        try:
            if not archive_path.is_file():
                if local_bundle:
                    raise ValueError(f'Supplement does not exist: {local_bundle}')
                part = archive_path.with_suffix('.part')
                try:
                    with urlopen(Request(bundle['url'], headers={'User-Agent': 'SHarP-corpus/1.0'}), timeout=timeout) as response, part.open('wb') as handle:
                        copy_bounded(response, handle, bundle['bytes'])
                    if digest(part) != bundle['sha256']:
                        raise ValueError('Supplement archive checksum mismatch')
                    part.replace(archive_path)
                finally:
                    part.unlink(missing_ok=True)
            if digest(archive_path) != bundle['sha256']:
                raise ValueError(f'Existing supplement checksum mismatch: {archive_path}')
            with tarfile.open(archive_path, 'r:gz') as archive:
                # Keep the supplement's attribution and per-document licenses.
                attribution = archive.getmember('ATTRIBUTION.md')
                if not attribution.isfile() or attribution.size > 1024 * 1024:
                    raise ValueError('Invalid supplement attribution file')
                (pool / (bundle['id'] + '-ATTRIBUTION.md')).write_bytes(archive.extractfile(attribution).read())
                for row, source in needed:
                    dest = pool / (row['sha256'] + '.pdf')
                    if dest.exists():
                        raise ValueError(f'Existing file has wrong bytes; preserved: {dest}')
                    member = archive.getmember(source['archive_member'])
                    if not member.isfile() or not 0 < member.size <= MAX_PDF:
                        raise ValueError('Invalid supplement PDF member')
                    part = dest.with_suffix('.part')
                    try:
                        with archive.extractfile(member) as content, part.open('wb') as handle:
                            copy_bounded(content, handle, MAX_PDF)
                        if not valid_pdf(part, row['sha256']):
                            raise ValueError(f'Supplement PDF hash mismatch: {row["doi"]}')
                        part.replace(dest)
                    finally:
                        part.unlink(missing_ok=True)
            reports.append({'bundle': bundle['id'], 'status': 'verified', 'imported': len(needed)})
        except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
            reports.append({'bundle': bundle['id'], 'url': bundle['url'], 'status': 'failed', 'error': str(exc)})
    return reports


def selected_documents(manifest, split):
    rows = manifest['documents']
    by_hash = {row['sha256']: row for row in rows}
    if len(by_hash) != len(rows):
        raise ValueError('Duplicate document hashes in acquisition manifest')
    expected = set()
    for name in ('train', 'validation') if split == 'all' else (split,):
        split_manifest = json.loads((ROOT / f'configs/splits/litqa2_{name}.json').read_text())
        if split_manifest.get('corpus_version') != manifest.get('corpus_version'):
            raise ValueError('Corpus version differs between source and split manifests')
        expected.update(split_manifest['corpus_document_sha256'])
    if expected - by_hash.keys():
        raise ValueError(f'Acquisition manifest lacks {len(expected - by_hash.keys())} frozen PDFs')
    selected = [by_hash[key] for key in sorted(expected)]
    for row in selected:
        if not row.get('doi') or not row.get('sources'):
            raise ValueError(f'Missing DOI/source URL for {row["sha256"]}')
        if any(not source.get('url', '').startswith('https://') for source in row['sources']):
            raise ValueError('Acquisition manifest must contain public HTTPS source URLs')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['train', 'validation', 'all'], default='all')
    p.add_argument('--pdf-pool', type=Path, default=ROOT / 'outputs/data/litqa2/pdfs')
    p.add_argument('--manifest', type=Path, default=ROOT / 'configs/litqa2_corpus.json')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--import-dir', type=Path, help='Copy exact matching PDFs from a browser-download directory before checking')
    p.add_argument('--supplement', type=Path, help='Use an already downloaded licensed PDF supplement archive')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--retries', type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.workers <= 8 or a.timeout < 1 or a.retries < 1:
        p.error('Use 1–8 workers and positive timeout/retries')
    try:
        manifest = json.loads(a.manifest.read_text())
        rows = selected_documents(manifest, a.split)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(1, f'Cannot prepare the frozen corpus: {exc}\n')
    a.pdf_pool.mkdir(parents=True, exist_ok=True)
    if a.import_dir:
        try:
            imported = import_pdfs(a.import_dir, rows, a.pdf_pool)
        except (OSError, ValueError) as exc:
            p.exit(1, f'Cannot import PDFs: {exc}\n')
        (a.pdf_pool / 'import-report.json').write_text(json.dumps(imported, indent=2) + '\n')
        print(f'Imported/matched {len(imported["matched"])} PDFs; unmatched files: {len(imported["unmatched"])}')
    report = a.pdf_pool / f'acquisition-{a.split}.json'
    bundles = prepare_bundles(manifest, rows, a.pdf_pool, a.timeout, a.supplement) if not a.check_only or a.supplement else []
    if a.check_only and not a.supplement and report.is_file():
        try:
            previous = json.loads(report.read_text())
        except (OSError, ValueError) as exc:
            print(f'Cannot read previous acquisition report: {exc}', file=sys.stderr)
            previous = {}
        # Keep unresolved download diagnostics; a local check is not a retry.
        for item in previous.get('bundles', []):
            if item.get('status') != 'failed':
                continue
            bundle_id = item.get('bundle')
            unresolved = any(
                source.get('bundle') == bundle_id
                and not valid_pdf(a.pdf_pool / (row['sha256'] + '.pdf'), row['sha256'])
                for row in rows for source in row['sources']
            )
            if unresolved:
                url = next((b['url'] for b in manifest.get('pdf_bundles', [])
                            if b['id'] == bundle_id), item.get('url', 'unknown URL'))
                bundles.append(dict(item, url=url, previous_attempt=True))
    for item in bundles:
        if item['status'] == 'failed':
            label = 'Previous supplement download failed' if item.get('previous_attempt') else 'Supplement download failed'
            print(f"{label}: {item['bundle']} ({item['url']}): {item['error']}", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda row: obtain(row, a.pdf_pool, a.check_only, a.timeout, a.retries), rows))
    missing = [row for row in results if row['status'] not in {'verified', 'downloaded'}]
    report.write_text(json.dumps({'split': a.split, 'expected': len(rows), 'verified': len(rows)-len(missing), 'bundles': bundles, 'missing': missing}, indent=2) + '\n')
    print(f'Verified {len(rows)-len(missing)}/{len(rows)} frozen PDFs. Report: {report}')
    if missing:
        p.exit(1, 'Corpus incomplete. The report lists each missing DOI, exact source URL, and expected hash. '
               'Retry downloads or obtain those exact PDFs via their source pages; do not replace the frozen corpus.\n')


if __name__ == '__main__':
    main()
