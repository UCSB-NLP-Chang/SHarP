"""Regression checks for exact-byte corpus acquisition and archive handling."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('sharp_corpus_download', Path(__file__).resolve().parents[1] / 'scripts/download_litqa_corpus.py')
corpus = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(corpus)


class CorpusDownloadTests(unittest.TestCase):
    def test_supplement_is_downloaded_once_and_retains_attribution(self):
        payload = io.BytesIO(); rows = []
        with tarfile.open(fileobj=payload, mode='w:gz') as archive:
            for name, data in [('ATTRIBUTION.md', b'Individual CC licenses'),
                               ('pdfs/one.pdf', b'%PDF-1.4\none'), ('pdfs/two.pdf', b'%PDF-1.4\ntwo')]:
                member = tarfile.TarInfo(name); member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
                if name.endswith('.pdf'):
                    rows.append({'doi': name, 'sha256': hashlib.sha256(data).hexdigest(),
                                 'sources': [{'bundle': 'supplement', 'archive_member': name}]})
        content = payload.getvalue()
        manifest = {'pdf_bundles': [{'id': 'supplement', 'filename': 'supplement.tar.gz',
                    'url': 'https://example.org/supplement.tar.gz', 'bytes': len(content),
                    'sha256': hashlib.sha256(content).hexdigest()}]}
        with tempfile.TemporaryDirectory() as td:
            pool = Path(td)
            with patch.object(corpus, 'urlopen', return_value=io.BytesIO(content)) as network:
                result = corpus.prepare_bundles(manifest, rows, pool, 1)
                self.assertEqual(result[0]['imported'], 2)
                network.assert_called_once()
            self.assertEqual((pool / 'supplement-ATTRIBUTION.md').read_text(), 'Individual CC licenses')
            with patch.object(corpus, 'urlopen') as network:
                self.assertEqual(corpus.prepare_bundles(manifest, rows, pool, 1), [])
                network.assert_not_called()
        with tempfile.TemporaryDirectory() as td:
            pool = Path(td)
            with patch.object(corpus, 'urlopen', return_value=io.BytesIO(b'bad archive')):
                result = corpus.prepare_bundles(manifest, rows, pool, 1)
            self.assertEqual(result[0]['status'], 'failed')
            self.assertEqual(list(pool.glob('*.pdf')), [])

    def test_browser_import_matches_content_and_preserves_sources(self):
        pdf = b'%PDF-1.4\nimport fixture'; expected = hashlib.sha256(pdf).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); downloads = root / 'downloads'; downloads.mkdir()
            pool = root / 'pool'; pool.mkdir()
            source = downloads / 'arbitrary publisher name.PDF'; source.write_bytes(pdf)
            (downloads / 'wrong.pdf').write_bytes(b'%PDF-1.4\nwrong version')
            result = corpus.import_pdfs(downloads, [{'sha256': expected}], pool)
            self.assertEqual(result['matched'], [expected])
            self.assertEqual(len(result['unmatched']), 1)
            self.assertEqual(source.read_bytes(), pdf)
            self.assertEqual((pool / (expected + '.pdf')).read_bytes(), pdf)
            corpus.import_pdfs(downloads, [{'sha256': expected}], pool)

    def test_browser_only_sources_never_trigger_automated_requests(self):
        row = {'doi': '10.example/browser', 'sha256': '0' * 64,
               'sources': [{'url': 'https://example.org/article', 'access': 'browser'}]}
        with tempfile.TemporaryDirectory() as td, patch.object(corpus, 'urlopen') as network:
            result = corpus.obtain(row, Path(td), False, 1, 1)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('browser', result['errors'][0])
            network.assert_not_called()

    def test_exact_pdf_download_and_offline_resume(self):
        pdf = b'%PDF-1.4\nfixture'; expected = hashlib.sha256(pdf).hexdigest()
        row = {'doi': '10.example/fixture', 'sha256': expected, 'sources': [{'url': 'https://example.org/paper.pdf'}]}
        with tempfile.TemporaryDirectory() as td:
            pool = Path(td)
            with patch.object(corpus, 'urlopen', return_value=io.BytesIO(pdf)):
                self.assertEqual(corpus.obtain(row, pool, False, 1, 1)['status'], 'downloaded')
            with patch.object(corpus, 'urlopen') as network:
                self.assertEqual(corpus.obtain(row, pool, True, 1, 1)['status'], 'verified')
                network.assert_not_called()
            (pool / (expected + '.pdf')).write_bytes(b'corrupt existing file')
            with patch.object(corpus, 'urlopen') as network:
                self.assertEqual(corpus.obtain(row, pool, False, 1, 1)['status'], 'failed')
                network.assert_not_called()

    def test_wrong_bytes_are_not_installed(self):
        with tempfile.TemporaryDirectory() as td, patch.object(corpus, 'urlopen', return_value=io.BytesIO(b'<html>login</html>')):
            target = Path(td) / 'paper.pdf'
            with self.assertRaisesRegex(ValueError, 'different bytes'):
                corpus.fetch_candidate({'url': 'https://example.org/paper'}, target, '0' * 64, 1)
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_archive_reads_only_named_pdf_without_extracting_paths(self):
        pdf = b'%PDF-1.4\narchive fixture'; expected = hashlib.sha256(pdf).hexdigest()
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode='w:gz') as archive:
            entry = tarfile.TarInfo('../../unrelated.txt'); entry.size = 3
            archive.addfile(entry, io.BytesIO(b'bad'))
            entry = tarfile.TarInfo('article/main.pdf'); entry.size = len(pdf)
            archive.addfile(entry, io.BytesIO(pdf))
        with tempfile.TemporaryDirectory() as td, patch.object(corpus, 'urlopen', return_value=io.BytesIO(payload.getvalue())):
            target = Path(td) / 'paper.pdf'
            corpus.fetch_candidate({'url': 'https://example.org/article.tgz', 'archive_member': 'article/main.pdf'}, target, expected, 1)
            self.assertEqual(target.read_bytes(), pdf)
            self.assertEqual(list(Path(td).iterdir()), [target])
