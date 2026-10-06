"""Data preparation must reject incomplete inputs before any model run."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('sharp_prepare_data', Path(__file__).resolve().parents[1] / 'scripts/prepare_data.py')
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


class DataPreparationTests(unittest.TestCase):
    def test_existing_download_requires_exact_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'records.jsonl'; dest.write_bytes(b'changed')
            with patch.object(data, 'urlopen') as network:
                with self.assertRaisesRegex(ValueError, 'differs'):
                    data.download_verified('https://example.org/unused', dest, '0' * 64)
                network.assert_not_called()
                self.assertEqual(dest.read_bytes(), b'changed')

    def test_gaia_missing_attachment_and_snapshot_tampering(self):
        parquet = ModuleType('pyarrow.parquet')
        rows = [{'task_id': 'task-a', 'file_name': 'attachment.txt'}]
        parquet.read_table = lambda _: SimpleNamespace(to_pylist=lambda: rows)
        arrow = ModuleType('pyarrow'); arrow.parquet = parquet
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {'pyarrow': arrow, 'pyarrow.parquet': parquet}), patch.object(data, 'frozen_ids', return_value={'task-a'}):
            root = Path(td); split = root / '2023/validation'; split.mkdir(parents=True)
            (split / 'metadata.parquet').write_bytes(b'fixture metadata')
            with self.assertRaisesRegex(ValueError, 'Missing GAIA attachment'):
                data.check_gaia(root)
            attachment = split / 'attachment.txt'; attachment.write_text('original')
            report = data.check_gaia(root)
            source = json.loads(data.SOURCES.read_text())['gaia']
            (root / 'sharp-data.json').write_text(json.dumps(dict(report, revision=source['revision'])))
            self.assertEqual(data.prepare_gaia(root, True)['attachments'], 1)
            attachment.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed since preparation'):
                data.prepare_gaia(root, True)
            rows[0]['file_name'] = '../../../outside.txt'
            with self.assertRaisesRegex(ValueError, 'Missing GAIA attachment'):
                data.check_gaia(root)

    def test_missing_task_ids_fail(self):
        with self.assertRaisesRegex(ValueError, 'missing 1'):
            data.require_ids({'one', 'two'}, {'one'}, 'fixture')
