"""Regression checks for building a fresh PaperQA corpus without pre-existing caches."""
import asyncio
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1] / 'experiments/paperqa2'
sys.path.insert(0, str(CODE))
import run_validation_ladder as runner


class FreshCorpusTests(unittest.TestCase):
    def test_full_index_is_built_on_first_run_and_reused_on_resume(self):
        calls = []
        search = ModuleType('paperqa.agents.search')
        async def get_index(*, settings, build):
            calls.append(build)
        search.get_directory_index = get_index
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'corpus').mkdir(); (root/'metadata').mkdir()
            (root/'corpus/paper.pdf').write_bytes(b'test-only corpus fixture')
            args = SimpleNamespace(corpus_dir=root/'corpus', preprocess_dir=root/'metadata', run_tag='fresh')
            settings = SimpleNamespace(md5='test-settings', agent=SimpleNamespace(index=SimpleNamespace(name='fresh-full-parsing-paperqa2')))
            with patch.dict(sys.modules, {'paperqa.agents.search': search}), patch.object(runner, 'make_substrate_settings', return_value=settings), patch.object(runner, 'validate_index_artifacts', return_value={}):
                asyncio.run(runner.prepare_indexes(args, ('A18',)))
                asyncio.run(runner.prepare_indexes(args, ('A18',)))
            self.assertEqual(calls, [True, False])

    def test_all_pruned_substrate_and_config(self):
        import validation_ladder as ladder
        self.assertEqual(ladder.kept_components('A0'), ())
        self.assertEqual(ladder.index_label('A0'), 'low-parsing')
        self.assertEqual(len(ladder.disabled_components('A0')), 18)


if __name__ == '__main__':
    unittest.main()
