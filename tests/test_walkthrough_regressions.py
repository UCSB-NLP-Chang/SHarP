"""Failure-path regressions from the independent README walkthrough."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WalkthroughTests(unittest.TestCase):
    def test_embedding_cpu_overrides_inherited_gpu_and_opt_in_preserves_it(self):
        runner = ModuleType('run_single_off')
        seen = []
        runner.parse_args = lambda: (seen.append(os.environ.get('CUDA_VISIBLE_DEVICES')) or SimpleNamespace())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); tasks = root / 'tasks.jsonl'; tasks.write_text('{}\n')
            corpus = root / 'corpus'; corpus.mkdir()
            for extra, expected in [([], ''), (['--gpu-embeddings'], '3')]:
                argv = ['run_paperqa', '--phase', 'screen', '--tasks', str(tasks), '--corpus', str(corpus),
                        '--output', str(root/'out'), '--allow-custom-corpus', '--dry-run', *extra]
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), patch.dict(sys.modules, {'run_single_off': runner}), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    load_script('run_paperqa').main()
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], expected)
                self.assertEqual(seen[-1], expected)

    def test_embedding_settings_keep_cpu_default_and_support_gpu_opt_in(self):
        sys.path.insert(0, str(ROOT/'experiments/paperqa2'))
        import run_full
        def settings():
            return SimpleNamespace(agent=SimpleNamespace(index=SimpleNamespace()), parsing=SimpleNamespace())
        paperqa = ModuleType('paperqa')
        paperqa.Settings = SimpleNamespace(from_name=lambda _: settings())
        args = SimpleNamespace(model='fixture', api_base='http://localhost:8000/v1', api_key='EMPTY',
                               max_output_tokens=16384, embedding='st-fixture', corpus_dir=ROOT,
                               index_dir=ROOT, index_name='fixture', off_component=None)
        with patch.dict(sys.modules, {'paperqa': paperqa}):
            for flag, device in [('0', 'cpu'), ('1', 'cuda')]:
                with patch.dict(os.environ, {'SHARP_GPU_EMBEDDINGS': flag}):
                    self.assertEqual(run_full.make_settings(args).embedding_config, {'device': device})

    def test_bundle_failure_is_visible_and_survives_check_then_clears_after_import(self):
        module = load_script('download_litqa_corpus')
        pdf = b'%PDF-1.4\nfixture'
        import hashlib
        key = hashlib.sha256(pdf).hexdigest()
        url = 'https://example.org/supplement.tar.gz'
        rows = [{'doi': '10.test/paper', 'sha256': key,
                 'sources': [{'url': url, 'bundle': 'supplement', 'archive_member': 'pdfs/a.pdf'}]}]
        manifest = {'pdf_bundles': [{'id': 'supplement', 'url': url, 'filename': 'supplement.tar.gz',
                                    'bytes': 10, 'sha256': '0'*64}]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mf = root/'manifest.json'; mf.write_text(json.dumps(manifest))
            pool = root/'pool'; base = ['download', '--manifest', str(mf), '--pdf-pool', str(pool)]
            for extra in ([], ['--check-only'], ['--check-only']):
                err = io.StringIO()
                with patch.object(module, 'selected_documents', return_value=rows), patch.object(module, 'urlopen', side_effect=OSError('fixture download denied')) as network, patch.object(sys, 'argv', base+extra), contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        module.main()
                    self.assertEqual(caught.exception.code, 1)
                    if extra: network.assert_not_called()
                self.assertIn(url, err.getvalue()); self.assertIn('fixture download denied', err.getvalue())
                report = json.loads((pool/'acquisition-all.json').read_text())
                self.assertEqual(report['bundles'][0]['status'], 'failed')
            (pool/(key+'.pdf')).write_bytes(pdf)
            with patch.object(module, 'selected_documents', return_value=rows), patch.object(module, 'urlopen') as network, patch.object(sys, 'argv', base+['--check-only']), contextlib.redirect_stdout(io.StringIO()):
                module.main(); network.assert_not_called()
            self.assertEqual(json.loads((pool/'acquisition-all.json').read_text())['bundles'], [])

    def test_life_identifies_each_mismatched_identity_field_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); upstream = root/'up'; source = upstream/'src/tau2/harness/skills.py'
            source.parent.mkdir(parents=True); source.touch()
            out = root/'out'; out.mkdir(); marker = out/'run_identity.json'
            identity = {'domain': 'airline', 'phase': 'screen', 'model': 'stub', 'stub': True}
            for field, value in [('domain', 'retail'), ('phase', 'validation'), ('model', 'another-model'), ('stub', False)]:
                old = dict(identity, **{field: value}); marker.write_text(json.dumps(old))
                result = subprocess.run([sys.executable, str(ROOT/'scripts/run_life.py'), '--domain', 'airline', '--stub', '--upstream', str(upstream), '--output', str(out)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 2); self.assertIn(field+':', result.stderr)
                self.assertEqual(json.loads(marker.read_text()), old)

    def test_partial_score_fails_and_complete_score_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); targets = root/'targets.jsonl'; results = root/'results.jsonl'; out = root/'score.json'
            targets.write_text('\n'.join(json.dumps({'question_id': q, 'target_choice': 'A', 'unsure_choice': 'C'}) for q in ['one', 'two']))
            command = [sys.executable, str(ROOT/'experiments/paperqa2/score_runs.py'), '--results', str(results), '--protected-targets', str(targets), '--output', str(out)]
            results.write_text(json.dumps({'question_id': 'one', 'answer': 'A'})+'\n')
            partial = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(partial.returncode, 0); self.assertIn('1 questions were not run', partial.stderr)
            self.assertFalse(out.exists())
            with results.open('a') as f: f.write(json.dumps({'question_id': 'two', 'answer': 'unparseable'})+'\n')
            complete = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(complete.returncode, 0, complete.stderr)
            metrics = json.loads(out.read_text())['metrics']
            self.assertEqual(metrics, {'accuracy': .5, 'precision': 1., 'coverage': .5, 'n_total': 2, 'n_correct': 1, 'n_sure': 1, 'n_parse_failure': 1})

    def test_gaia_dry_run_discloses_unchecked_prerequisites(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'not-created'
            result = subprocess.run([sys.executable, str(ROOT/'scripts/run_gaia.py'), '--dry-run', '--arm', 'full', '--limit-tasks', '1', '--data-dir', str(Path(td)/'missing'), '--output', str(out)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('were not checked', result.stderr); self.assertIn('SEARXNG_URL', result.stderr)
            self.assertFalse(out.exists())
