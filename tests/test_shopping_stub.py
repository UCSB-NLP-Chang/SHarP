"""Regression for scripted actions against non-synthetic Shopping task data."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class ShoppingStubTests(unittest.TestCase):
    def test_real_product_cart_and_normal_termination(self):
        code = Path(__file__).resolve().parents[1] / 'experiments/shopping/common'
        with patch.dict(sys.modules), patch.object(sys, 'path', [str(code), *sys.path]):
            sys.modules.pop('common', None)
            spec = importlib.util.spec_from_file_location('shopping_smoke_fixture', code / 'fake_model.py')
            stub = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stub)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = root / 'case'; state.mkdir()
            log = root / 'tools.jsonl'
            (state / 'cart.json').write_text(json.dumps({'items': [], 'used_coupons': []}))
            (state / 'products.jsonl').write_text('\n'.join(json.dumps(p) for p in [
                {'product_id': 'out-of-stock', 'stock_quantity': 0},
                {'product_id': 'actual-product', 'stock_quantity': 1},
            ]))
            log.write_text('\n'.join(json.dumps({'name': name}) for name in ['get_user_info', 'execute_code']))
            with patch.dict(os.environ, JIT_TOOL_STATE_ROOT=td, JIT_TOOL_LOG=str(log)):
                self.assertEqual(stub.next_exec_action(), ('add_product_to_cart', {'product_id': 'actual-product', 'quantity': 1}))
                with log.open('a') as f:
                    f.write('\n' + json.dumps({'name': 'add_product_to_cart'}))
                with self.assertRaisesRegex(RuntimeError, 'failed to add'):
                    stub.next_exec_action()
                (state / 'cart.json').write_text(json.dumps({'items': [{'product_id': 'actual-product'}], 'used_coupons': []}))
                self.assertEqual(stub.next_exec_action(), ('get_cart_info', {}))
                with log.open('a') as f:
                    f.write('\n' + json.dumps({'name': 'get_cart_info'}))
                self.assertEqual(stub.next_exec_action()[0], 'final_answer')
