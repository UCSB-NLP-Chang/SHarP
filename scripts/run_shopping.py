#!/usr/bin/env python3
"""Run the frozen JIT Agentfold/Shopping screen or validation ladder on Linux."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['screen', 'validation', 'performance', 'all-pruned'], default='screen')
    p.add_argument('--upstream', type=Path, default=ROOT / 'external/shopping')
    p.add_argument('--output', type=Path)
    p.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))
    p.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'Qwen3.5-122B-A10B'))
    p.add_argument('--arm', action='append', help='Base arm (e.g. full, off_c_compact, S5); repeat to select several')
    p.add_argument('--limit-tasks', type=int)
    p.add_argument('--stub', action='store_true', help='Scripted responses; requires Linux/bwrap, never paper results')
    p.add_argument('--dry-run', action='store_true', help='Validate dataset and print plan without model calls')
    a = p.parse_args()
    if a.limit_tasks is not None and a.limit_tasks < 1:
        p.error('--limit-tasks must be positive')
    up = a.upstream.resolve()
    if not (up / 'dataset/deepplanning_shopping/tools/base_shopping_tool.py').is_file():
        p.error('Pinned JIT checkout missing; run setup.sh shopping or pass --upstream')
    code = ROOT / 'experiments/shopping' / ('validation' if a.phase == 'all-pruned' else a.phase)
    parent = ROOT / 'experiments/shopping/common'
    out = (a.output or ROOT / 'outputs/shopping' / a.phase / ('stub' if a.stub else 'model')).resolve()
    os.environ.update(JIT_ROOT=str(out), JIT_UPSTREAM=str(up), JIT_PARENT_CODE=str(parent),
                      JIT_FAKE_MODEL='1' if a.stub else '0', OPENAI_MODEL=a.model)
    sys.path[:0] = [str(code), str(parent), str(up)]
    import cohort_fine as CO
    import units as U
    import common as C
    mode = 'screen' if a.phase == 'screen' else 'premise' if a.phase == 'all-pruned' else 'ladder'
    # Include full and all-pruned reference configurations with the intermediate rungs.
    if mode == 'ladder':
        CO.MODES[mode]['arms'] = list(dict.fromkeys(['S0', *U.LADDER_ARMS, 'full']))
    if a.phase == 'all-pruned':
        CO.MODES[mode]['arms'] = ['S0']
    cohort = CO.build(mode)
    if a.arm and set(a.arm) - set(cohort['arms']):
        p.error(f"Unknown arms; choose from {cohort['arms']}")
    tasks = cohort['tasks'][:a.limit_tasks]
    selected = [(t, arm) for t in tasks for arm in t['arms'] if not a.arm or U.split_label(arm)[0] in a.arm]
    print(json.dumps({'phase': a.phase, 'tasks': len(tasks), 'cells': len(selected), 'arms': cohort['arms'], 'stub': a.stub, 'output': str(out)}), flush=True)
    if a.dry_run:
        return
    if sys.platform != 'linux' or not shutil.which('bwrap'):
        p.error('Execution requires Linux and bubblewrap (bwrap); --dry-run works on macOS')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'phase': a.phase, 'stub': a.stub, 'model': a.model, 'base_url': a.base_url}
        for name, value in [('run.json', identity), ('cohort.json', cohort)]:
            target = out / name
            if target.exists() and C.load_json(target) != value:
                p.error(f'{target} differs; use a fresh output directory')
            if not target.exists():
                C.save(target, value)
        protocol = 'SHarP Agentfold/Shopping: frozen splits; temperature 0; thinking off; official tools; isolated execute_code; three validation trials.\n'
        proto = out / 'PROTOCOL.md'
        if proto.exists() and proto.read_text() != protocol:
            p.error('Protocol changed; use a fresh output directory')
        proto.write_text(protocol)
        import cell_fine
        cell_fine.freeze_contract()
        failures = []
        rows = []
        for task, arm in selected:
            result = subprocess.run([sys.executable, str(code / 'cell_fine.py'), '--index', str(task['index']), '--arm', arm, '--base', a.base_url])
            if result.returncode:
                failures.append({'task': task['question_id'], 'arm': arm, 'exit_code': result.returncode})
            else:
                row = C.load_json(out / 'cells' / task['question_id'] / arm / 'canonical.json')
                if a.stub and row['termination'] != 'final_answer':
                    failures.append({'task': task['question_id'], 'arm': arm,
                                     'reason': 'Scripted smoke did not reach final_answer',
                                     'termination': row['termination']})
                    continue
                rows.append(row)
        C.save(out / 'summary.json', {'stub': a.stub, 'valid_cells': len(rows), 'failed_cells': failures})
        if a.phase == 'screen' and not a.stub:
            C.save(out / 'components.json', U.UIDS)
            with (out / 'single_off.csv').open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['task_id', 'component', 'score', 'tokens'])
                for r in rows:
                    if r['base_arm'].startswith('off_'):
                        w.writerow([r['question_id'], r['base_arm'][4:], r['score']['case_score'], r['total_tokens']])
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
