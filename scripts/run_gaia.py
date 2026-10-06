#!/usr/bin/env python3
"""Run the final patched OpenHands/GAIA pipeline with the frozen module specs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/gaia'))
from gaia_loo_factors import ARMS
from ladder import spec_for


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.part')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.chmod(0o600)
    temp.replace(path)


def summarize(row):
    result = row.get('test_result') or {}
    blocks = result.get('metrics_by_usage')
    if not isinstance(blocks, dict):
        raise ValueError('Missing per-usage token telemetry; verify the GAIA release patch')
    usages = [u for values in blocks.values() for u in values]
    if not usages or row.get('error'):
        raise ValueError('Incomplete model run; inspect the attempt output and log')
    return {'task_id': row['instance_id'], 'score': result['score'],
            'input_tokens': sum(u['prompt_tokens'] for u in usages),
            'output_tokens': sum(u['completion_tokens'] for u in usages),
            'usage_ids': sorted(blocks)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['screen', 'validation', 'all-pruned'], default='screen')
    p.add_argument('--arm', action='append')
    p.add_argument('--upstream', type=Path, default=ROOT / 'external/gaia')
    p.add_argument('--data-dir', type=Path, help='Prepared GAIA snapshot; default: <upstream>/benchmarks/gaia/data')
    p.add_argument('--output', type=Path)
    p.add_argument('--llm-config', type=Path, help='Private OpenHands LLM JSON; otherwise use OPENAI_* environment variables')
    p.add_argument('--limit-tasks', type=int)
    p.add_argument('--trials', type=int, default=1, help='Repeated runs, not a deterministic seed guarantee')
    p.add_argument('--timeout', type=int, default=2400)
    p.add_argument('--dry-run', action='store_true', help='Print the run plan only; does not check data, search, or model availability')
    a = p.parse_args()
    if a.trials < 1 or a.timeout < 1 or (a.limit_tasks is not None and a.limit_tasks < 1):
        p.error('Trial count, timeout and task limit must be positive')
    split = 'train' if a.phase == 'screen' else 'validation'
    tasks = json.loads((ROOT / 'configs/splits/gaia.json').read_text())[split][:a.limit_tasks]
    defaults = ARMS if a.phase == 'screen' else ['allpruned'] if a.phase == 'all-pruned' else ['allpruned', *[f'A{n}' for n in range(3, 28, 3)]]
    arms = a.arm or defaults
    if len(set(arms)) != len(arms):
        p.error('Duplicate arms')
    specs = {arm: spec_for(arm) for arm in arms}
    for spec in specs.values():
        spec['mcp_searxng_args'] = ['-y', 'mcp-searxng@2.3.0']
    cfg = json.loads(a.llm_config.read_text()) if a.llm_config else {
        'model': 'openai/' + os.environ.get('OPENAI_MODEL', 'Qwen3.5-122B-A10B'),
        'base_url': os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'),
        'api_key': os.environ.get('OPENAI_API_KEY', 'EMPTY'),
        'temperature': 0.7, 'top_p': 0.8, 'native_tool_calling': True,
        'litellm_extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
    }
    out = (a.output or ROOT / 'outputs/gaia' / a.phase).resolve()
    print(json.dumps({'phase': a.phase, 'tasks': len(tasks), 'arms': arms, 'trials': a.trials, 'cells': len(tasks)*len(arms)*a.trials, 'output': str(out)}), flush=True)
    if a.dry_run:
        print("Plan only: prepared data, SEARXNG_URL/search availability, and model endpoint were not checked.", file=sys.stderr)
        return
    up = a.upstream.resolve()
    if not (up / 'benchmarks/gaia/run_infer.py').is_file():
        p.error('Run setup.sh gaia first')
    from prepare_data import prepare_gaia
    data_dir = (a.data_dir or up / 'benchmarks/gaia/data').resolve()
    try:
        prepare_gaia(data_dir, check_only=True)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'phase': a.phase, 'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                    'source_sha256': {str(f.relative_to(up)): hashlib.sha256(f.read_bytes()).hexdigest() for part in ['gaia', 'utils'] for f in (up / 'benchmarks' / part).glob('*.py')}}
        ip = out / 'run.json'
        if ip.exists() and json.loads(ip.read_text()) != identity:
            p.error('Existing output belongs to another configuration; use a fresh directory')
        save(ip, identity)
        config = out / 'llm.private.json'; save(config, cfg)
        results = []; failures = []
        for arm in arms:
            sp = out / 'specs' / f'{arm}.json'
            if sp.exists() and json.loads(sp.read_text()) != specs[arm]:
                p.error(f'Existing spec changed: {sp}')
            save(sp, specs[arm])
            for task in tasks:
                for trial in range(a.trials):
                    cell = out / 'cells' / arm / task / f't{trial}'
                    canonical = cell / 'canonical.json'
                    if canonical.exists():
                        results.append(json.loads(canonical.read_text())); continue
                    attempt = cell / 'attempts' / str(time.time_ns()); attempt.mkdir(parents=True)
                    select = attempt / 'select.txt'; select.write_text(task + '\n')
                    env = dict(os.environ, GAIA_PRUNE_SPEC=str(sp), GAIA_PERSIST_DIR=str(attempt/'persist'),
                               SHARP_GAIA_DATA_DIR=str(data_dir),
                               EVAL_INSTANCE_TIMEOUT=str(a.timeout), LITELLM_LOCAL_MODEL_COST_MAP='True')
                    env.pop('GAIA_RESUME_DIR', None); env.pop('GAIA_RESUME_CONV_ID', None)
                    # Final benchmark termination/scoring is used directly.
                    env.pop('GAIA_ROBUST_ANSWER', None)
                    cmd = [sys.executable, '-m', 'benchmarks.gaia.run_infer', str(config), '--level', '2023_all',
                           '--split', 'validation', '--workspace', 'local', '--n-limit', '1', '--num-workers', '1',
                           '--max-iterations', '30', '--tool-preset', 'default', '--select', str(select),
                           '--output-dir', str(attempt/'out')]
                    with (attempt/'run.log').open('w') as log:
                        proc = subprocess.Popen(cmd, cwd=up, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            proc.wait(timeout=a.timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                            failures.append({'arm': arm, 'task': task, 'trial': trial, 'reason': 'wall timeout'})
                            continue
                        except BaseException:
                            if proc.poll() is None:
                                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                            raise
                    try:
                        records = [json.loads(line) for file in (attempt/'out').rglob('output.jsonl') for line in file.read_text().splitlines() if line.strip()]
                        if proc.returncode or len(records) != 1 or str(records[0]['instance_id']) != task:
                            raise ValueError('Expected exactly one completed task output')
                        result = dict(summarize(records[0]), arm=arm, trial=trial, attempt=str(attempt.relative_to(out)))
                        save(canonical, result); results.append(result)
                        print(json.dumps(result), flush=True)
                    except (ValueError, KeyError) as exc:
                        failures.append({'arm': arm, 'task': task, 'trial': trial, 'reason': str(exc)})
        save(out/'summary.json', {'results': results, 'failures': failures})
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
