#!/usr/bin/env python3
"""Score a complete PaperQA single-off grid and export SHarP screening input."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/paperqa2'))
from components import component_names
from protocol import score_answer
from runner_common import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--protected-targets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New export directory')
    a = p.parse_args()
    targets_list = read_jsonl(a.protected_targets)
    targets = {str(r['question_id']):r for r in targets_list}
    if len(targets) != len(targets_list) or not targets:
        p.error('Targets must be nonempty with unique question IDs')
    rows = []
    for name in component_names():
        cells = read_jsonl(a.results/f'{name}.jsonl')
        if len(cells) != len(targets) or {str(c['question_id']) for c in cells} != set(targets):
            p.error(f'{name} is incomplete or has duplicate/non-frozen question IDs')
        for cell in cells:
            if cell['run_status'] != 'ok':
                p.error(f'{name} contains a noncanonical/error row')
            qid = str(cell['question_id']); telemetry = cell['telemetry']
            tokens = telemetry['input_tokens'] + telemetry['output_tokens']
            rows.append([qid,name,int(score_answer(cell['answer'],targets[qid])['correct']),tokens])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'components.json').write_text(json.dumps(component_names(),indent=2)+'\n')
    with (a.output/'single_off.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task_id','component','score','tokens']);w.writerows(rows)
    print(f'Exported {len(rows)} paired cells; preprocessing tokens are excluded.')


if __name__ == '__main__':
    main()
