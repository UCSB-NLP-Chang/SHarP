#!/usr/bin/env python3
"""Validate all 23 agentfold module cuts against the pinned JIT source, offline."""
import argparse
import ast
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/shopping/validation"))
import units
import variant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, default=ROOT / "external/shopping")
    a = p.parse_args()
    source = a.upstream.resolve() / "harness_factory/harnesses/agentfold"
    if not (source / "action.py").is_file():
        p.error("Run setup.sh shopping or supply the pinned JIT checkout with --upstream")
    with tempfile.TemporaryDirectory(prefix="sharp-shopping-") as td:
        dest = Path(td)
        cases = [("full", frozenset())] + [(u, frozenset([u])) for u in units.UIDS] + [("all_pruned", frozenset(units.UIDS))]
        for name, off in cases:
            variant.build_harness(off, source, dest / name)
            changed = []
            for fn in variant.HARNESS_FILES:
                original, generated = source / fn, dest / name / "agentfold" / fn
                if original.is_file() and original.read_bytes() != generated.read_bytes():
                    changed.append(fn)
                if generated.suffix == ".py" and generated.exists():
                    ast.parse(generated.read_text(), filename=fn)
            if name == "full":
                assert not changed, changed
            elif name != "all_pruned":
                assert len(changed) == {"M": 1, "S": 1, "P": 1, "T": 0, "G": 0}[units.LAYER[name]], (name, changed)
        # Each task guidance cut must preserve the objective and user request.
        import ast as py_ast
        adapter = py_ast.parse((a.upstream / "benchmark/adapter/deepplanning.py").read_text())
        prompts = {n.targets[0].id: py_ast.literal_eval(n.value) for n in adapter.body
                   if isinstance(n, py_ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], py_ast.Name)
                   and n.targets[0].id in {f"SHOPPING_SYSTEM_PROMPT_L{i}" for i in (1, 2, 3)}}
        assert len(prompts) == 3
        for prompt in prompts.values():
            task = prompt + "\n\n## User Request\n\nSYNTHETIC QUERY\n\n## Workspace\nTemporary workspace."
            assert variant.cut_task(task, prompt, frozenset()) == task
            cut = variant.cut_task(task, prompt, frozenset(units.UIDS))
            assert "Core Mission" in cut and "SYNTHETIC QUERY" in cut and "Guiding Principles" not in cut
            assert "## Workspace" not in cut
    print("Passed: full identity, all 23 module cuts, all-pruned build, and three task-prompt levels.")


if __name__ == "__main__":
    main()
