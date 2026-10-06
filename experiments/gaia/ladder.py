"""Frozen 27-module efficiency-oriented GAIA retention ladder."""
from gaia_loo_factors import FACTOR_NAMES, arm_spec, render

EFFICIENCY_ORDER = [
    'search', 'fetch', 'delegation', 'terminal', 'ROLE', 'PROCESS_MANAGEMENT',
    'FILE_SYSTEM_GUIDELINES', 'task_tracker', 'condenser', 'EFFICIENCY',
    'PROBLEM_SOLVING_WORKFLOW', 'SECURITY', 'TROUBLESHOOTING', 'ENVIRONMENT_SETUP',
    'think', 'browser', 'PULL_REQUESTS', 'CODE_QUALITY', 'intro', 'finish',
    'EXTERNAL_SERVICES', 'SECURITY_RISK_ASSESSMENT', 'MEMORY', 'SELF_DOCUMENTATION',
    'VERSION_CONTROL', 'file_editor', 'microagents',
]
assert set(EFFICIENCY_ORDER) == set(FACTOR_NAMES)


def spec_for(arm):
    if arm == 'allpruned':
        return render(FACTOR_NAMES)
    if arm.startswith('A') and arm[1:].isdigit():
        n = int(arm[1:])
        if not 0 <= n <= len(FACTOR_NAMES):
            raise ValueError(f'Invalid rung {arm}')
        return render(set(FACTOR_NAMES) - set(EFFICIENCY_ORDER[:n]))
    return arm_spec(arm)
