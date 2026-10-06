"""Adapt the pinned DeepPlanning shopping tools to the JIT tool interface."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys

from scripts.tools.base import Tool

import common
SOURCE = common.DATASET / 'tools'


class OfficialToolAdapter(Tool):
    output_type='string'
    def __init__(self,backend,schema):
        self.backend=backend
        self.name=schema['name'];self.description=schema['description']
        params=schema['parameters'];self.inputs=deepcopy(params['properties'])
        for name,p in self.inputs.items():
            p.setdefault('description','')
            if name not in params.get('required',[]):p['nullable']=True
        super().__init__()
        self.validate_arguments()
    def forward(self,**kwargs):
        return self.backend.call(kwargs)


def create_official_tools(database_dir,tool_state_dir):
    """Fresh isolated cart plus product/profile copies; never copy answer files."""
    database_dir=Path(database_dir).resolve();tool_state_dir=Path(tool_state_dir).resolve()
    # Refuse to reset an existing run. No in-place canonical replacement.
    tool_state_dir.mkdir(parents=True,exist_ok=False)
    for name in ['products.jsonl','user_info.json']:
        shutil.copyfile(database_dir/name,tool_state_dir/name)
    (tool_state_dir/'cart.json').write_text(json.dumps({
        'items':[],'used_coupons':[],'summary':{'total_items_count':0,'total_price':0}})+'\n')
    source=str(SOURCE)
    if source not in sys.path:sys.path.insert(0,source)
    base=importlib.import_module('base_shopping_tool')
    if Path(base.__file__).resolve()!=SOURCE/'base_shopping_tool.py':
        raise RuntimeError('Refuse a conflicting official-tool module import')
    for p in sorted(SOURCE.glob('*.py')):
        if p.stem not in ['__init__','base_shopping_tool']:importlib.import_module(p.stem)
    schemas={s['function']['name']:s['function'] for s in json.loads((SOURCE/'shopping_tool_schema.json').read_text())}
    assert set(schemas)==set(base.TOOL_REGISTRY)
    return {name:OfficialToolAdapter(cls(cfg={'database_path':str(tool_state_dir)}),schemas[name])
            for name,cls in base.TOOL_REGISTRY.items()}
