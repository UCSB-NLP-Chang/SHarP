"""Scripted stand-ins for every model role. CPU dry run only (JIT_FAKE_MODEL=1).

Enabled only by JIT_FAKE_MODEL=1 for integration checks. The
scripts exercise the real upstream code paths (meta generation parsing, judge
selection, kernel execution with the official tools, scoring) with canned
outputs against the selected task's isolated product and cart state.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import types
from pathlib import Path

import common

CROSS = "Cross-store: ¥30 off every ¥300"
BRAND = "Same-brand: ¥25 off every ¥200"
# Positional script for the native-tool-call (reference_native) dry run, where
# every model call IS an action step. The kernel arms use next_exec_action().
FIXTURE_CALLS = [
    ("get_user_info", {}),
    ("execute_code", {"code": "print(2 * 400)", "code_type": "python"}),
    ("add_product_to_cart", {"product_id": "synthetic_0", "quantity": 2}),
    ("add_coupon_to_cart", {"coupon_name": CROSS, "quantity": 1}),
    ("get_cart_info", {}),
    ("final_answer", {"answer": "Synthetic cart complete."}),
]
_LOCK = threading.Lock()
_STATE = {"exec_calls": 0}


def reset() -> None:
    _STATE["exec_calls"] = 0


def _stack_names() -> set:
    names = set()
    frame = sys._getframe(1)
    while frame is not None:
        names.add(frame.f_code.co_name)
        frame = frame.f_back
    return names


def harness_text(name: str) -> str:
    """A seed harness wrapped in the five tagged blocks the meta model must emit."""
    from jit.harness_ops import SECTION_TAG_TO_FILE
    root = common.UPSTREAM / "harness_factory" / "harnesses" / name
    parts = ["Architecture analysis: fixture harness copied verbatim from the seed bank for the dry run."]
    for tag, filename in SECTION_TAG_TO_FILE.items():
        parts.append(f"<<<{tag}>>>\n{(root / filename).read_text(encoding='utf-8').rstrip()}\n<<<END_{tag}>>>")
    return "\n\n".join(parts)


def _executed_tools() -> set:
    """Which tools this cell has really executed (patched Tool.__call__ log)."""
    path = os.environ.get("JIT_TOOL_LOG", "")
    if not path or not Path(path).is_file():
        return set()
    return {r["name"] for r in common.read_jsonl(path) if r.get("name")}


def _current_cart() -> dict:
    """The newest official-tool cart under JIT_TOOL_STATE_ROOT, or an empty one."""
    root = os.environ.get("JIT_TOOL_STATE_ROOT", "")
    carts = sorted(Path(root).glob("*/cart.json"), key=lambda p: p.stat().st_mtime) if root else []
    for path in reversed(carts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return dict(EMPTY_CART)


def _available_product() -> dict:
    """Select a stocked product from tool-visible state, never scoring targets."""
    root = Path(os.environ["JIT_TOOL_STATE_ROOT"])
    carts = sorted(root.glob("*/cart.json"), key=lambda p: p.stat().st_mtime)
    if not carts:
        raise RuntimeError("Shopping smoke test has no initialized tool state")
    products = common.read_jsonl(carts[-1].parent / "products.jsonl")
    for product in products:
        if product.get("stock_quantity", 0) > 0:
            return product
    raise RuntimeError("Shopping smoke test requires a stocked product")


def next_exec_action():
    """The next scripted execution action, decided from OBSERVED tool state.

    Deliberately not a positional script: a harness makes model calls the
    fixture cannot classify (planning, folding, review), and a positional
    script would let one of those swallow the add-to-cart step and silently
    turn the dry run into a 0-score. Reading what has actually been executed
    makes the dry run converge for every harness.
    """
    executed = _executed_tools()
    cart = _current_cart()
    if "get_user_info" not in executed:
        return "get_user_info", {}
    if "execute_code" not in executed:
        return "execute_code", {"code": "print(2 * 400)", "code_type": "python"}
    if not cart.get("items"):
        if "add_product_to_cart" in executed:
            raise RuntimeError("Shopping smoke test failed to add a product")
        product = _available_product()
        return "add_product_to_cart", {"product_id": product["product_id"],
                                       "quantity": min(2, product["stock_quantity"])}
    # Preserve coupon coverage for the synthetic fixture. Real tasks do not
    # necessarily offer its coupon or meet its eligibility conditions.
    if any(item.get("product_id") == "synthetic_0" for item in cart["items"]) and not cart.get("used_coupons"):
        return "add_coupon_to_cart", {"coupon_name": CROSS, "quantity": 1}
    if "get_cart_info" not in executed:
        return "get_cart_info", {}
    return "final_answer", {"answer": "Scripted tool integration complete."}


def next_exec_content() -> str:
    with _LOCK:
        _STATE["exec_calls"] += 1
        name, args = next_exec_action()
    return json.dumps({"think": "fixture", "tools": [{"name": name, "arguments": args}]})


def fake_call(self, messages, stop_sequences=None, grammar=None, tools_to_call_from=None, **kwargs):
    """Drop-in for OpenAIServerModel.__call__ (installed by jit_patches in fake mode)."""
    from scripts.models.base import ChatMessage
    role = getattr(self, "_jit_role", "unknown")
    names = _stack_names()
    if role == "meta":
        content = harness_text(os.environ.get("JIT_FAKE_HARNESS", "plan_and_execute"))
    elif role == "judge":
        content = json.dumps({"best": 1, "reason": "fixture judge"})
    elif "init_plan" in names or "update_plan" in names:
        content = "Fixture plan: use the scripted actions."
    elif "_do_summarize" in names:
        content = "<summary>Fixture summary.</summary>"
    elif "_force_final_answer" in names:
        content = json.dumps({"think": "", "answer": "forced fixture answer"})
    else:
        content = next_exec_content()
    self._record_token_usage(input_token_count=len(json.dumps(messages, default=str)) // 4,
                             output_token_count=max(1, len(content) // 4))
    message = ChatMessage(role="assistant", content=content)
    message.raw = types.SimpleNamespace(
        choices=[types.SimpleNamespace(finish_reason="stop")],
        usage=types.SimpleNamespace(model_dump=lambda: {"prompt_tokens": self.last_input_token_count,
                                                        "completion_tokens": self.last_output_token_count}),
    )
    return message


class FakeOpenAIClient:
    """Minimal stand-in for openai.OpenAI used by reference_native in the dry run.

    Returns one canned native tool call per request following FIXTURE_CALLS
    (or a custom script), with usage numbers and no reasoning field.
    """

    def __init__(self, script=None, plain_text_first: bool = False):
        self.script = list(script or FIXTURE_CALLS)
        self.requests = []
        self.plain_text_first = plain_text_first
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **payload):
        self.requests.append(payload)
        assert payload.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False, payload
        assert payload.get("tools"), "tools must be passed in the request"
        n_calls = len(self.requests)
        prompt_tokens = len(json.dumps(payload["messages"], default=str)) // 4
        if self.plain_text_first and n_calls == 1:
            message = types.SimpleNamespace(content="Let me think about it.", tool_calls=None, reasoning_content=None)
            finish = "stop"
        else:
            idx = n_calls - 2 if self.plain_text_first else n_calls - 1
            name, args = self.script[min(idx, len(self.script) - 1)]
            tc = types.SimpleNamespace(id=f"call_{n_calls}", type="function",
                                       function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))
            message = types.SimpleNamespace(content="", tool_calls=[tc], reasoning_content=None)
            finish = "tool_calls"
        usage = types.SimpleNamespace(model_dump=lambda: {"prompt_tokens": prompt_tokens, "completion_tokens": 12})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)],
                                     usage=usage, model_dump=lambda: {"fake": True})


PRODUCTS = [{
    "product_id": f"synthetic_{i}",
    "name": "singular cobalt jacket" if i == 0 else f"plain garment {i}",
    "brand": "FixtureA" if i < 2 else "FixtureB", "color": "Blue" if i < 2 else "White", "size": "M",
    "price": 400 if i == 0 else 100 + i, "stock_quantity": 10,
    "sales_volume": {"monthly": i + 1, "total": i + 10}, "rating": {"average_score": 4.0, "total_reviews": 10},
    "shipping_info": {"origin": "Beijing", "provider": "China Post"},
    "applicable_coupons": [CROSS, BRAND] if i < 2 else [],
} for i in range(20)]
USER = {"user_id": "synthetic_user", "address": "Guangdong", "is_vip": True, "coupons": {CROSS: 2, BRAND: 2}}
EMPTY_CART = {"items": [], "used_coupons": [], "summary": {"total_items_count": 0, "total_price": 0}}
VALIDATION = {"ground_truth_products": [{"product_id": "synthetic_0", "quantity": 2}], "ground_truth_coupons": {CROSS: 1}}


# Use one representative fixture per task level.
def write_fixture_dataset(root: Path, question_ids=("level1_case1", "level2_case1", "level3_case2")) -> Path:
    """A tiny dataset with the upstream layout (data/ + database_level<L>/case_<C>/).

    One case per level, so common.build_cohort(..., per_level=1) exercises the
    real cohort code path and every arm has a task to run in the dry run.
    """
    for question_id in question_ids:
        level, case = question_id.replace("level", "").split("_case")
        data = root / "data"
        db = root / f"database_level{level}" / f"case_{case}"
        data.mkdir(parents=True, exist_ok=True)
        db.mkdir(parents=True, exist_ok=True)
        (data / f"level_{level}_query_meta.json").write_text(json.dumps(
            [{"id": case, "query": "Invented test fixture only: buy two singular cobalt jackets at the lowest price."}]) + "\n")
        (db / "products.jsonl").write_text("".join(json.dumps(p) + "\n" for p in PRODUCTS))
        (db / "user_info.json").write_text(json.dumps(USER, ensure_ascii=False, indent=2) + "\n")
        (db / "cart.json").write_text(json.dumps(EMPTY_CART) + "\n")
        (db / "validation_cases.json").write_text(json.dumps(VALIDATION, ensure_ascii=False) + "\n")
    return root
