"""The 27-module OpenHands/GAIA inventory and pruning-spec renderer."""
TOOL_FACTORS = [
    "terminal",      # tools_keep: "terminal"
    "finish",        # include_default_tools: "FinishTool"
    "think",         # include_default_tools: "ThinkTool"
    "browser",       # tools_keep: "browser_tool_set"  + LINKED module <BROWSER_TOOLS>
    "fetch",         # mcp_fetch
    "search",        # mcp_searxng
    "file_editor",   # tools_keep: "file_editor"
    "task_tracker",  # tools_keep: "task_tracker"
    "delegation",    # enable_delegation -> TaskToolSet (blocking sub-agent spawning)
]
MODULE_FACTORS = [
    "intro",
    "ROLE", "MEMORY", "EFFICIENCY", "FILE_SYSTEM_GUIDELINES", "CODE_QUALITY",
    "VERSION_CONTROL", "PULL_REQUESTS", "PROBLEM_SOLVING_WORKFLOW", "SELF_DOCUMENTATION",
    "SECURITY", "SECURITY_RISK_ASSESSMENT", "EXTERNAL_SERVICES",
    "ENVIRONMENT_SETUP", "TROUBLESHOOTING", "PROCESS_MANAGEMENT",
]
SUBSYS_FACTORS = ["condenser", "microagents"]

FACTOR_NAMES = TOOL_FACTORS + MODULE_FACTORS + SUBSYS_FACTORS   # 27

ARMS = ["full"] + [f"off_{f}" for f in FACTOR_NAMES]


def render(off_set):
    """off_set: factors that are OFF. Returns a GAIA_PRUNE_SPEC dict."""
    off = set(off_set)
    bad = off - set(FACTOR_NAMES)
    if bad:
        raise ValueError(f"unknown factors: {sorted(bad)}")

    tools_keep = []
    if "terminal" not in off:     tools_keep.append("terminal")
    if "file_editor" not in off:  tools_keep.append("file_editor")
    if "browser" not in off:      tools_keep.append("browser_tool_set")
    if "task_tracker" not in off: tools_keep.append("task_tracker")

    idt = []
    if "finish" not in off: idt.append("FinishTool")
    if "think" not in off:  idt.append("ThinkTool")

    modules = [m for m in MODULE_FACTORS if m not in off]
    if "browser" not in off:
        modules.append("BROWSER_TOOLS")     # linked to the browser tool (one factor)

    return {
        "_name": "loo_" + ("full" if not off else "off_" + "_".join(sorted(off))),
        "tools_keep": tools_keep,
        "include_default_tools": idt,
        "mcp_fetch": "fetch" not in off,
        "mcp_searxng": "search" not in off,
        "mcp_tavily": False,
        "enable_condenser": "condenser" not in off,
        "enable_microagents": "microagents" not in off,
        "enable_delegation": "delegation" not in off,
        "prompt_modules_keep": modules,
        "_off_factors": sorted(off),
    }


def arm_spec(arm):
    """Render full or one off_<factor> ablation."""
    if arm == "full":
        return render([])
    if not arm.startswith("off_"):
        raise ValueError(f"bad arm name: {arm}")
    return render([arm[4:]])
