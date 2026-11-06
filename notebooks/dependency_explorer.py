import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="full",
    app_title="Module Dependency & Dead Code Explorer",
)


@app.cell(hide_code=True)
def _():
    import ast
    import contextlib
    from collections import defaultdict, deque
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    return Path, ast, contextlib, defaultdict, deque, go, mo, np, pd


@app.cell(hide_code=True)
def _(Path, ast, contextlib, defaultdict):
    from dataclasses import dataclass, field

    @dataclass
    class EdgeInfo:
        source: str
        target: str
        kind: str  # "top_level" | "deferred" | "type_checking"
        symbols: set[str] = field(default_factory=set)
        locations: list[str] = field(default_factory=list)

    @dataclass
    class ModuleInfo:
        name: str
        root_group: str  # "src" | "scripts" | "notebooks"
        subsystem: str
        rel_path: str
        loc: int

    @dataclass
    class SymbolInfo:
        module: str
        name: str
        kind: str  # "function" | "class" | "method" | "field"
        parent_class: str
        lineno: int
        is_private: bool
        is_dunder: bool
        category: str  # "unused" | "test_only" | "internal_only" | "active"
        same_file_refs: int
        src_refs: list[str] = field(default_factory=list)
        scripts_refs: list[str] = field(default_factory=list)
        tests_refs: list[str] = field(default_factory=list)
        imported_by: list[str] = field(default_factory=list)

    def is_marimo_cell(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr == "cell":
                return True
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "cell":
                return True
        return False

    def parse_repo_dependencies(
        repo_root: Path,
    ) -> tuple[
        dict[str, ModuleInfo],
        list[EdgeInfo],
        list[SymbolInfo],
        dict[str, list[SymbolInfo]],
        dict[str, list[tuple[str, str]]],
        dict[str, list[tuple[str, str, str]]],
    ]:
        modules: dict[str, ModuleInfo] = {}
        file_map: dict[str, tuple[Path, str, Path]] = {}

        src_root = repo_root / "src"
        if src_root.is_dir():
            for p in sorted(src_root.rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                rel = p.relative_to(src_root)
                text = p.read_text(encoding="utf-8")
                if p.name == "__init__.py":
                    stripped = [
                        line
                        for line in text.splitlines()
                        if line.strip() and not line.strip().startswith("#") and "from __future__" not in line
                    ]
                    if not stripped:
                        continue
                    mod_name = ".".join(rel.parent.parts)
                else:
                    mod_name = ".".join(rel.with_suffix("").parts)

                subsystem = ".".join(rel.parent.parts) if rel.parent.parts else "src"
                modules[mod_name] = ModuleInfo(
                    name=mod_name,
                    root_group="src",
                    subsystem=subsystem,
                    rel_path=p.relative_to(repo_root).as_posix(),
                    loc=len(text.splitlines()),
                )
                file_map[mod_name] = (p, "src", src_root)

        for extra_group in ("scripts", "notebooks"):
            group_root = repo_root / extra_group
            if not group_root.is_dir():
                continue
            for p in sorted(group_root.rglob("*.py")):
                if any(part.startswith("__") for part in p.parts):
                    continue
                rel = p.relative_to(group_root)
                if p.name == "__init__.py":
                    continue
                text = p.read_text(encoding="utf-8")
                mod_name = f"{extra_group}.{'.'.join(rel.with_suffix('').parts)}"
                sub_parts = (extra_group, *rel.parent.parts)
                subsystem = ".".join(sub_parts)
                modules[mod_name] = ModuleInfo(
                    name=mod_name,
                    root_group=extra_group,
                    subsystem=subsystem,
                    rel_path=p.relative_to(repo_root).as_posix(),
                    loc=len(text.splitlines()),
                )
                file_map[mod_name] = (p, extra_group, group_root)

        known_modules = set(modules)
        raw_edges: dict[tuple[str, str], dict[str, EdgeInfo]] = defaultdict(dict)

        def resolve_internal_target(mod_str: str) -> str | None:
            if not mod_str:
                return None
            if mod_str in known_modules:
                return mod_str
            parts = mod_str.split(".")
            while parts:
                candidate = ".".join(parts)
                if candidate in known_modules:
                    return candidate
                parts.pop()
            return None

        # Parse ASTs and extract imports for all files in file_map
        parsed_trees: dict[str, ast.AST] = {}
        for mod_name, (p, root_group, base_root) in file_map.items():
            rel = p.relative_to(base_root)
            tree = ast.parse(p.read_text(encoding="utf-8"))
            parsed_trees[mod_name] = tree

            tc_nodes: set[ast.AST] = set()
            func_scope: dict[ast.AST, str] = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    test = node.test
                    is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                    )
                    if is_tc:
                        for sub in ast.walk(node):
                            tc_nodes.add(sub)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if root_group == "notebooks" and is_marimo_cell(node):
                        continue
                    for sub in ast.walk(node):
                        if sub not in func_scope:
                            func_scope[sub] = node.name

            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue

                if node in tc_nodes:
                    kind = "type_checking"
                    loc_label = f"L{node.lineno} (TYPE_CHECKING)"
                elif node in func_scope:
                    kind = "deferred"
                    loc_label = f"L{node.lineno} (in {func_scope[node]}())"
                else:
                    kind = "top_level"
                    loc_label = f"L{node.lineno}"

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = resolve_internal_target(alias.name)
                        if target and target != mod_name:
                            info = raw_edges[(mod_name, target)].setdefault(
                                kind, EdgeInfo(source=mod_name, target=target, kind=kind)
                            )
                            info.symbols.add(alias.asname or alias.name)
                            info.locations.append(loc_label)

                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        pkg_parts = list(rel.parent.parts) if root_group == "src" else [root_group, *rel.parent.parts]
                        for _ in range(node.level - 1):
                            if pkg_parts:
                                pkg_parts.pop()
                        base_mod = ".".join(pkg_parts)
                        full_mod = f"{base_mod}.{node.module}" if node.module else base_mod
                    else:
                        full_mod = node.module or ""

                    if not full_mod:
                        continue

                    for alias in node.names:
                        submod_candidate = f"{full_mod}.{alias.name}"
                        target = (
                            submod_candidate if submod_candidate in known_modules else resolve_internal_target(full_mod)
                        )
                        if target and target != mod_name:
                            info = raw_edges[(mod_name, target)].setdefault(
                                kind, EdgeInfo(source=mod_name, target=target, kind=kind)
                            )
                            info.symbols.add(alias.name)
                            info.locations.append(loc_label)

        priority = ("top_level", "deferred", "type_checking")
        edges: list[EdgeInfo] = []
        for (_src, _dst), by_kind in sorted(raw_edges.items()):
            for k in priority:
                if k in by_kind:
                    chosen = by_kind[k]
                    for other_k, other_info in by_kind.items():
                        if other_k != k:
                            chosen.symbols.update(other_info.symbols)
                            chosen.locations.extend(other_info.locations)
                    edges.append(chosen)
                    break

        # --- Symbol and Dead Code Analysis Across All Scopes ---
        # Also parse tests/ for reference tracking
        test_trees: dict[str, ast.AST] = {}
        tests_root = repo_root / "tests"
        if tests_root.is_dir():
            for p in sorted(tests_root.rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                with contextlib.suppress(Exception):
                    test_trees[p.name] = ast.parse(p.read_text(encoding="utf-8"))

        # Collect file reference tokens
        # For each file: group, set of imported symbols per target_mod, set of all Name IDs, set of all Attribute attrs
        file_tokens: dict[str, tuple[str, dict[str, set[str]], set[str], set[str]]] = {}
        for mod_name, (_p, root_group, _base_root) in file_map.items():
            tree = parsed_trees[mod_name]
            imp_syms: dict[str, set[str]] = defaultdict(set)
            name_ids: set[str] = set()
            attr_names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    t = resolve_internal_target(node.module or "")
                    if t:
                        for a in node.names:
                            imp_syms[t].add(a.name)
                elif isinstance(node, ast.Name):
                    name_ids.add(node.id)
                elif isinstance(node, ast.Attribute):
                    attr_names.add(node.attr)
            file_tokens[mod_name] = (root_group, imp_syms, name_ids, attr_names)

        for tname, ttree in test_trees.items():
            imp_syms = defaultdict(set)
            name_ids = set()
            attr_names = set()
            for node in ast.walk(ttree):
                if isinstance(node, ast.ImportFrom):
                    t = resolve_internal_target(node.module or "")
                    if t:
                        for a in node.names:
                            imp_syms[t].add(a.name)
                elif isinstance(node, ast.Name):
                    name_ids.add(node.id)
                elif isinstance(node, ast.Attribute):
                    attr_names.add(node.attr)
            file_tokens[f"tests.{tname}"] = ("tests", imp_syms, name_ids, attr_names)

        # Collect definitions in src/ modules
        symbols_list: list[SymbolInfo] = []
        module_symbols_map: dict[str, list[SymbolInfo]] = defaultdict(list)
        internal_calls_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
        external_calls_map: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

        for mod_name, (_p, root_group, _base_root) in file_map.items():
            if root_group != "src":
                continue
            tree = parsed_trees[mod_name]

            # Track imports for call resolution
            local_imported_syms: dict[str, tuple[str, str]] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    t = resolve_internal_target(node.module or "")
                    if t:
                        for a in node.names:
                            local_imported_syms[a.asname or a.name] = (t, a.name)

            local_defined_names: set[str] = set()
            raw_defs: list[tuple[str, str, str, int, bool, bool]] = []

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_priv = node.name.startswith("_") and not (
                        node.name.startswith("__") and node.name.endswith("__")
                    )
                    is_dund = node.name.startswith("__") and node.name.endswith("__")
                    raw_defs.append((node.name, "function", "", node.lineno, is_priv, is_dund))
                    local_defined_names.add(node.name)

                    # Extract local and external calls from this function
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            call_name = (
                                sub.func.id
                                if isinstance(sub.func, ast.Name)
                                else (sub.func.attr if isinstance(sub.func, ast.Attribute) else None)
                            )
                            if call_name and call_name != node.name and call_name in local_imported_syms:
                                ext_mod, ext_sym = local_imported_syms[call_name]
                                external_calls_map[mod_name].append((node.name, ext_mod, ext_sym))

                elif isinstance(node, ast.ClassDef):
                    is_priv = node.name.startswith("_") and not (
                        node.name.startswith("__") and node.name.endswith("__")
                    )
                    is_dund = node.name.startswith("__") and node.name.endswith("__")
                    raw_defs.append((node.name, "class", "", node.lineno, is_priv, is_dund))
                    local_defined_names.add(node.name)

                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sub_priv = sub.name.startswith("_") and not (
                                sub.name.startswith("__") and sub.name.endswith("__")
                            )
                            sub_dund = sub.name.startswith("__") and sub.name.endswith("__")
                            raw_defs.append((sub.name, "method", node.name, sub.lineno, sub_priv, sub_dund))
                            local_defined_names.add(sub.name)

                            for cnode in ast.walk(sub):
                                if isinstance(cnode, ast.Call):
                                    call_name = (
                                        cnode.func.id
                                        if isinstance(cnode.func, ast.Name)
                                        else (cnode.func.attr if isinstance(cnode.func, ast.Attribute) else None)
                                    )
                                    if call_name and call_name in local_imported_syms:
                                        ext_mod, ext_sym = local_imported_syms[call_name]
                                        external_calls_map[mod_name].append(
                                            (f"{node.name}.{sub.name}", ext_mod, ext_sym)
                                        )

                        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                            sub_priv = sub.target.id.startswith("_")
                            raw_defs.append((sub.target.id, "field", node.name, sub.lineno, sub_priv, False))
                            local_defined_names.add(sub.target.id)

            # Extract intra-module calls
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            call_name = (
                                sub.func.id
                                if isinstance(sub.func, ast.Name)
                                else (sub.func.attr if isinstance(sub.func, ast.Attribute) else None)
                            )
                            if call_name and call_name in local_defined_names and call_name != node.name:
                                internal_calls_map[mod_name].append((node.name, call_name))

            # Reference counting across scopes for each defined symbol
            for sym, kind, parent, lineno, is_priv, is_dund in raw_defs:
                # Count same file references (excluding definition line)
                same_file_refs = 0
                for node in ast.walk(tree):
                    if hasattr(node, "lineno") and node.lineno == lineno:
                        continue
                    if (isinstance(node, ast.Name) and node.id == sym) or (
                        isinstance(node, ast.Attribute) and node.attr == sym
                    ):
                        same_file_refs += 1

                src_refs: list[str] = []
                scripts_refs: list[str] = []
                tests_refs: list[str] = []
                imported_by: list[str] = []

                for other_name, (grp, imp_map, name_ids, attr_names) in file_tokens.items():
                    if other_name == mod_name:
                        continue
                    has_ref = False
                    is_direct_import = sym in imp_map.get(mod_name, set())

                    if is_direct_import:
                        imported_by.append(other_name)
                        has_ref = True
                    elif (
                        kind in ("function", "class")
                        and (sym in name_ids or sym in attr_names)
                        and (mod_name in imp_map or any(m.startswith(mod_name) for m in imp_map))
                    ) or sym in attr_names:
                        has_ref = True

                    if has_ref:
                        if grp == "src":
                            src_refs.append(other_name)
                        elif grp in ("scripts", "notebooks"):
                            scripts_refs.append(other_name)
                        elif grp == "tests":
                            tests_refs.append(other_name)

                # Determine category
                if is_dund:
                    cat = "active"  # Standard Python protocol dunders are invoked by runtime
                elif src_refs or scripts_refs:
                    cat = "active"
                elif tests_refs:
                    cat = "test_only"
                elif same_file_refs > 0:
                    cat = "internal_only"
                else:
                    cat = "unused"

                sym_info = SymbolInfo(
                    module=mod_name,
                    name=sym,
                    kind=kind,
                    parent_class=parent,
                    lineno=lineno,
                    is_private=is_priv,
                    is_dunder=is_dund,
                    category=cat,
                    same_file_refs=same_file_refs,
                    src_refs=sorted(src_refs),
                    scripts_refs=sorted(scripts_refs),
                    tests_refs=sorted(tests_refs),
                    imported_by=sorted(imported_by),
                )
                symbols_list.append(sym_info)
                module_symbols_map[mod_name].append(sym_info)

        return (
            modules,
            edges,
            symbols_list,
            module_symbols_map,
            internal_calls_map,
            external_calls_map,
        )

    repo_root = Path(__file__).resolve().parent.parent
    (
        all_modules,
        all_edges,
        all_symbols,
        all_module_symbols,
        all_internal_calls,
        all_external_calls,
    ) = parse_repo_dependencies(repo_root)
    return (
        all_edges,
        all_external_calls,
        all_internal_calls,
        all_module_symbols,
        all_modules,
        all_symbols,
    )


@app.cell(hide_code=True)
def _(all_modules, mo):
    include_scripts = mo.ui.checkbox(value=False, label="Include scripts/")
    include_notebooks = mo.ui.checkbox(value=False, label="Include notebooks/")

    show_top_level = mo.ui.checkbox(value=True, label="Top-level runtime imports")
    show_deferred = mo.ui.checkbox(value=True, label="Deferred (function-local) imports")
    show_type_checking = mo.ui.checkbox(value=False, label="TYPE_CHECKING imports")

    show_edge_symbols = mo.ui.checkbox(value=False, label="Show imported symbols on edges")
    hide_isolated = mo.ui.checkbox(value=False, label="Hide isolated modules")
    group_by_subsystem = mo.ui.checkbox(value=True, label="Group Mermaid subgraphs")
    highlight_special = mo.ui.checkbox(value=True, label="Highlight cycle-breakers & cross-subpackage edges")
    mermaid_dir = mo.ui.dropdown(options=["TD", "LR"], value="TD", label="Mermaid orientation")

    # Dynamically assign color palette to all discovered subsystems across the repository
    _palette = [
        ("#eef6ff", "#2563eb"),  # blue
        ("#f0fdf4", "#16a34a"),  # green
        ("#fefce8", "#ca8a04"),  # amber
        ("#fdf2f8", "#db2777"),  # pink
        ("#f3f4f6", "#4b5563"),  # slate
        ("#f5f3ff", "#7c3aed"),  # violet
        ("#ecfeff", "#0891b2"),  # cyan
        ("#fff7ed", "#ea580c"),  # orange
        ("#fef2f2", "#dc2626"),  # red
        ("#f0fdfa", "#0d9488"),  # teal
    ]
    all_discovered_subsystems = sorted({m.subsystem for m in all_modules.values()})
    subsystem_colors = {sub: _palette[i % len(_palette)] for i, sub in enumerate(all_discovered_subsystems)}
    return (
        group_by_subsystem,
        hide_isolated,
        highlight_special,
        include_notebooks,
        include_scripts,
        mermaid_dir,
        show_deferred,
        show_edge_symbols,
        show_top_level,
        show_type_checking,
        subsystem_colors,
    )


@app.cell(hide_code=True)
def _(
    all_edges,
    all_modules,
    defaultdict,
    include_notebooks,
    include_scripts,
    mo,
):
    active_groups = {"src"}
    if include_scripts.value:
        active_groups.add("scripts")
    if include_notebooks.value:
        active_groups.add("notebooks")

    available_modules = {m: info for m, info in all_modules.items() if info.root_group in active_groups}
    available_subsystems = sorted({info.subsystem for info in available_modules.values()})

    # Compute incoming edge count so high-fan-in utility modules appear first in the exclusion selector
    in_counts: dict[str, int] = defaultdict(int)
    for _e in all_edges:
        if _e.source in available_modules and _e.target in available_modules:
            in_counts[_e.target] += 1

    src_count = max(1, sum(1 for info in available_modules.values() if info.root_group == "src"))
    default_excluded = [
        m
        for m, info in available_modules.items()
        if info.root_group == "src" and in_counts[m] >= 0.6 * src_count and not any(_e.source == m for _e in all_edges)
    ]

    ranked_for_exclusion = sorted(available_modules, key=lambda m: (-in_counts[m], m))

    subsystem_filter = mo.ui.multiselect(
        options=available_subsystems,
        value=available_subsystems,
        label="Subsystems",
    )
    excluded_modules_ui = mo.ui.multiselect(
        options=ranked_for_exclusion,
        value=default_excluded,
        label="Exclude modules (sorted by fan-in)",
    )
    focus_module = mo.ui.dropdown(
        options=["All modules (no focus)", *sorted(available_modules)],
        value="All modules (no focus)",
        label="Focus module",
    )
    focus_direction = mo.ui.dropdown(
        options=[
            "Both (upstream & downstream)",
            "Upstream (what focus imports)",
            "Downstream (what imports focus)",
        ],
        value="Both (upstream & downstream)",
        label="Focus direction",
    )
    focus_hops = mo.ui.slider(
        start=1,
        stop=6,
        step=1,
        value=1,
        label="Focus depth (hops)",
    )
    return (
        available_modules,
        excluded_modules_ui,
        focus_direction,
        focus_hops,
        focus_module,
        subsystem_filter,
    )


@app.cell(hide_code=True)
def _(
    excluded_modules_ui,
    focus_direction,
    focus_hops,
    focus_module,
    group_by_subsystem,
    hide_isolated,
    highlight_special,
    include_notebooks,
    include_scripts,
    mermaid_dir,
    mo,
    show_deferred,
    show_edge_symbols,
    show_top_level,
    show_type_checking,
    subsystem_filter,
):
    header = mo.md(
        """
        # 🕸️ Module Dependency & Dead Code Explorer

        Interactive AST dependency graph and symbol-level dead code detector for `src/` (with optional `scripts/` and `notebooks/` overlays).
        """
    )

    controls_panel = mo.vstack(
        [
            header,
            mo.hstack(
                [
                    subsystem_filter,
                    excluded_modules_ui,
                    focus_module,
                    focus_direction,
                    focus_hops,
                    mermaid_dir,
                ],
                justify="start",
                wrap=True,
            ),
            mo.hstack(
                [
                    include_scripts,
                    include_notebooks,
                    show_top_level,
                    show_deferred,
                    show_type_checking,
                    show_edge_symbols,
                    hide_isolated,
                    group_by_subsystem,
                    highlight_special,
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    controls_panel
    return


@app.cell(hide_code=True)
def _(
    all_edges,
    available_modules,
    defaultdict,
    deque,
    excluded_modules_ui,
    focus_direction,
    focus_hops,
    focus_module,
    hide_isolated,
    show_deferred,
    show_top_level,
    show_type_checking,
    subsystem_filter,
):
    allowed_kinds: set[str] = set()
    if show_top_level.value:
        allowed_kinds.add("top_level")
    if show_deferred.value:
        allowed_kinds.add("deferred")
    if show_type_checking.value:
        allowed_kinds.add("type_checking")

    excluded_mods = set(excluded_modules_ui.value)
    selected_subsystems = set(subsystem_filter.value)

    candidate_edges = [
        _e
        for _e in all_edges
        if _e.kind in allowed_kinds
        and _e.source in available_modules
        and _e.target in available_modules
        and _e.source not in excluded_mods
        and _e.target not in excluded_mods
    ]

    focus = focus_module.value
    if focus != "All modules (no focus)" and focus in available_modules:
        fwd_adj: dict[str, set[str]] = defaultdict(set)
        rev_adj: dict[str, set[str]] = defaultdict(set)
        for _e in candidate_edges:
            fwd_adj[_e.source].add(_e.target)
            rev_adj[_e.target].add(_e.source)

        def bfs_reach(start: str, adj: dict[str, set[str]], max_hops: int) -> set[str]:
            visited = {start}
            queue = deque([(start, 0)])
            while queue:
                curr, depth = queue.popleft()
                if depth >= max_hops:
                    continue
                for nxt in adj.get(curr, ()):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, depth + 1))
            return visited

        hops = int(focus_hops.value)
        mode = focus_direction.value
        upstream_nodes = bfs_reach(focus, fwd_adj, hops) if "Upstream" in mode or "Both" in mode else {focus}
        downstream_nodes = bfs_reach(focus, rev_adj, hops) if "Downstream" in mode or "Both" in mode else {focus}
        active_nodes = upstream_nodes | downstream_nodes

        filtered_edges = [
            _e
            for _e in candidate_edges
            if _e.source in active_nodes
            and _e.target in active_nodes
            and (
                ("Both" in mode)
                or ("Upstream" in mode and _e.source in upstream_nodes and _e.target in upstream_nodes)
                or ("Downstream" in mode and _e.source in downstream_nodes and _e.target in downstream_nodes)
            )
        ]
        filtered_modules = {
            _m: _info for _m, _info in available_modules.items() if _m in active_nodes and _m not in excluded_mods
        }
    else:
        filtered_modules = {
            _m: _info
            for _m, _info in available_modules.items()
            if _info.subsystem in selected_subsystems and _m not in excluded_mods
        }
        filtered_edges = [
            _e for _e in candidate_edges if _e.source in filtered_modules and _e.target in filtered_modules
        ]

    if hide_isolated.value:
        connected = {_e.source for _e in filtered_edges} | {_e.target for _e in filtered_edges}
        if focus != "All modules (no focus)" and focus in filtered_modules:
            connected.add(focus)
        filtered_modules = {_m: _info for _m, _info in filtered_modules.items() if _m in connected}
    return filtered_edges, filtered_modules, focus


@app.cell(hide_code=True)
def _(all_edges, available_modules, defaultdict):
    def is_cross_subpackage_edge(src_mod: str, dst_mod: str) -> bool:
        src_info = available_modules[src_mod]
        dst_info = available_modules[dst_mod]
        if src_info.root_group != "src" or dst_info.root_group != "src":
            return False
        src_sub = src_info.subsystem
        dst_sub = dst_info.subsystem
        if src_sub == dst_sub:
            return False
        # Peer subpackages at depth >= 2 (e.g. pkg.subA -> pkg.subB)
        return len(src_sub.split(".")) >= 2 and len(dst_sub.split(".")) >= 2

    def find_two_and_three_cycles(edges_subset):
        adj: dict[str, dict[str, str]] = defaultdict(dict)
        for _e in edges_subset:
            adj[_e.source][_e.target] = _e.kind

        cycles: list[tuple[tuple[str, ...], list[str]]] = []
        seen: set[frozenset[str]] = set()

        nodes = sorted(adj)
        for u in nodes:
            for v, k_uv in adj[u].items():
                if u in adj.get(v, {}):
                    key = frozenset((u, v))
                    if key not in seen:
                        seen.add(key)
                        k_vu = adj[v][u]
                        cycles.append(((u, v, u), [f"{u} -> {v} ({k_uv})", f"{v} -> {u} ({k_vu})"]))
                for w, k_vw in adj.get(v, {}).items():
                    if w != u and u in adj.get(w, {}):
                        key3 = frozenset((u, v, w))
                        if key3 not in seen:
                            seen.add(key3)
                            k_wu = adj[w][u]
                            cycles.append(
                                (
                                    (u, v, w, u),
                                    [
                                        f"{u} -> {v} ({k_uv})",
                                        f"{v} -> {w} ({k_vw})",
                                        f"{w} -> {u} ({k_wu})",
                                    ],
                                )
                            )
        return cycles

    active_all_edges = [_e for _e in all_edges if _e.source in available_modules and _e.target in available_modules]
    runtime_edges = [_e for _e in active_all_edges if _e.kind in {"top_level", "deferred"}]
    runtime_cycles = find_two_and_three_cycles(runtime_edges)
    all_kind_cycles = find_two_and_three_cycles(active_all_edges)

    deferred_edges = [_e for _e in active_all_edges if _e.kind == "deferred"]
    cross_subpackage_edges = [_e for _e in runtime_edges if is_cross_subpackage_edge(_e.source, _e.target)]
    return (
        all_kind_cycles,
        cross_subpackage_edges,
        deferred_edges,
        is_cross_subpackage_edge,
        runtime_cycles,
    )


@app.cell(hide_code=True)
def _(
    filtered_edges,
    filtered_modules,
    focus,
    group_by_subsystem,
    highlight_special,
    is_cross_subpackage_edge,
    mermaid_dir,
    show_edge_symbols,
    subsystem_colors,
):
    def node_id(mod_name: str) -> str:
        return "m_" + mod_name.replace(".", "_").replace("-", "_")

    def short_label(mod_name: str, subsystem: str) -> str:
        top_pkg = subsystem.split(".", maxsplit=1)[0]
        if mod_name.startswith(f"{top_pkg}.") and mod_name != top_pkg:
            return mod_name.removeprefix(f"{top_pkg}.")
        return mod_name

    def format_edge_label(symbols: set[str]) -> str:
        if not symbols or not show_edge_symbols.value:
            return ""
        sorted_syms = sorted(symbols)
        if len(sorted_syms) <= 2:
            return f'|"{", ".join(sorted_syms)}"|'
        return f'|"{sorted_syms[0]}, {sorted_syms[1]} (+{len(sorted_syms) - 2})"|'

    lines = [f"flowchart {mermaid_dir.value}"]

    if group_by_subsystem.value:
        by_sub: dict[str, list[str]] = {}
        for _m, _info in sorted(filtered_modules.items()):
            by_sub.setdefault(_info.subsystem, []).append(_m)

        for _sub_idx, (_sub_name, _mod_list) in enumerate(sorted(by_sub.items())):
            lines.append(f'    subgraph sg_{_sub_idx}["{_sub_name}"]')
            lines.extend(
                f'        {node_id(_m)}["{short_label(_m, filtered_modules[_m].subsystem)}"]' for _m in _mod_list
            )
            lines.append("    end")
    else:
        lines.extend(
            f'    {node_id(_m)}["{short_label(_m, filtered_modules[_m].subsystem)}"]' for _m in sorted(filtered_modules)
        )

    deferred_link_indices: list[int] = []
    tc_link_indices: list[int] = []
    cross_link_indices: list[int] = []

    for _edge_idx, _e in enumerate(filtered_edges):
        u = node_id(_e.source)
        v = node_id(_e.target)
        sym_lbl = format_edge_label(_e.symbols)

        if _e.kind == "top_level":
            lines.append(f"    {u} -->{sym_lbl} {v}")
        elif _e.kind == "deferred":
            lines.append(f"    {u} -.->|deferred{sym_lbl.replace('|', ' ')}| {v}")
            deferred_link_indices.append(_edge_idx)
        else:
            lines.append(f"    {u} -.->|tc{sym_lbl.replace('|', ' ')}| {v}")
            tc_link_indices.append(_edge_idx)

        if _e.kind == "top_level" and is_cross_subpackage_edge(_e.source, _e.target):
            cross_link_indices.append(_edge_idx)

    for _m, _info in sorted(filtered_modules.items()):
        fill, stroke = subsystem_colors.get(_info.subsystem, ("#ffffff", "#333333"))
        if _m == focus:
            lines.append(f"    style {node_id(_m)} fill:#fde68a,stroke:#b45309,stroke-width:3px,color:#111827")
        else:
            lines.append(f"    style {node_id(_m)} fill:{fill},stroke:{stroke},stroke-width:1.5px,color:#111827")

    if highlight_special.value:
        lines.extend(
            f"    linkStyle {_idx} stroke:#dc2626,stroke-width:2.5px,stroke-dasharray:5 5"
            for _idx in deferred_link_indices
        )
        lines.extend(f"    linkStyle {_idx} stroke:#7c3aed,stroke-width:2.5px" for _idx in cross_link_indices)
        lines.extend(
            f"    linkStyle {_idx} stroke:#9ca3af,stroke-width:1.2px,stroke-dasharray:3 3" for _idx in tc_link_indices
        )

    mermaid_source = "\n".join(lines)
    return (mermaid_source,)


@app.cell(hide_code=True)
def _(
    all_external_calls,
    all_internal_calls,
    all_module_symbols,
    defaultdict,
    filtered_edges,
    focus,
    mo,
    pd,
):
    if focus == "All modules (no focus)" or focus not in all_module_symbols:
        focus_view = mo.md(
            "> [!NOTE]\n"
            "> Select a specific **Focus module** in the top controls to inspect its exported API, "
            "dead code status, upstream imported symbols, and function/class call graph."
        )
    else:
        _syms = all_module_symbols.get(focus, [])
        _cat_badge = {
            "active": "🟢 Active",
            "internal_only": "🔵 Internal-Only",
            "test_only": "🟡 Test-Only",
            "unused": "🔴 Unused Anywhere",
        }

        sym_rows = [
            {
                "Symbol": f"{_s.parent_class}.{_s.name}" if _s.parent_class else _s.name,
                "Kind": _s.kind,
                "Line": _s.lineno,
                "Status": _cat_badge.get(_s.category, _s.category),
                "Privacy": "Private (_)" if _s.is_private else "Public",
                "Same-File Refs": _s.same_file_refs,
                "Src Refs": len(_s.src_refs),
                "Scripts/NB Refs": len(_s.scripts_refs),
                "Tests Refs": len(_s.tests_refs),
                "Imported By": ", ".join(_s.imported_by[:4]) + ("..." if len(_s.imported_by) > 4 else "")
                if _s.imported_by
                else "—",
            }
            for _s in sorted(_syms, key=lambda x: (x.category, x.name))
        ]
        focus_sym_df = pd.DataFrame(sym_rows)

        # Upstream imports for this module
        upstream_rows = [
            {
                "Imported Target Module": _e.target,
                "Kind": _e.kind,
                "Symbols Imported": ", ".join(sorted(_e.symbols)) if _e.symbols else "whole module",
                "Locations": ", ".join(_e.locations),
            }
            for _e in filtered_edges
            if _e.source == focus
        ]
        focus_up_df = pd.DataFrame(upstream_rows)

        # Build symbol-level Mermaid call graph for this module
        local_calls = all_internal_calls.get(focus, [])
        ext_calls = all_external_calls.get(focus, [])

        sym_mermaid_lines = ["flowchart LR"]
        sym_mermaid_lines.append(f'    subgraph sg_local["{focus} (Definitions)"]')
        for s in _syms:
            s_id = f"s_{s.name}_{s.lineno}"
            label = f"{s.parent_class}.{s.name}" if s.parent_class else s.name
            sym_mermaid_lines.append(f'        {s_id}["{label} ({s.kind})"]')
        sym_mermaid_lines.append("    end")

        # Local internal calls
        for caller, callee in local_calls:
            # find ids
            caller_s = next((s for _s in _syms if s.name == caller), None)
            callee_s = next((s for _s in _syms if s.name == callee), None)
            if caller_s and callee_s:
                sym_mermaid_lines.append(
                    f"    s_{caller_s.name}_{caller_s.lineno} --> s_{callee_s.name}_{callee_s.lineno}"
                )

        # External calls
        by_ext_mod: dict[str, set[str]] = defaultdict(set)
        for _caller, ext_mod, ext_sym in ext_calls:
            by_ext_mod[ext_mod].add(ext_sym)

        for ext_idx, (ext_mod, ext_sym_set) in enumerate(sorted(by_ext_mod.items())):
            sym_mermaid_lines.append(f'    subgraph sg_ext_{ext_idx}["{ext_mod}"]')
            sym_mermaid_lines.extend(f'        ext_{ext_idx}_{esym}["{esym}"]' for esym in sorted(ext_sym_set))
            sym_mermaid_lines.append("    end")

        for caller, ext_mod, ext_sym in ext_calls:
            caller_s = next((_s for _s in _syms if caller in (_s.name, f"{_s.parent_class}.{_s.name}")), None)
            ext_idx = [m for m, _ in sorted(by_ext_mod.items())].index(ext_mod)
            if caller_s:
                sym_mermaid_lines.append(f"    s_{caller_s.name}_{caller_s.lineno} -.-> ext_{ext_idx}_{ext_sym}")

        focus_call_graph_mmd = "\n".join(sym_mermaid_lines)

        focus_view = mo.vstack(
            [
                mo.md(f"## 🎯 Focus Drill-Down: `{focus}`"),
                mo.hstack(
                    [
                        mo.stat(value=f"{len(_syms)}", label="Symbols Defined"),
                        mo.stat(
                            value=f"{sum(1 for _s in _syms if s.category == 'active')}",
                            label="🟢 Active",
                        ),
                        mo.stat(
                            value=f"{sum(1 for _s in _syms if s.category == 'internal_only')}",
                            label="🔵 Internal-Only",
                        ),
                        mo.stat(
                            value=f"{sum(1 for _s in _syms if s.category == 'test_only')}",
                            label="🟡 Test-Only",
                        ),
                        mo.stat(
                            value=f"{sum(1 for _s in _syms if s.category == 'unused')}",
                            label="🔴 Unused Anywhere",
                        ),
                    ]
                ),
                mo.md("### 📦 Exported Symbols & Usage"),
                mo.ui.table(focus_sym_df, selection=None),
                mo.md("### 📥 Upstream Modules & Symbols Imported"),
                mo.ui.table(focus_up_df, selection=None)
                if not focus_up_df.empty
                else mo.md("_No upstream internal imports._"),
                mo.md("### 🔬 Symbol-Level Internal Call & Usage Graph"),
                mo.mermaid(focus_call_graph_mmd)
                if len(_syms) <= 40
                else mo.md("_Module has more than 40 symbols; see table above for full reference._"),
            ]
        )
    return (focus_view,)


@app.cell(hide_code=True)
def _(mo):
    category_options = [
        "All Categories",
        "🔴 Unused Anywhere (Dead Code)",
        "🟡 Test-Only",
        "🔵 Internal-Only (Private Candidate)",
        "🟢 Active in Production",
    ]
    kind_options = ["All Kinds", "function", "class", "method", "field"]

    cat_filter = mo.ui.dropdown(options=category_options, value="All Categories", label="Category")
    kind_filter = mo.ui.dropdown(options=kind_options, value="All Kinds", label="Symbol Kind")
    hide_priv_toggle = mo.ui.checkbox(value=False, label="Exclude private (_*) symbols")
    hide_dund_toggle = mo.ui.checkbox(value=True, label="Exclude dunder (__*__) methods")
    search_input = mo.ui.text(value="", label="Search symbol name")

    dead_code_controls = mo.hstack(
        [cat_filter, kind_filter, hide_priv_toggle, hide_dund_toggle, search_input],
        justify="start",
        wrap=True,
    )
    return (
        cat_filter,
        dead_code_controls,
        hide_dund_toggle,
        hide_priv_toggle,
        kind_filter,
        search_input,
    )


@app.cell(hide_code=True)
def _(
    all_symbols,
    cat_filter,
    dead_code_controls,
    hide_dund_toggle,
    hide_priv_toggle,
    kind_filter,
    mo,
    pd,
    search_input,
):
    cat_key_map = {
        "🔴 Unused Anywhere (Dead Code)": "unused",
        "🟡 Test-Only": "test_only",
        "🔵 Internal-Only (Private Candidate)": "internal_only",
        "🟢 Active in Production": "active",
    }
    _cat_badge = {
        "active": "🟢 Active",
        "internal_only": "🔵 Internal-Only",
        "test_only": "🟡 Test-Only",
        "unused": "🔴 Unused Anywhere",
    }

    filtered_syms = all_symbols
    if cat_filter.value in cat_key_map:
        target_cat = cat_key_map[cat_filter.value]
        filtered_syms = [_s for _s in filtered_syms if _s.category == target_cat]

    if kind_filter.value != "All Kinds":
        filtered_syms = [_s for _s in filtered_syms if _s.kind == kind_filter.value]

    if hide_priv_toggle.value:
        filtered_syms = [_s for _s in filtered_syms if not _s.is_private]

    if hide_dund_toggle.value:
        filtered_syms = [_s for _s in filtered_syms if not _s.is_dunder]

    if search_input.value.strip():
        q = search_input.value.strip().lower()
        filtered_syms = [_s for _s in filtered_syms if q in _s.name.lower() or q in _s.module.lower()]

    _rows = [
        {
            "Module": _s.module,
            "Symbol": f"{_s.parent_class}.{_s.name}" if _s.parent_class else _s.name,
            "Kind": _s.kind,
            "Line": _s.lineno,
            "Category": _cat_badge.get(_s.category, _s.category),
            "Privacy": "Private (_)" if _s.is_private else "Public",
            "Same-File Refs": _s.same_file_refs,
            "Src Refs": len(_s.src_refs),
            "Scripts/NB Refs": len(_s.scripts_refs),
            "Tests Refs": len(_s.tests_refs),
            "Imported By": ", ".join(_s.imported_by[:3]) + ("..." if len(_s.imported_by) > 3 else "")
            if _s.imported_by
            else "—",
        }
        for _s in filtered_syms
    ]
    sym_inventory_df = pd.DataFrame(_rows)

    # Actionable dead code list (public unused symbols)
    public_unused = [_s for _s in all_symbols if _s.category == "unused" and not _s.is_private and not _s.is_dunder]
    dead_code_summary = mo.md(
        f"""
        ### 🚨 Actionable Dead Code Summary ({len(public_unused)} public symbols unused anywhere)
        The following public definitions in `src/` have **zero references** across `src/`, `scripts/`, `notebooks/`, and `tests/`:

        | Module | Symbol | Kind | Line |
        | :--- | :--- | :--- | :--- |
        """
        + "\n".join(
            f"| `{_s.module}` | `{_s.parent_class + '.' if _s.parent_class else ''}{_s.name}` | {_s.kind} | L{_s.lineno} |"
            for _s in sorted(public_unused, key=lambda x: (x.module, x.lineno))
        )
    )

    dead_code_tab = mo.vstack(
        [
            mo.md("## 💀 Dead Code & Symbol Inventory Across `src/`"),
            mo.hstack(
                [
                    mo.stat(value=f"{len(all_symbols)}", label="Total Definitions"),
                    mo.stat(
                        value=f"{sum(1 for _s in all_symbols if _s.category == 'active')}",
                        label="🟢 Active",
                    ),
                    mo.stat(
                        value=f"{sum(1 for _s in all_symbols if _s.category == 'internal_only')}",
                        label="🔵 Internal-Only",
                    ),
                    mo.stat(
                        value=f"{sum(1 for _s in all_symbols if _s.category == 'test_only')}",
                        label="🟡 Test-Only",
                    ),
                    mo.stat(
                        value=f"{sum(1 for _s in all_symbols if _s.category == 'unused')}",
                        label="🔴 Unused Anywhere",
                    ),
                ]
            ),
            dead_code_controls,
            mo.md(f"**Showing {len(filtered_syms)} symbols:**"),
            mo.ui.table(sym_inventory_df, selection=None),
            dead_code_summary,
        ]
    )
    return (dead_code_tab,)


@app.cell(hide_code=True)
def _(
    defaultdict,
    filtered_edges,
    filtered_modules,
    focus,
    go,
    np,
    pd,
    subsystem_colors,
):
    adj_out: dict[str, list[str]] = defaultdict(list)
    adj_in: dict[str, list[str]] = defaultdict(list)
    for _e in filtered_edges:
        if _e.kind == "top_level":
            adj_out[_e.source].append(_e.target)
            adj_in[_e.target].append(_e.source)

    memo_rank: dict[str, int] = {}
    visiting: set[str] = set()

    def get_rank(m: str) -> int:
        if m in memo_rank:
            return memo_rank[m]
        if m in visiting:
            return 0
        visiting.add(m)
        targets = [t for t in adj_out.get(m, []) if t in filtered_modules]
        rank = 0 if not targets else 1 + max(get_rank(t) for t in targets)
        visiting.remove(m)
        memo_rank[m] = rank
        return rank

    for _m in sorted(filtered_modules):
        get_rank(_m)

    visible_subsystems = sorted({info.subsystem for info in filtered_modules.values()})
    subsystem_x_band = {sub: float(idx) * 1.4 for idx, sub in enumerate(visible_subsystems)}

    rank_groups: dict[int, list[str]] = defaultdict(list)
    for _m in sorted(
        filtered_modules,
        key=lambda x: (subsystem_x_band.get(filtered_modules[x].subsystem, 0.0), x),
    ):
        rank_groups[memo_rank[_m]].append(_m)

    pos: dict[str, tuple[float, float]] = {}
    for _rank, _mods_in_rank in rank_groups.items():
        _n = len(_mods_in_rank)
        for _i, _m in enumerate(_mods_in_rank):
            _sub_center = subsystem_x_band.get(filtered_modules[_m].subsystem, 0.0)
            _offset = (_i - (_n - 1) / 2.0) * 0.55
            pos[_m] = (_sub_center * 1.8 + _offset, float(_rank))

    network_fig = go.Figure()

    edge_styles = {
        "top_level": ("Top-level runtime", "#475569", "solid", 1.6),
        "deferred": ("Deferred (cycle-breaker)", "#dc2626", "dash", 2.4),
        "type_checking": ("TYPE_CHECKING", "#94a3b8", "dot", 1.2),
    }

    for _kind, (_label, _color, _dash, _width) in edge_styles.items():
        _kind_edges = [_e for _e in filtered_edges if _e.kind == _kind]
        if not _kind_edges:
            continue
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        mid_x: list[float] = []
        mid_y: list[float] = []
        hover_texts: list[str] = []

        for _e in _kind_edges:
            x0, y0 = pos[_e.source]
            x1, y1 = pos[_e.target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            mid_x.append(0.65 * x0 + 0.35 * x1)
            mid_y.append(0.65 * y0 + 0.35 * y1)
            syms = ", ".join(sorted(_e.symbols))
            locs = ", ".join(_e.locations)
            hover_texts.append(
                f"<b>{_e.source} → {_e.target}</b><br>Kind: {_label}<br>Symbols: {syms}<br>Locations: {locs}"
            )

        network_fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": _width, "color": _color, "dash": _dash},
                hoverinfo="none",
                name=_label,
            )
        )
        network_fig.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers",
                marker={"size": 7, "color": _color, "symbol": "triangle-down", "opacity": 0.75},
                text=hover_texts,
                hoverinfo="text",
                showlegend=False,
            )
        )

    out_edges_by_mod: dict[str, list[str]] = defaultdict(list)
    in_edges_by_mod: dict[str, list[str]] = defaultdict(list)
    for _e in filtered_edges:
        out_edges_by_mod[_e.source].append(f"{_e.target} ({_e.kind})")
        in_edges_by_mod[_e.target].append(f"{_e.source} ({_e.kind})")

    for _sub_name in visible_subsystems:
        _sub_color = subsystem_colors.get(_sub_name, ("#eef6ff", "#2563eb"))[1]
        _sub_mods = [_m for _m, _info in sorted(filtered_modules.items()) if _info.subsystem == _sub_name]
        if not _sub_mods:
            continue
        _top_pkg = _sub_name.split(".", maxsplit=1)[0]
        nx = [pos[_m][0] for _m in _sub_mods]
        ny = [pos[_m][1] for _m in _sub_mods]
        labels = [
            _m.removeprefix(f"{_top_pkg}.") if _m.startswith(f"{_top_pkg}.") and _m != _top_pkg else _m
            for _m in _sub_mods
        ]
        sizes = [
            22 if _m == focus else min(28, 11 + 2 * (len(out_edges_by_mod[_m]) + len(in_edges_by_mod[_m])))
            for _m in _sub_mods
        ]
        line_widths = [3.5 if _m == focus else 1.5 for _m in _sub_mods]
        line_colors = ["#b45309" if _m == focus else "#ffffff" for _m in _sub_mods]
        hovers = [
            f"<b>{_m}</b> ({filtered_modules[_m].rel_path}, {filtered_modules[_m].loc} LOC)<br>"
            f"Subsystem: {_sub_name} | Layer depth: {memo_rank[_m]}<br>"
            f"Imports ({len(out_edges_by_mod[_m])}): {', '.join(out_edges_by_mod[_m][:8]) or 'None'}"
            f"{' ...' if len(out_edges_by_mod[_m]) > 8 else ''}<br>"
            f"Imported by ({len(in_edges_by_mod[_m])}): {', '.join(in_edges_by_mod[_m][:8]) or 'None'}"
            f"{' ...' if len(in_edges_by_mod[_m]) > 8 else ''}"
            for _m in _sub_mods
        ]

        network_fig.add_trace(
            go.Scatter(
                x=nx,
                y=ny,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont={"size": 10},
                hovertext=hovers,
                hoverinfo="text",
                marker={
                    "size": sizes,
                    "color": _sub_color,
                    "line": {"width": line_widths, "color": line_colors},
                },
                name=_sub_name,
            )
        )

    network_fig.update_layout(
        title="Layered Dependency DAG (Y = dependency depth above leaf modules, hover edges for imported symbols)",
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "title": ""},
        yaxis={"showgrid": True, "zeroline": False, "dtick": 1, "title": "Dependency Layer Depth"},
        height=720,
        margin={"l": 40, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        plot_bgcolor="#fafafa",
    )

    ordered_mods = sorted(
        filtered_modules,
        key=lambda m: (filtered_modules[m].subsystem, m),
    )
    idx_map = {m: i for i, m in enumerate(ordered_mods)}
    n_mods = len(ordered_mods)
    mat = np.zeros((n_mods, n_mods), dtype=int)
    hover_mat = [["" for _ in range(n_mods)] for _ in range(n_mods)]

    kind_code = {"top_level": 1, "deferred": 2, "type_checking": 3}
    for i, src_m in enumerate(ordered_mods):
        for j, dst_m in enumerate(ordered_mods):
            hover_mat[i][j] = f"<b>{src_m} → {dst_m}</b><br>No dependency"

    for _e in filtered_edges:
        i = idx_map[_e.source]
        j = idx_map[_e.target]
        mat[i, j] = kind_code[_e.kind]
        hover_mat[i][j] = (
            f"<b>{_e.source} → {_e.target}</b><br>"
            f"Kind: {_e.kind}<br>"
            f"Symbols: {', '.join(sorted(_e.symbols))}<br>"
            f"Locations: {', '.join(_e.locations)}"
        )

    matrix_fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=ordered_mods,
            y=ordered_mods,
            text=hover_mat,
            hoverinfo="text",
            colorscale=[
                [0.0, "#f8fafc"],
                [0.33, "#2563eb"],
                [0.66, "#dc2626"],
                [1.0, "#94a3b8"],
            ],
            zmin=0,
            zmax=3,
            showscale=False,
            xgap=1,
            ygap=1,
        )
    )
    matrix_fig.update_layout(
        title="Adjacency Matrix (Row imports Column: Blue = Top-level, Red = Deferred, Gray = TYPE_CHECKING)",
        xaxis={"title": "Imported Module (Target)", "tickangle": 45, "side": "bottom"},
        yaxis={"title": "Importer Module (Source)", "autorange": "reversed"},
        height=max(550, n_mods * 18 + 160),
        margin={"l": 180, "r": 20, "t": 60, "b": 140},
    )

    _metric_rows = []
    for _m in ordered_mods:
        _info = filtered_modules[_m]
        top_out = [_e.target for _e in filtered_edges if _e.source == _m and _e.kind == "top_level"]
        def_out = [_e.target for _e in filtered_edges if _e.source == _m and _e.kind == "deferred"]
        tc_out = [_e.target for _e in filtered_edges if _e.source == _m and _e.kind == "type_checking"]
        fan_out = len(top_out) + len(def_out)
        fan_in = sum(1 for _e in filtered_edges if _e.target == _m and _e.kind in {"top_level", "deferred"})
        instability = round(fan_out / (fan_in + fan_out), 2) if (fan_in + fan_out) > 0 else 0.0
        _metric_rows.append(
            {
                "Module": _m,
                "Group": _info.root_group,
                "Subsystem": _info.subsystem,
                "LOC": _info.loc,
                "Layer Depth": memo_rank[_m],
                "Runtime Fan-In": fan_in,
                "Runtime Fan-Out": fan_out,
                "Instability I": instability,
                "Top-Level Imports": ", ".join(top_out),
                "Deferred Imports": ", ".join(def_out),
                "TYPE_CHECKING Imports": ", ".join(tc_out),
            }
        )
    metrics_df = pd.DataFrame(_metric_rows)
    return matrix_fig, metrics_df, network_fig


@app.cell(hide_code=True)
def _(
    all_kind_cycles,
    cross_subpackage_edges,
    dead_code_tab,
    deferred_edges,
    filtered_edges,
    filtered_modules,
    focus_view,
    matrix_fig,
    mermaid_source,
    metrics_df,
    mo,
    network_fig,
    runtime_cycles,
):
    mermaid_download = mo.download(
        data=mermaid_source.encode("utf-8"),
        filename="dependency_graph.mmd",
        mimetype="text/plain",
        label="Download Mermaid source (.mmd)",
    )

    mermaid_tab = mo.vstack(
        [
            mo.md(
                f"**Showing {len(filtered_modules)} modules and {len(filtered_edges)} dependency edges.** "
                "Use the built-in diagram toolbar (top-right of diagram) to zoom, pan, or export as SVG."
            ),
            mo.mermaid(mermaid_source),
            mo.accordion(
                {
                    "Raw Mermaid Source & Download": mo.vstack(
                        [mermaid_download, mo.md(f"```mermaid\n{mermaid_source}\n```")]
                    )
                }
            ),
        ]
    )

    plotly_tab = mo.vstack(
        [
            mo.ui.plotly(network_fig),
            mo.ui.plotly(matrix_fig),
        ]
    )

    deferred_lines = [
        f"- **`{e.source}` → `{e.target}`** at {', '.join(e.locations)}: imports `{', '.join(sorted(e.symbols))}`"
        for e in deferred_edges
    ]
    runtime_cycle_lines = [f"- **{' → '.join(path)}**: {'; '.join(step_desc)}" for path, step_desc in runtime_cycles]
    all_cycle_lines = [f"- **{' → '.join(path)}**: {'; '.join(step_desc)}" for path, step_desc in all_kind_cycles]
    cross_lines = [
        f"- **`{e.source}` → `{e.target}`** ({e.kind}, {', '.join(e.locations)}): `{', '.join(sorted(e.symbols))}`"
        for e in cross_subpackage_edges
    ]

    diagnostics_md = mo.md(
        f"""
        ### 🔄 Runtime Cycles (Top-Level + Deferred Imports)
        {chr(10).join(runtime_cycle_lines) if runtime_cycle_lines else "- None"}

        ### ⏱️ Deferred (Function-Local) Imports ({len(deferred_edges)} total)
        These imports occur inside function bodies rather than at module top level:
        {chr(10).join(deferred_lines) if deferred_lines else "- None"}

        ### 🔗 Peer Subpackage Runtime Couplings ({len(cross_subpackage_edges)} total)
        Edges where a nested subpackage in `src/` imports from a sibling subpackage:
        {chr(10).join(cross_lines) if cross_lines else "- None"}

        ### 🏷️ All 2- and 3-Node Cycles Including `TYPE_CHECKING` ({len(all_kind_cycles)} total)
        {chr(10).join(all_cycle_lines) if all_cycle_lines else "- None"}
        """
    )

    diagnostics_tab = mo.vstack(
        [
            diagnostics_md,
            mo.md("### 📊 Module Metrics & Instability Table"),
            mo.ui.table(metrics_df, selection=None),
        ]
    )

    tabs = mo.ui.tabs(
        {
            "Mermaid Dependency Graph": mermaid_tab,
            "Focus & Symbol Call Graph": focus_view,
            "Dead Code & Symbol Inventory": dead_code_tab,
            "Plotly Network & Adjacency Matrix": plotly_tab,
            "Cycle & Architecture Diagnostics": diagnostics_tab,
        }
    )
    tabs
    return


if __name__ == "__main__":
    app.run()
