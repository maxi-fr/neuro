import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="Module Dependency Explorer")


@app.cell(hide_code=True)
def _():
    import ast
    from collections import defaultdict, deque
    from dataclasses import dataclass, field
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    return Path, ast, dataclass, defaultdict, deque, field, go, mo, np, pd


@app.cell(hide_code=True)
def _(Path, ast, dataclass, defaultdict, field):
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

    def is_marimo_cell(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr == "cell":
                return True
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "cell":
                return True
        return False

    def parse_repo_dependencies(repo_root: Path) -> tuple[dict[str, ModuleInfo], list[EdgeInfo]]:
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

        for mod_name, (p, root_group, base_root) in file_map.items():
            rel = p.relative_to(base_root)
            tree = ast.parse(p.read_text(encoding="utf-8"))

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

        return modules, edges

    repo_root = Path(__file__).resolve().parent.parent
    all_modules, all_edges = parse_repo_dependencies(repo_root)
    return all_edges, all_modules


@app.cell(hide_code=True)
def _(all_modules, mo):
    include_scripts = mo.ui.checkbox(value=False, label="Include scripts/")
    include_notebooks = mo.ui.checkbox(value=False, label="Include notebooks/")

    show_top_level = mo.ui.checkbox(value=True, label="Top-level runtime imports")
    show_deferred = mo.ui.checkbox(value=True, label="Deferred (function-local) imports")
    show_type_checking = mo.ui.checkbox(value=False, label="TYPE_CHECKING imports")

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
    show_top_level,
    show_type_checking,
    subsystem_filter,
):
    header = mo.md(
        """
        # 🕸️ Module Dependency Explorer

        Filesystem-driven AST dependency graph for `src/`, with optional `scripts/` and `notebooks/` overlays.
        An arrow **`A --> B`** means **`A` imports `B`**.
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
    subsystem_colors,
):
    def node_id(mod_name: str) -> str:
        return "m_" + mod_name.replace(".", "_").replace("-", "_")

    def short_label(mod_name: str, subsystem: str) -> str:
        # Strip shared top-level package prefix only when inside a multi-part package
        top_pkg = subsystem.split(".", maxsplit=1)[0]
        if mod_name.startswith(f"{top_pkg}.") and mod_name != top_pkg:
            return mod_name.removeprefix(f"{top_pkg}.")
        return mod_name

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
        if _e.kind == "top_level":
            lines.append(f"    {u} --> {v}")
        elif _e.kind == "deferred":
            lines.append(f"    {u} -.->|deferred| {v}")
            deferred_link_indices.append(_edge_idx)
        else:
            lines.append(f"    {u} -.->|tc| {v}")
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
        _top_pkg = _sub_name.split(".")[0]
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

    rows = []
    for _m in ordered_mods:
        _info = filtered_modules[_m]
        top_out = [_e.target for _e in filtered_edges if _e.source == _m and _e.kind == "top_level"]
        def_out = [_e.target for _e in filtered_edges if _e.source == _m and _e.kind == "deferred"]
        tc_out = [_e.target for _e in filtered_edges if _e.source == _m and _e.kind == "type_checking"]
        fan_out = len(top_out) + len(def_out)
        fan_in = sum(1 for _e in filtered_edges if _e.target == _m and _e.kind in {"top_level", "deferred"})
        instability = round(fan_out / (fan_in + fan_out), 2) if (fan_in + fan_out) > 0 else 0.0
        rows.append(
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
    metrics_df = pd.DataFrame(rows)
    return matrix_fig, metrics_df, network_fig


@app.cell(hide_code=True)
def _(
    all_kind_cycles,
    cross_subpackage_edges,
    deferred_edges,
    filtered_edges,
    filtered_modules,
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
            "Plotly Network & Adjacency Matrix": plotly_tab,
            "Cycle & Architecture Diagnostics": diagnostics_tab,
        }
    )
    tabs
    return


if __name__ == "__main__":
    app.run()
