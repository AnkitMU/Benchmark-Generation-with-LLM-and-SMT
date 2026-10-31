"""
Exact upper bound for the steered slot-assignment problem.

The outer layer of steered generation assigns each compiled slot
(family x polarity x complexity box) to a (pattern, class) pair whose
profiled footprint supports the slot's box, subject to the per-class
ceiling U. Relaxing the suite-level effects the greedy fill also faces
(duplicate suppression, per-context joint consistency, binding search),
the maximum number of simultaneously fillable slots is exactly a
bipartite b-matching:

    slots (unit supply)  --edge iff some profiled in-family pattern
                           reaches the slot's box for that class-->
    classes (capacity U) --> sink

Maximum matching = max flow, computed here with Dinic's algorithm in
pure Python (no solver dependency). The result is a certified optimum
N* of the relaxation, hence a sound upper bound on any fill strategy;
the realised suite size N gives the optimality ratio N / N* surfaced
in the steered deviation report.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple


class _Dinic:
    """Dinic max-flow on a small layered graph (source -> slots -> classes -> sink)."""

    def __init__(self, n: int):
        self.n = n
        self.adj: List[List[int]] = [[] for _ in range(n)]  # node -> edge ids
        self.to: List[int] = []
        self.cap: List[int] = []

    def add_edge(self, u: int, v: int, c: int) -> None:
        # Forward edge at even id, residual at odd id: residual(e) == e ^ 1.
        self.adj[u].append(len(self.to)); self.to.append(v); self.cap.append(c)
        self.adj[v].append(len(self.to)); self.to.append(u); self.cap.append(0)

    def _bfs_levels(self, s: int, t: int) -> List[int]:
        level = [-1] * self.n
        level[s] = 0
        queue = [s]
        for u in queue:
            for eid in self.adj[u]:
                v = self.to[eid]
                if self.cap[eid] > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        return level if level[t] >= 0 else []

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        while True:
            level = self._bfs_levels(s, t)
            if not level:
                return flow
            it = [0] * self.n

            def dfs(u: int, f: int) -> int:
                if u == t:
                    return f
                while it[u] < len(self.adj[u]):
                    eid = self.adj[u][it[u]]
                    v = self.to[eid]
                    if self.cap[eid] > 0 and level[v] == level[u] + 1:
                        d = dfs(v, min(f, self.cap[eid]))
                        if d > 0:
                            self.cap[eid] -= d
                            self.cap[eid ^ 1] += d
                            return d
                    it[u] += 1
                return 0

            while True:
                pushed = dfs(s, 10 ** 9)
                if pushed == 0:
                    break
                flow += pushed


def max_fill_assignment(slot_classes: Sequence[Set[str]],
                        class_caps: Dict[str, int]
                        ) -> Tuple[int, List[Optional[str]]]:
    """Exact optimum of the outer assignment relaxation, WITH a witness.

    Args:
        slot_classes: per slot, the set of classes with at least one profiled
            in-family (pattern, class) pair whose footprint reaches the slot's
            box -- the same support predicate the greedy fill uses.
        class_caps: per-class ceiling U on accepted constraints.

    Returns:
        ``(optimum, assignment)`` where ``optimum`` is the maximum number of
        slots simultaneously assignable to a supporting class without
        exceeding any class ceiling, and ``assignment[i]`` is slot ``i``'s
        class in one such optimal matching (``None`` for slots left out).
        The witness lets the fill loop TARGET the optimum instead of merely
        being measured against it.
    """
    n_slots = len(slot_classes)
    assignment: List[Optional[str]] = [None] * n_slots
    classes = sorted({c for cs in slot_classes for c in cs
                      if class_caps.get(c, 0) > 0})
    if not classes:
        return 0, assignment
    idx = {c: i for i, c in enumerate(classes)}
    source = 0
    base_slot = 1
    base_cls = base_slot + n_slots
    sink = base_cls + len(classes)

    g = _Dinic(sink + 1)
    for i, cs in enumerate(slot_classes):
        g.add_edge(source, base_slot + i, 1)
        for c in cs:
            j = idx.get(c)
            if j is not None:
                g.add_edge(base_slot + i, base_cls + j, 1)
    for c, j in idx.items():
        g.add_edge(base_cls + j, sink, max(0, int(class_caps[c])))
    flow = g.max_flow(source, sink)

    # Recover the matching: a slot->class forward edge (even id, unit
    # capacity) is used iff saturated.
    cls_of_node = {base_cls + j: c for c, j in idx.items()}
    for i in range(n_slots):
        for eid in g.adj[base_slot + i]:
            if eid % 2 == 0 and g.cap[eid] == 0:
                assignment[i] = cls_of_node[g.to[eid]]
                break
    return flow, assignment


def max_fill_upper_bound(slot_classes: Sequence[Set[str]],
                         class_caps: Dict[str, int]) -> int:
    """Exact optimum of the outer assignment relaxation (value only)."""
    return max_fill_assignment(slot_classes, class_caps)[0]
