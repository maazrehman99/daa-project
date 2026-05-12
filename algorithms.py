"""
FoodPappa – Algorithm Implementations
Dijkstra, A*, and Bellman-Ford for graph-based shortest-path routing.

Why three algorithms?
  Dijkstra and A* are fast (O((V+E)logV)) but cannot handle negative edge
  weights — they would give incorrect results on the express lanes.
  Bellman-Ford is slower (O(V·E)) but handles negative weights correctly,
  and can find shorter routes that exploit the express-lane shortcuts.
  Running all three lets us compare trade-offs in speed vs. path quality.
"""

import heapq
import math
import time
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _dist(positions: Dict[int, Tuple[float, float]], u: int, v: int) -> float:
    """Euclidean distance between two nodes using their (possibly jittered) positions.

    Used both as the A* heuristic base and for computing physical distance
    along a reconstructed path (separate from the weighted cost).
    """
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _reconstruct(prev: Dict[int, int], src: int, dst: int) -> Optional[List[int]]:
    """
    Walk the predecessor map backwards from dst to src to recover the path.

    The predecessor map (prev[v] = u) is built during search — each time we
    find a better route to v via u, we record u as v's predecessor.
    Walking backwards from dst gives the shortest path in reverse order.

    A visited-set guards against cycles in the predecessor map (shouldn't
    happen in correct graphs, but defensive coding is worthwhile here).
    """
    path, cur = [], dst
    visited = set()

    while cur != -1:
        if cur in visited:
            return None  # cycle in predecessor map — graph likely has a negative cycle
        visited.add(cur)
        path.append(cur)
        if cur == src:
            break                          # reached the source — stop walking back
        cur = prev.get(cur, -1)            # -1 sentinel means "no predecessor recorded"

    if not path or path[-1] != src:
        return None   # dst is unreachable from src

    return list(reversed(path))   # reverse so the path reads src → ... → dst


def _path_distance(path: List[int], positions: Dict[int, Tuple[float, float]]) -> float:
    """
    Compute the total *physical* (Euclidean) length of a path.

    This is intentionally different from the weighted cost:
      - cost   = sum of (distance × traffic_factor) over each edge
      - distance = sum of raw Euclidean distances (for fuel calculation)
    """
    if not path or len(path) < 2:
        return 0.0
    return sum(_dist(positions, path[i], path[i + 1]) for i in range(len(path) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# DIJKSTRA
# ─────────────────────────────────────────────────────────────────────────────

def dijkstra(
    adj: Dict[int, List[Tuple[int, float]]],
    positions: Dict[int, Tuple[float, float]],
    src: int,
    dst: int,
) -> dict:
    """
    Classic Dijkstra's algorithm.
    - Guarantees shortest path for non-negative weights.
    - Skips negative-weight edges (undefined behaviour otherwise).
    - Time complexity: O((V + E) log V)
    """
    t0 = time.perf_counter()
    N = len(adj)   # used only for sizing if needed; Dijkstra is node-driven

    # dist[n] = best known cost from src to n.
    # Initialise to infinity (unreachable) for all nodes except the source.
    dist = {n: math.inf for n in adj}
    dist[src] = 0.0

    # prev[v] = the node we arrived from when we found the best route to v.
    # Together with dist, this lets us reconstruct the full path after search.
    prev: Dict[int, int] = {}

    # Closed set: once a node is finalised (popped from the heap with the
    # lowest possible cost), we never update it again.  This is the key
    # correctness invariant for Dijkstra on non-negative graphs.
    visited = set()

    # Min-heap entries are (cost, node_id).
    # We use a lazy-deletion approach: if a stale (higher-cost) entry is popped
    # for an already-visited node, we just skip it.
    pq = [(0.0, src)]
    explored = 0   # count nodes finalised — useful for comparing algorithm efficiency

    while pq:
        d, u = heapq.heappop(pq)   # extract the node with the currently lowest known cost

        if u in visited:
            continue   # stale heap entry — a better path to u was already finalised

        visited.add(u)
        explored += 1

        # Early exit: once the destination is finalised, its distance is optimal.
        # We don't need to continue expanding the rest of the graph.
        if u == dst:
            break

        for v, w in adj[u]:
            if w < 0:
                # Dijkstra's correctness proof requires non-negative weights.
                # A negative edge would allow a node already in 'visited' to be
                # updated again, breaking the greedy selection invariant.
                continue  # Dijkstra cannot handle negative weights — skip express lanes

            nd = d + w   # tentative cost to reach v through u
            if nd < dist[v]:
                # Found a better path to v — update and push to heap
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    path = _reconstruct(prev, src, dst)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    physical_dist = _path_distance(path, positions) if path else math.inf

    return {
        "path": path,
        "cost": dist[dst],               # weighted shortest-path cost
        "distance": physical_dist,       # raw Euclidean path length (for fuel calc)
        "nodes_explored": explored,
        "time_ms": elapsed_ms,
        "handles_negative": False,       # flag used by the UI to show algorithm capabilities
    }


# ─────────────────────────────────────────────────────────────────────────────
# A* (A-STAR)
# ─────────────────────────────────────────────────────────────────────────────

def astar(
    adj: Dict[int, List[Tuple[int, float]]],
    positions: Dict[int, Tuple[float, float]],
    src: int,
    dst: int,
) -> dict:
    """
    A* algorithm with Euclidean heuristic.
    - Faster than Dijkstra in practice by guiding search toward destination.
    - Admissible heuristic: straight-line distance × min_traffic_factor (0.78).
    - Skips negative-weight edges (same limitation as Dijkstra).
    - Time complexity: O((V + E) log V) worst-case, better in practice.
    """
    # The heuristic h(n) estimates the remaining cost from n to dst.
    # Multiplying Euclidean distance by MIN_TRAFFIC (≤ actual minimum traffic)
    # keeps the heuristic *admissible* — it never overestimates the true cost.
    # Admissibility guarantees that A* finds the optimal path.
    MIN_TRAFFIC = 0.78  # heuristic scale factor (must be ≤ min edge traffic = 0.8)

    def h(u: int) -> float:
        # Straight-line distance to destination scaled by the lightest possible
        # traffic factor.  This is always ≤ the actual edge weight, so the
        # heuristic never overestimates (admissibility condition).
        return _dist(positions, u, dst) * MIN_TRAFFIC

    t0 = time.perf_counter()

    # g[n] = best known cost from src to n (same role as dist[] in Dijkstra)
    g: Dict[int, float] = {n: math.inf for n in adj}
    g[src] = 0.0

    prev: Dict[int, int] = {}

    # closed set: nodes whose optimal cost has been confirmed
    closed = set()

    # Heap entries are (f_score, node_id) where f = g + h.
    # f is our best guess at total path cost through this node.
    pq = [(h(src), src)]
    explored = 0

    while pq:
        f, u = heapq.heappop(pq)   # expand the node with the best estimated total cost

        if u in closed:
            continue   # lazy deletion — this is a stale entry

        closed.add(u)
        explored += 1

        # A* terminates early for the same reason as Dijkstra: once the goal
        # is popped from the heap, its g-score is provably optimal.
        if u == dst:
            break

        for v, w in adj[u]:
            if w < 0:
                continue  # A* also cannot safely handle negative weights

            ng = g[u] + w   # tentative g-score for v via u

            if ng < g[v]:
                # Better path to v found — update g and push new f estimate
                g[v] = ng
                prev[v] = u
                # f(v) = g(v) + h(v): known cost so far + heuristic remaining cost
                heapq.heappush(pq, (ng + h(v), v))

    path = _reconstruct(prev, src, dst)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    physical_dist = _path_distance(path, positions) if path else math.inf

    return {
        "path": path,
        "cost": g[dst],
        "distance": physical_dist,
        "nodes_explored": explored,
        "time_ms": elapsed_ms,
        "handles_negative": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BELLMAN-FORD
# ─────────────────────────────────────────────────────────────────────────────

def bellman_ford(
    adj: Dict[int, List[Tuple[int, float]]],
    positions: Dict[int, Tuple[float, float]],
    src: int,
    dst: int,
) -> dict:
    """
    Bellman-Ford algorithm.
    - Handles negative-weight edges correctly.
    - Detects negative cycles.
    - Time complexity: O(V × E) — significantly slower than Dijkstra/A*.
    - Uses early termination when no updates occur in a full pass.
    """
    t0 = time.perf_counter()
    N = len(adj)   # number of nodes — Bellman-Ford needs exactly N-1 relaxation passes

    # Flatten adjacency lists into a single edge list for efficient iteration.
    # Bellman-Ford repeatedly scans ALL edges, so a flat list avoids repeated
    # dict iteration overhead compared to iterating adj.items() each pass.
    edges: List[Tuple[int, int, float]] = [
        (u, v, w) for u, neighbors in adj.items() for v, w in neighbors
    ]

    # dist[n] = best known cost from src to n, initialised to infinity.
    dist = {n: math.inf for n in adj}
    dist[src] = 0.0

    prev: Dict[int, int] = {}
    iterations = 0   # actual number of passes performed (may be < N-1 due to early stop)

    # ── Main relaxation loop ────────────────────────────────────────────────
    # Bellman-Ford's correctness: after k passes, dist[v] holds the shortest
    # path from src to v using at most k edges.  Since any simple path has
    # at most N-1 edges, N-1 passes suffice (unless there's a negative cycle).
    for iteration in range(N - 1):
        updated = False   # track whether any improvement was made this pass

        for u, v, w in edges:
            # Only relax from u if u is reachable (dist[u] < inf).
            # Without this guard, inf + w might wrap or propagate incorrectly.
            if dist[u] < math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True   # at least one improvement found this pass

        iterations += 1

        if not updated:
            # No improvements in this entire pass → all distances have converged.
            # This early termination can cut iterations from N-1 (~9999) to just
            # a few dozen on graphs with clear shortest-path structure.
            break  # converged early

    # ── Negative-cycle detection ─────────────────────────────────────────────
    # Run one more full relaxation pass.  If any distance can *still* be
    # reduced after N-1 passes, there must be a negative cycle reachable from
    # src — because a simple shortest path can't have more than N-1 edges.
    has_neg_cycle = False
    for u, v, w in edges:
        if dist[u] < math.inf and dist[u] + w < dist[v]:
            has_neg_cycle = True
            break

    path = _reconstruct(prev, src, dst)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    physical_dist = _path_distance(path, positions) if path else math.inf

    return {
        "path": path,
        "cost": dist[dst],
        "distance": physical_dist,
        "nodes_explored": N,    # BF processes all N nodes each iteration — no selective expansion
        "iterations": iterations,
        "has_negative_cycle": has_neg_cycle,
        "time_ms": elapsed_ms,
        "handles_negative": True,   # BF's main advantage over Dijkstra/A*
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED RUN (two-leg journey: rider → restaurant → customer)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_route(
    algo_fn,
    adj: Dict[int, List[Tuple[int, float]]],
    positions: Dict[int, Tuple[float, float]],
    rider_start: int,
    restaurant: int,
    destination: int,
) -> dict:
    """
    Run algo_fn for both legs and combine metrics.

    A delivery has two legs:
      Leg 1: rider's current position  →  restaurant  (pick up food)
      Leg 2: restaurant                →  customer destination (deliver food)

    Combining into a single function call keeps app.py clean and ensures
    that the same algorithm is used for both legs (fair comparison).
    """
    leg1 = algo_fn(adj, positions, rider_start, restaurant)   # pick-up leg
    leg2 = algo_fn(adj, positions, restaurant, destination)   # delivery leg

    # Sum costs; treat inf (unreachable) as 0 to avoid propagating inf into
    # the total (UI handles display of partial failures separately).
    total_cost = (leg1["cost"] if math.isfinite(leg1["cost"]) else 0) + \
                 (leg2["cost"] if math.isfinite(leg2["cost"]) else 0)
    total_dist = (leg1["distance"] if math.isfinite(leg1["distance"]) else 0) + \
                 (leg2["distance"] if math.isfinite(leg2["distance"]) else 0)

    # Fuel cost is proportional to physical distance, not weighted cost —
    # the traffic factor affects time/priority, not literal fuel consumption.
    fuel = total_dist * 0.05

    return {
        "path_leg1": leg1["path"],
        "path_leg2": leg2["path"],
        "cost": total_cost,
        "distance": total_dist,
        "fuel_cost": fuel,
        "nodes_explored": leg1["nodes_explored"] + leg2["nodes_explored"],
        "time_ms": leg1["time_ms"] + leg2["time_ms"],
        "handles_negative": leg1["handles_negative"],
        # Carry through algorithm-specific fields (e.g. iterations for BF,
        # has_negative_cycle) without hard-coding their names here.
        "extra": {k: v for k, v in leg1.items() if k not in
                  ("path", "cost", "distance", "nodes_explored", "time_ms", "handles_negative")},
    }
