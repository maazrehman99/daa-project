"""
FoodPappa – City Graph Generator
Produces a realistic urban road network with 10,000+ nodes.

Design rationale:
  A 100×100 grid is the simplest structure that yields a meaningfully large
  graph (10 000 nodes, ~22 000+ undirected edges) while still being small
  enough to run Bellman-Ford in a reasonable demo time.  Random positional
  jitter breaks the perfectly-uniform lattice so edge weights are not all
  identical and shortest-path decisions become interesting.
"""

import math
import random
from typing import Dict, List, Tuple, Set


class CityGraph:
    """
    100×100 grid graph (10,000 nodes) simulating an urban road network.

    Nodes:  intersections with slight random positional jitter.
    Edges:  4-connected grid + ~30% diagonal connections.
            Each edge weight = physical_distance × traffic_factor.
    Express lanes: ~120 directed edges with negative weights
                   (one-way fast routes spanning 22+ rows top→bottom).
                   These require Bellman-Ford for correct shortest-path computation.
    """

    # ── class-level constants ────────────────────────────────────────────────
    COLS = 100                # city width in grid columns
    ROWS = 100                # city height in grid rows  → 10 000 nodes total
    JITTER = 0.35             # max positional noise per axis (keeps nodes visually distinct)
    TRAFFIC_MIN = 0.8         # lightest possible traffic multiplier
    TRAFFIC_MAX = 2.5         # heaviest possible traffic multiplier (rush hour)
    NUM_EXPRESS = 120         # number of express-lane directed edges to create
    EXPRESS_ROW_SPAN = 22     # minimum row gap for an express lane (ensures large distance)
    NEG_W_MIN = -7.0          # most-negative express-lane weight (biggest time saving)
    NEG_W_MAX = -0.6          # least-negative express-lane weight (marginal saving)
    FUEL_PER_UNIT = 0.05      # $ per unit of physical distance (used by run_full_route)

    def __init__(self, seed: int = 42):
        # A seeded RNG makes every run deterministic — important for reproducible demos
        self.rng = random.Random(seed)

        # Core data structures
        self.positions: Dict[int, Tuple[float, float]] = {}   # node_id → (x, y) with jitter
        self.adj: Dict[int, List[Tuple[int, float]]] = {}     # node_id → [(neighbour, weight)]
        self.negative_edges: Set[Tuple[int, int]] = set()     # set of (u, v) express-lane pairs
        self.restaurants: List[int] = []                      # node IDs of the 5 restaurants
        self.total_edges = 0    # count of *undirected* grid edges (excludes express return arcs)
        self.express_count = 0  # how many express lanes were successfully placed

        self._build()

    # ── helpers ──────────────────────────────────────────────────────────────

    def nid(self, col: int, row: int) -> int:
        """Row-major encoding: unique integer ID for a (col, row) grid cell."""
        return row * self.COLS + col

    def col_row(self, nid: int) -> Tuple[int, int]:
        """Inverse of nid() — recover (col, row) from a node ID."""
        return nid % self.COLS, nid // self.COLS

    def _rr(self, a: float, b: float) -> float:
        """Uniform random float in [a, b)."""
        return a + self.rng.random() * (b - a)

    def _ri(self, a: int, b: int) -> int:
        """Uniform random integer in [a, b-1] (exclusive upper bound for slice-like usage)."""
        return self.rng.randint(a, b - 1)

    def _euclidean(self, u: int, v: int) -> float:
        """Straight-line (Euclidean) distance between two nodes using their jittered positions."""
        x1, y1 = self.positions[u]
        x2, y2 = self.positions[v]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ── construction ─────────────────────────────────────────────────────────

    def _build(self):
        """
        Build the full city graph in three passes:
          1. Create nodes with jittered positions.
          2. Add bidirectional grid edges (horizontal, vertical, ~30% diagonal).
          3. Overlay express lanes (directed, negative-weight, top→bottom only).
          4. Pin restaurant locations.
        """
        N = self.COLS * self.ROWS  # noqa: F841 — kept for readability

        # ── Pass 1: node positions ───────────────────────────────────────────
        # Jitter breaks lattice symmetry so edge weights are heterogeneous,
        # making the graph a more realistic routing benchmark.
        for row in range(self.ROWS):
            for col in range(self.COLS):
                nid = self.nid(col, row)
                self.positions[nid] = (
                    col + self._rr(-self.JITTER, self.JITTER),
                    row + self._rr(-self.JITTER, self.JITTER),
                )
                self.adj[nid] = []   # initialise adjacency list (empty for now)

        # ── Pass 2: bidirectional road edges ────────────────────────────────
        def add(u, v, w):
            # Both directions get the same weight — this models two-way streets.
            # Express lanes are added separately and are *not* two-way.
            self.adj[u].append((v, w))
            self.adj[v].append((u, w))
            self.total_edges += 1   # count undirected edges for reporting

        for row in range(self.ROWS):
            for col in range(self.COLS):
                u = self.nid(col, row)

                # Horizontal edge: connect to the node one column to the right
                if col + 1 < self.COLS:
                    v = self.nid(col + 1, row)
                    add(u, v, self._euclidean(u, v) * self._rr(self.TRAFFIC_MIN, self.TRAFFIC_MAX))

                # Vertical edge: connect to the node one row below
                if row + 1 < self.ROWS:
                    v = self.nid(col, row + 1)
                    add(u, v, self._euclidean(u, v) * self._rr(self.TRAFFIC_MIN, self.TRAFFIC_MAX))

                # Diagonal edge (SE direction): added only ~30% of the time to simulate
                # cut-through alleyways rather than full diagonal road coverage.
                # Uses a tighter traffic cap (2.0) because diagonals are typically side streets.
                if col + 1 < self.COLS and row + 1 < self.ROWS and self.rng.random() < 0.3:
                    v = self.nid(col + 1, row + 1)
                    add(u, v, self._euclidean(u, v) * self._rr(self.TRAFFIC_MIN, 2.0))

        # ── Pass 3: express lanes ────────────────────────────────────────────
        # Express lanes model elevated highways / metro shortcuts that physically
        # skip large portions of the city.  Their negative weights mean that
        # *using* the lane reduces accumulated route cost — analogous to a fast
        # bypass that more than compensates for any detour to reach it.
        #
        # Key safety constraint: every express lane goes strictly top→bottom
        # (src_row < dst_row).  This makes it topologically impossible to form
        # a negative cycle because any return path must travel upward through
        # normal (positive-weight) roads, and the total forward + backward
        # positive cost always exceeds |neg_weight|.
        attempts = 0
        while self.express_count < self.NUM_EXPRESS and attempts < 3000:
            attempts += 1  # guard against infinite loop if graph is very crowded

            # Pick a source row that leaves room for the required vertical span
            sr = self._ri(0, self.ROWS - self.EXPRESS_ROW_SPAN)
            # Destination row must be at least EXPRESS_ROW_SPAN rows below source
            dr = self._ri(sr + self.EXPRESS_ROW_SPAN, self.ROWS)
            sc = self._ri(0, self.COLS)
            dc = self._ri(0, self.COLS)

            u = self.nid(sc, sr)
            v = self.nid(dc, dr)
            if u == v:
                continue  # degenerate case — skip self-loops

            # Forward (express) arc: negative weight rewards using this shortcut
            neg_w = self._rr(self.NEG_W_MIN, self.NEG_W_MAX)
            self.adj[u].append((v, neg_w))
            self.negative_edges.add((u, v))   # record so algorithms can identify it

            # Reverse arc: very expensive, modelling the long way back on surface roads.
            # Weight = 4 × physical distance + 2 × |neg_w| ensures no incentive to
            # travel the express lane in reverse.
            self.adj[v].append((u, self._euclidean(u, v) * 4 + abs(neg_w) * 2))

            self.express_count += 1

        # ── Pass 4: restaurant placement ─────────────────────────────────────
        # Five prototype (col, row) positions spread across the four quadrants
        # plus the city centre.  A small random jitter is applied so restaurants
        # don't sit on exactly the same grid intersection each run, mimicking
        # real-world variability in vendor locations.
        rest_positions = [(18, 18), (81, 17), (50, 50), (20, 80), (80, 80)]
        for col, row in rest_positions:
            # Clamp to [2, 97] to keep restaurants away from the very border
            jc = max(2, min(97, col + self._ri(-4, 5)))
            jr = max(2, min(97, row + self._ri(-4, 5)))
            self.restaurants.append(self.nid(jc, jr))

    # ── queries ──────────────────────────────────────────────────────────────

    def nearest_node(self, wx: float, wy: float) -> int:
        """
        Fast nearest-node lookup using grid rounding + local 7×7 search.

        A global brute-force search over all 10 000 nodes would be O(N).
        Rounding to the nearest integer column/row and checking a 7×7 window
        gives O(1) lookups while still being correct for jitter ≤ 0.35.
        """
        col = round(wx)
        row = round(wy)
        best, best_d = -1, math.inf

        # Search a ±3 neighbourhood to account for position jitter
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nc, nr = col + dc, row + dr
                if not (0 <= nc < self.COLS and 0 <= nr < self.ROWS):
                    continue   # skip out-of-bounds cells
                nid = self.nid(nc, nr)
                dx = self.positions[nid][0] - wx
                dy = self.positions[nid][1] - wy
                d = dx * dx + dy * dy   # squared distance — avoids sqrt cost in comparison
                if d < best_d:
                    best_d = d
                    best = nid
        return best

    def nearest_restaurant(self, wx: float, wy: float) -> Tuple[int, float]:
        """
        Linear scan over the 5 restaurants — small enough that no index is needed.
        Returns the (node_id, euclidean_distance) of the closest restaurant.
        """
        best, best_d = self.restaurants[0], math.inf
        for rid in self.restaurants:
            dx = self.positions[rid][0] - wx
            dy = self.positions[rid][1] - wy
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_d:
                best_d = d
                best = rid
        return best, best_d

    def stats(self) -> dict:
        """Summary statistics for UI display and sanity-checking."""
        N = self.COLS * self.ROWS
        # Directed edge count includes both directions of each road + express reverse arcs
        E = sum(len(v) for v in self.adj.values())
        return {
            "nodes": N,
            "directed_edges": E,
            "undirected_edges": self.total_edges,
            "express_lanes": self.express_count,
            "restaurants": len(self.restaurants),
        }

    # ── NetworkX export ───────────────────────────────────────────────────────

    def to_networkx(self):
        """
        Build and return a NetworkX DiGraph from the city graph.

        This is used for structural analysis (degree distribution, connectivity
        checks, independent path verification) rather than for actual routing —
        the custom Dijkstra/A*/Bellman-Ford implementations in algorithms.py
        are faster and provide richer metadata.

        Node attributes:
            x, y        — jittered position
            col, row    — integer grid coordinates
            is_restaurant — True if the node is one of the 5 restaurant nodes

        Edge attributes:
            weight  — float edge cost (can be negative for express lanes)
            express — True if (u, v) is in self.negative_edges

        Returns:
            nx.DiGraph
        """
        import networkx as nx  # imported here so city_graph.py has no hard dep on networkx

        G = nx.DiGraph()

        # ── Add nodes ─────────────────────────────────────────────────────────
        restaurant_set = set(self.restaurants)  # O(1) membership test
        for nid, (x, y) in self.positions.items():
            col, row = self.col_row(nid)
            G.add_node(
                nid,
                x=x,
                y=y,
                col=col,
                row=row,
                is_restaurant=(nid in restaurant_set),
            )

        # ── Add edges ─────────────────────────────────────────────────────────
        for u, neighbours in self.adj.items():
            for v, w in neighbours:
                G.add_edge(
                    u, v,
                    weight=float(w),
                    express=((u, v) in self.negative_edges),
                )

        return G
