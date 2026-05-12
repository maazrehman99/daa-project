"""
FoodPappa – Comprehensive Test Suite
Tests CityGraph construction, all three routing algorithms, run_full_route,
and the NetworkX DiGraph export.

Run with:  python3 test_algorithms.py
"""

import math
import random
import unittest
import networkx as nx

from city_graph import CityGraph
from algorithms import dijkstra, astar, bellman_ford, run_full_route


# ─────────────────────────────────────────────────────────────────────────────
# CityGraph construction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCityGraph(unittest.TestCase):
    """Verify the graph is built with the correct structure."""

    @classmethod
    def setUpClass(cls):
        # Build once; all tests in this class reuse the same graph object
        cls.g = CityGraph(seed=42)

    def test_node_count(self):
        self.assertEqual(len(self.g.positions), 10_000)

    def test_adj_covers_all_nodes(self):
        # Every node must have an adjacency list (even if empty)
        self.assertEqual(len(self.g.adj), 10_000)

    def test_express_lane_count(self):
        # Construction loop targets exactly NUM_EXPRESS = 120
        self.assertEqual(self.g.express_count, 120)

    def test_negative_edges_set_size(self):
        # negative_edges set must match express_count
        self.assertEqual(len(self.g.negative_edges), 120)

    def test_restaurant_count(self):
        self.assertEqual(len(self.g.restaurants), 5)

    def test_no_negative_cycles_top_to_bottom(self):
        # Express lanes are guaranteed to go from a lower row to a higher row,
        # so there can be no directed cycle with negative total weight.
        for u, v in self.g.negative_edges:
            _, row_u = self.g.col_row(u)
            _, row_v = self.g.col_row(v)
            self.assertGreater(
                row_v, row_u,
                msg=f"Express lane {u}→{v} violates top→bottom constraint "
                    f"(row {row_u} → row {row_v})"
            )

    def test_all_express_edges_present_in_adj(self):
        # Each (u, v) in negative_edges must appear in adj[u]
        for u, v in self.g.negative_edges:
            adj_targets = [nb for nb, _ in self.g.adj[u]]
            self.assertIn(v, adj_targets,
                          msg=f"Express edge {u}→{v} missing from adj[{u}]")

    def test_all_express_edges_are_negative(self):
        # Every edge in negative_edges must have weight < 0 in adj
        for u, v in self.g.negative_edges:
            w = next(w for nb, w in self.g.adj[u] if nb == v)
            self.assertLess(w, 0,
                            msg=f"Express edge {u}→{v} has non-negative weight {w}")

    def test_nearest_node_centre(self):
        node = self.g.nearest_node(50.0, 50.0)
        self.assertIn(node, self.g.positions)
        x, y = self.g.positions[node]
        # Centre node should be within 2 world-units of (50, 50)
        self.assertAlmostEqual(x, 50.0, delta=2.0)
        self.assertAlmostEqual(y, 50.0, delta=2.0)

    def test_nearest_node_deterministic(self):
        # Same input must always return the same node
        self.assertEqual(
            self.g.nearest_node(25.5, 75.5),
            self.g.nearest_node(25.5, 75.5),
        )

    def test_stats_returns_required_keys(self):
        keys = self.g.stats().keys()
        for k in ("nodes", "directed_edges", "undirected_edges",
                  "express_lanes", "restaurants"):
            self.assertIn(k, keys)

    def test_stats_node_count(self):
        self.assertEqual(self.g.stats()["nodes"], 10_000)

    def test_stats_express_count(self):
        self.assertEqual(self.g.stats()["express_lanes"], 120)

    def test_total_undirected_edges_reasonable(self):
        # Minimum: 100×99 (H) + 99×100 (V) = 19,800 base edges
        self.assertGreater(self.g.stats()["undirected_edges"], 19_000)


# ─────────────────────────────────────────────────────────────────────────────
# Dijkstra tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDijkstra(unittest.TestCase):
    """Test Dijkstra correctness, output structure, and timing."""

    @classmethod
    def setUpClass(cls):
        g = CityGraph(seed=42)
        cls.adj = g.adj
        cls.pos = g.positions
        cls.src = g.nid(10, 10)
        cls.dst = g.nid(90, 90)
        cls.g   = g

    def _run(self, src=None, dst=None):
        return dijkstra(self.adj, self.pos,
                        src if src is not None else self.src,
                        dst if dst is not None else self.dst)

    def test_finds_path(self):
        r = self._run()
        self.assertIsNotNone(r["path"])
        self.assertGreater(len(r["path"]), 1)

    def test_path_starts_at_source(self):
        self.assertEqual(self._run()["path"][0], self.src)

    def test_path_ends_at_destination(self):
        self.assertEqual(self._run()["path"][-1], self.dst)

    def test_cost_is_nonnegative(self):
        self.assertGreaterEqual(self._run()["cost"], 0.0)

    def test_cost_is_finite(self):
        self.assertTrue(math.isfinite(self._run()["cost"]))

    def test_handles_negative_flag_false(self):
        self.assertFalse(self._run()["handles_negative"])

    def test_timing_under_2s(self):
        self.assertLess(self._run()["time_ms"], 2000)

    def test_src_equals_dst_gives_zero_cost(self):
        r = dijkstra(self.adj, self.pos, self.src, self.src)
        self.assertEqual(r["cost"], 0.0)

    def test_src_equals_dst_single_node_path(self):
        r = dijkstra(self.adj, self.pos, self.src, self.src)
        self.assertEqual(r["path"], [self.src])

    def test_path_hops_exist_in_adj(self):
        # Every consecutive pair in the path must be a real edge in adj
        path = self._run()["path"]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            adj_targets = {nb for nb, _ in self.adj[u]}
            self.assertIn(v, adj_targets,
                          msg=f"Path hop {u}→{v} is not a real edge")

    def test_distance_field_positive(self):
        r = self._run()
        self.assertGreater(r["distance"], 0.0)

    def test_nodes_explored_positive(self):
        self.assertGreater(self._run()["nodes_explored"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# A* tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAstar(unittest.TestCase):
    """Test A* correctness and its agreement with Dijkstra (admissible heuristic)."""

    @classmethod
    def setUpClass(cls):
        g = CityGraph(seed=42)
        cls.adj = g.adj
        cls.pos = g.positions
        cls.src = g.nid(10, 10)
        cls.dst = g.nid(90, 90)

    def test_finds_path(self):
        r = astar(self.adj, self.pos, self.src, self.dst)
        self.assertIsNotNone(r["path"])

    def test_path_endpoints(self):
        r = astar(self.adj, self.pos, self.src, self.dst)
        self.assertEqual(r["path"][0],  self.src)
        self.assertEqual(r["path"][-1], self.dst)

    def test_agrees_with_dijkstra_cost(self):
        # A* with an admissible heuristic must return the same optimal cost as Dijkstra
        # when both skip negative edges (the condition that makes both correct).
        d_cost = dijkstra(self.adj, self.pos, self.src, self.dst)["cost"]
        a_cost = astar  (self.adj, self.pos, self.src, self.dst)["cost"]
        self.assertAlmostEqual(
            d_cost, a_cost, places=5,
            msg="A* and Dijkstra must agree — heuristic should be admissible",
        )

    def test_handles_negative_flag_false(self):
        r = astar(self.adj, self.pos, self.src, self.dst)
        self.assertFalse(r["handles_negative"])

    def test_cost_finite_and_positive(self):
        r = astar(self.adj, self.pos, self.src, self.dst)
        self.assertTrue(math.isfinite(r["cost"]))
        self.assertGreater(r["cost"], 0.0)

    def test_timing_under_2s(self):
        r = astar(self.adj, self.pos, self.src, self.dst)
        self.assertLess(r["time_ms"], 2000)


# ─────────────────────────────────────────────────────────────────────────────
# Bellman-Ford tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBellmanFord(unittest.TestCase):
    """Test Bellman-Ford including negative-edge exploitation and cycle detection."""

    @classmethod
    def setUpClass(cls):
        g = CityGraph(seed=42)
        cls.adj = g.adj
        cls.pos = g.positions
        # Choose src in top half so express lanes (top→bottom) may be exploited
        cls.src = g.nid(5, 5)
        cls.dst = g.nid(90, 90)
        cls.g   = g

    def test_finds_path(self):
        r = bellman_ford(self.adj, self.pos, self.src, self.dst)
        self.assertIsNotNone(r["path"])
        self.assertGreater(len(r["path"]), 1)

    def test_path_endpoints(self):
        r = bellman_ford(self.adj, self.pos, self.src, self.dst)
        self.assertEqual(r["path"][0],  self.src)
        self.assertEqual(r["path"][-1], self.dst)

    def test_no_negative_cycle_detected(self):
        # Top→bottom express lane design guarantees no negative cycles exist
        r = bellman_ford(self.adj, self.pos, self.src, self.dst)
        self.assertFalse(r["has_negative_cycle"])

    def test_cost_le_dijkstra(self):
        # BF can use express lanes, so its cost must be ≤ Dijkstra's cost
        d_cost = dijkstra(self.adj, self.pos,     self.src, self.dst)["cost"]
        b_cost = bellman_ford(self.adj, self.pos, self.src, self.dst)["cost"]
        if math.isfinite(d_cost) and math.isfinite(b_cost):
            self.assertLessEqual(
                b_cost, d_cost + 1e-6,
                msg="Bellman-Ford must find cost ≤ Dijkstra (it handles negative edges)",
            )

    def test_handles_negative_flag_true(self):
        r = bellman_ford(self.adj, self.pos, self.src, self.dst)
        self.assertTrue(r["handles_negative"])

    def test_iterations_field_present_and_positive(self):
        r = bellman_ford(self.adj, self.pos, self.src, self.dst)
        self.assertIn("iterations", r)
        self.assertGreater(r["iterations"], 0)

    def test_nodes_explored_equals_graph_size(self):
        # BF explores every node in every pass — no selective expansion
        r = bellman_ford(self.adj, self.pos, self.src, self.dst)
        self.assertEqual(r["nodes_explored"], len(self.adj))

    def test_cost_finite(self):
        r = bellman_ford(self.adj, self.pos, self.src, self.dst)
        self.assertTrue(math.isfinite(r["cost"]))


# ─────────────────────────────────────────────────────────────────────────────
# run_full_route tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunFullRoute(unittest.TestCase):
    """Test the two-leg wrapper that models rider → restaurant → customer."""

    @classmethod
    def setUpClass(cls):
        g = CityGraph(seed=42)
        cls.adj   = g.adj
        cls.pos   = g.positions
        cls.rider = g.nid(5, 5)
        cls.rest  = g.restaurants[0]
        cls.dest  = g.nid(80, 80)

    def _run(self, fn):
        return run_full_route(fn, self.adj, self.pos,
                              self.rider, self.rest, self.dest)

    def test_dijkstra_both_legs_not_none(self):
        r = self._run(dijkstra)
        self.assertIsNotNone(r["path_leg1"])
        self.assertIsNotNone(r["path_leg2"])

    def test_leg1_starts_at_rider(self):
        r = self._run(dijkstra)
        self.assertEqual(r["path_leg1"][0], self.rider)

    def test_leg1_ends_at_restaurant(self):
        r = self._run(dijkstra)
        self.assertEqual(r["path_leg1"][-1], self.rest)

    def test_leg2_starts_at_restaurant(self):
        r = self._run(dijkstra)
        self.assertEqual(r["path_leg2"][0], self.rest)

    def test_leg2_ends_at_destination(self):
        r = self._run(dijkstra)
        self.assertEqual(r["path_leg2"][-1], self.dest)

    def test_fuel_cost_formula(self):
        r = self._run(dijkstra)
        # fuel_cost must equal total physical distance × 0.05
        self.assertAlmostEqual(r["fuel_cost"], r["distance"] * 0.05, places=9)

    def test_bf_both_legs_not_none(self):
        r = self._run(bellman_ford)
        self.assertIsNotNone(r["path_leg1"])
        self.assertIsNotNone(r["path_leg2"])

    def test_bf_cost_le_dijkstra_full_route(self):
        d_r = self._run(dijkstra)
        b_r = self._run(bellman_ford)
        if math.isfinite(d_r["cost"]) and math.isfinite(b_r["cost"]):
            self.assertLessEqual(b_r["cost"], d_r["cost"] + 1e-6)

    def test_total_time_is_sum_of_legs(self):
        # run_full_route sums leg1 + leg2 times; verify by running each leg separately
        r_full = self._run(dijkstra)
        leg1   = dijkstra(self.adj, self.pos, self.rider, self.rest)
        leg2   = dijkstra(self.adj, self.pos, self.rest,  self.dest)
        expected_ms = leg1["time_ms"] + leg2["time_ms"]
        # Allow ±5 ms tolerance for OS scheduling variance
        self.assertAlmostEqual(r_full["time_ms"], expected_ms, delta=5.0)

    def test_handles_negative_matches_algorithm(self):
        # For Dijkstra: handles_negative should be False
        self.assertFalse(self._run(dijkstra)["handles_negative"])
        # For Bellman-Ford: handles_negative should be True
        self.assertTrue(self._run(bellman_ford)["handles_negative"])


# ─────────────────────────────────────────────────────────────────────────────
# NetworkX export tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNetworkX(unittest.TestCase):
    """Test the to_networkx() export and consistency with the Python graph."""

    @classmethod
    def setUpClass(cls):
        cls.g = CityGraph(seed=42)
        cls.G = cls.g.to_networkx()  # full graph (includes negative express lanes)

        # Non-negative subgraph for NX Dijkstra verification — NX Dijkstra refuses to
        # run on any graph that contains negative weights (even off-path edges).
        # Our custom Dijkstra skips negative edges, so both should agree on G_pos.
        cls.G_pos = nx.DiGraph()
        for n, d in cls.G.nodes(data=True):
            cls.G_pos.add_node(n, **d)
        for u, v, d in cls.G.edges(data=True):
            if d["weight"] >= 0:
                cls.G_pos.add_edge(u, v, **d)

    def test_is_directed_graph(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_node_count_matches(self):
        self.assertEqual(self.G.number_of_nodes(), 10_000)

    def test_express_edges_tagged(self):
        # Edges with express=True must equal exactly 120 (NUM_EXPRESS)
        count = sum(1 for _, _, d in self.G.edges(data=True) if d.get("express"))
        self.assertEqual(count, 120)

    def test_restaurant_nodes_attribute_true(self):
        for r_node in self.g.restaurants:
            self.assertTrue(
                self.G.nodes[r_node]["is_restaurant"],
                msg=f"Node {r_node} should have is_restaurant=True",
            )

    def test_non_restaurant_attribute_false(self):
        non_rest = next(n for n in self.G.nodes() if n not in self.g.restaurants)
        self.assertFalse(self.G.nodes[non_rest]["is_restaurant"])

    def test_node_has_position_attributes(self):
        # Every node should carry x, y, col, row
        n = list(self.G.nodes())[0]
        for attr in ("x", "y", "col", "row"):
            self.assertIn(attr, self.G.nodes[n],
                          msg=f"Node {n} is missing attribute '{attr}'")

    def test_edge_weights_match_adj(self):
        # Spot-check 20 edges: NetworkX weight must match adj weight exactly
        rng     = random.Random(0)
        checked = 0
        for u in rng.choices(list(self.g.adj.keys()), k=40):
            for v, w in self.g.adj[u]:
                if self.G.has_edge(u, v):
                    nx_w = self.G[u][v]["weight"]
                    self.assertAlmostEqual(
                        nx_w, w, places=9,
                        msg=f"Weight mismatch on ({u},{v}): adj={w}, NX={nx_w}",
                    )
                    checked += 1
                    if checked >= 20:
                        return
        self.assertGreater(checked, 0, "No edges were spot-checked")

    def test_nx_dijkstra_agrees_with_custom(self):
        # Our custom Dijkstra skips negative edges; NX Dijkstra is run on G_pos
        # (non-negative subgraph) so both operate on the same edge set.
        # They must return identical optimal costs.
        src = self.g.nid(10, 10)
        dst = self.g.nid(30, 30)
        our_cost = dijkstra(self.g.adj, self.g.positions, src, dst)["cost"]
        nx_cost  = nx.dijkstra_path_length(self.G_pos, src, dst, weight="weight")
        self.assertAlmostEqual(our_cost, nx_cost, places=4,
            msg="Custom Dijkstra and nx.dijkstra_path_length must agree on non-negative subgraph")

    def test_negative_weight_edges_exist(self):
        neg = [w for _, _, d in self.G.edges(data=True)
               if (w := d.get("weight", 0)) < 0]
        self.assertEqual(len(neg), 120,
            msg="Exactly 120 negative-weight edges expected in NX graph")

    def test_express_edges_are_negative(self):
        for u, v, d in self.G.edges(data=True):
            if d.get("express"):
                self.assertLess(d["weight"], 0,
                    msg=f"Express edge ({u},{v}) should have negative weight")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
