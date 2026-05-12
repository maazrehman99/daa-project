"""
FoodPappa – Terminal Analysis
Runs all three algorithms on the 10,000-node city graph and prints a
side-by-side performance comparison table.

Usage:
    python3 main_analysis.py
    python3 main_analysis.py --seed 123 --trials 5
"""

import math
import random
import argparse
import time
from city_graph import CityGraph
from algorithms import dijkstra, astar, bellman_ford, run_full_route

# ── Terminal colours (ANSI) ───────────────────────────────────────────────────
Y  = "\033[93m"   # Dijkstra – yellow/gold
G  = "\033[92m"   # A* – green
P  = "\033[95m"   # Bellman-Ford – magenta/purple
B  = "\033[94m"   # info blue
R  = "\033[91m"   # red / warning
W  = "\033[97m"   # white
DIM= "\033[2m"
RST= "\033[0m"
BOLD="\033[1m"


def fmt_ms(v):
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.2f} ms" if v < 1 else f"{v:.1f} ms"

def fmt_f(v, dec=2):
    return "N/A" if not math.isfinite(v) else f"{v:.{dec}f}"

def fmt_n(v):
    return "N/A" if not math.isfinite(v) else f"{int(v):,}"


def print_header():
    print()
    print(f"{BOLD}{W}{'─'*72}{RST}")
    print(f"{BOLD}{W}  FoodPappa – Delivery Route Optimization · Algorithm Comparison{RST}")
    print(f"{W}{'─'*72}{RST}")
    print()


def print_graph_stats(g: CityGraph):
    s = g.stats()
    print(f"{B}City Graph Stats:{RST}")
    print(f"  Nodes          : {W}{s['nodes']:,}{RST}")
    print(f"  Undirected edges: {W}{s['undirected_edges']:,}{RST}")
    print(f"  Directed edges  : {W}{s['directed_edges']:,}{RST}")
    print(f"  Express lanes   : {G}{s['express_lanes']}{RST}  (negative-weight, directed)")
    print(f"  Restaurants     : {W}{s['restaurants']}{RST}")
    print()


def print_comparison(results: dict, rider: int, rest: int, dest: int):
    d, a, b = results["dijkstra"], results["astar"], results["bellman"]

    def pick_winner(dv, av, bv, lower=True):
        vals = [(dv, "dijkstra"), (av, "astar"), (bv, "bellman")]
        finite = [(v, n) for v, n in vals if math.isfinite(v)]
        if not finite:
            return set()
        best = min(finite, key=lambda x: x[0])[0] if lower else max(finite, key=lambda x: x[0])[0]
        return {n for v, n in finite if abs(v - best) < 1e-9}

    def mark(val_str, algo, winners):
        star = f" {G}★{RST}" if algo in winners else ""
        return val_str + star

    row_sep = f"  {'─'*16}{'─'*17}{'─'*17}{'─'*16}"
    hdr = (f"  {DIM}{'Metric':<16}{RST}"
           f"{Y}{'Dijkstra':>16}{RST}  "
           f"{G}{'A*':>14}{RST}  "
           f"{P}{'Bellman-Ford':>14}{RST}")

    print(f"{BOLD}Route:{RST} Rider {W}#{rider}{RST} → Restaurant {W}#{rest}{RST} → Customer {W}#{dest}{RST}")
    print()
    print(hdr)
    print(row_sep)

    metrics = [
        ("Time",          fmt_ms(d["time_ms"]),    fmt_ms(a["time_ms"]),    fmt_ms(b["time_ms"]),
                          pick_winner(d["time_ms"], a["time_ms"], b["time_ms"])),
        ("Distance",      fmt_f(d["distance"])+" u",fmt_f(a["distance"])+" u",fmt_f(b["distance"])+" u",
                          pick_winner(d["distance"],a["distance"],b["distance"])),
        ("Route Cost",    fmt_f(d["cost"]),         fmt_f(a["cost"]),         fmt_f(b["cost"]),
                          pick_winner(d["cost"],a["cost"],b["cost"])),
        ("Fuel ($)",      "$"+fmt_f(d["fuel_cost"]),"$"+fmt_f(a["fuel_cost"]),"$"+fmt_f(b["fuel_cost"]),
                          pick_winner(d["fuel_cost"],a["fuel_cost"],b["fuel_cost"])),
        ("Nodes Explored",fmt_n(d["nodes_explored"]),fmt_n(a["nodes_explored"]),"All 10,000",
                          pick_winner(d["nodes_explored"],a["nodes_explored"],math.inf)),
        ("Neg. Weights",  "✗ No",                  "✗ No",                  "✓ Yes",
                          {"bellman"}),
    ]

    colors = {"dijkstra": Y, "astar": G, "bellman": P}

    for label, dv, av, bv, winners in metrics:
        dc = colors["dijkstra"] if "dijkstra" in winners else ""
        ac = colors["astar"] if "astar" in winners else ""
        bc = colors["bellman"] if "bellman" in winners else ""
        print(f"  {DIM}{label:<16}{RST}"
              f"{dc}{dv:>16}{RST}  "
              f"{ac}{av:>14}{RST}  "
              f"{bc}{bv:>14}{RST}")

    print(row_sep)
    print()

    # Key insight
    if math.isfinite(b["cost"]) and math.isfinite(d["cost"]) and b["cost"] < d["cost"] - 0.01:
        savings = (d["cost"] - b["cost"]) / d["cost"] * 100
        print(f"  {G}★ Bellman-Ford found a {savings:.1f}% cheaper route via express lanes{RST}")
        print(f"    that Dijkstra/A* cannot use (negative weights skipped).")
    else:
        print(f"  {DIM}Express lanes did not improve this particular route.{RST}")
    print()


def run_trial(g: CityGraph, seed: int) -> dict:
    rng = random.Random(seed)
    N = g.COLS * g.ROWS
    rider  = rng.randrange(N)
    rest   = rng.choice(g.restaurants)
    dest   = g.nearest_node(rng.uniform(5, 95), rng.uniform(5, 95))

    results = {}
    for name, fn in [("dijkstra", dijkstra), ("astar", astar), ("bellman", bellman_ford)]:
        results[name] = run_full_route(fn, g.adj, g.positions, rider, rest, dest)

    return results, rider, rest, dest


def main():
    parser = argparse.ArgumentParser(description="FoodPappa algorithm analysis")
    parser.add_argument("--seed",   type=int, default=42,  help="Graph seed")
    parser.add_argument("--trials", type=int, default=3,   help="Number of random routes to test")
    args = parser.parse_args()

    print_header()

    print(f"{B}Generating 10,000-node city graph (seed={args.seed})…{RST}")
    t0 = time.perf_counter()
    g = CityGraph(seed=args.seed)
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"  Done in {gen_ms:.1f} ms\n")
    print_graph_stats(g)

    for trial in range(1, args.trials + 1):
        print(f"{BOLD}{'═'*72}{RST}")
        print(f"{BOLD}  Trial {trial} / {args.trials}{RST}")
        print(f"{'═'*72}{RST}")
        results, rider, rest, dest = run_trial(g, seed=args.seed * 100 + trial)
        print_comparison(results, rider, rest, dest)

    print(f"{W}{'─'*72}{RST}")
    print(f"{DIM}  Graph: {g.COLS}×{g.ROWS} grid | {g.total_edges:,} edges | {g.express_count} express lanes{RST}")
    print(f"{DIM}  Algorithms: Dijkstra O((V+E)logV) · A* O((V+E)logV) · Bellman-Ford O(V·E){RST}")
    print(f"{W}{'─'*72}{RST}")
    print()


if __name__ == "__main__":
    main()
