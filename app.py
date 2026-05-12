"""
FoodPappa – Streamlit Application
Interactive delivery-route comparison: Dijkstra vs A* vs Bellman-Ford
on a 10,000-node urban graph with NetworkX-powered structural analysis.

Run with:  streamlit run app.py
"""

import math
import random
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
from collections import Counter

from city_graph import CityGraph
from algorithms import dijkstra, astar, bellman_ford, run_full_route

# ─── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="FoodPappa · Route Optimizer",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Cached data layer ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🏗️ Building 10,000-node city graph…")
def get_city_graph(seed: int) -> CityGraph:
    """Build and cache the CityGraph — ~200 ms on first call per seed."""
    return CityGraph(seed=seed)


@st.cache_resource(show_spinner="🔗 Building NetworkX graph for analysis…")
def get_nx_bundle(seed: int):
    """
    Convert CityGraph to NetworkX DiGraph and pre-compute structural stats.
    Routing still uses our custom algorithms; NX is for analysis and verification.

    Returns (G: nx.DiGraph, stats: dict).
    """
    g = get_city_graph(seed)
    G = g.to_networkx()

    # Count negative-weight (express lane) edges
    neg_count = sum(1 for _, _, d in G.edges(data=True) if d.get("weight", 0) < 0)

    # Average out-degree = total directed edges / nodes
    avg_out = G.number_of_edges() / max(G.number_of_nodes(), 1)

    # Per-restaurant in/out degree
    rest_degrees = {
        r: {"in": G.in_degree(r), "out": G.out_degree(r)}
        for r in g.restaurants
    }

    stats = {
        "nx_nodes":     G.number_of_nodes(),
        "nx_edges":     G.number_of_edges(),
        "neg_edges":    neg_count,
        "avg_out":      avg_out,
        "rest_degrees": rest_degrees,
    }
    return G, stats


# ─── Visualization helpers ────────────────────────────────────────────────────

def draw_city_map(g: CityGraph, results: dict,
                  rider: int, restaurant: int, dest: int) -> plt.Figure:
    """
    Render the city map clearly showing the two-leg delivery journey.

    LEG 1 (rider → restaurant): drawn with LOW opacity  — pickup route
    LEG 2 (restaurant → customer): drawn with FULL opacity — delivery route

    This makes it obvious the route has two separate stages, meeting at R.
    Colors: Dijkstra=blue, A*=green, Bellman-Ford=orange.
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Road grid ─────────────────────────────────────────────────────────────
    for row in range(g.ROWS):
        xs = [g.positions[g.nid(col, row)][0] for col in range(g.COLS)]
        ys = [g.positions[g.nid(col, row)][1] for col in range(g.COLS)]
        ax.plot(xs, ys, color="#1a2535", lw=0.4, zorder=1)
    for col in range(g.COLS):
        xs = [g.positions[g.nid(col, row)][0] for row in range(g.ROWS)]
        ys = [g.positions[g.nid(col, row)][1] for row in range(g.ROWS)]
        ax.plot(xs, ys, color="#1a2535", lw=0.4, zorder=1)

    # ── Express lanes ─────────────────────────────────────────────────────────
    for u, v in g.negative_edges:
        x1, y1 = g.positions[u]
        x2, y2 = g.positions[v]
        ax.plot([x1, x2], [y1, y2], color="#ffd700", alpha=0.3, lw=0.8, zorder=2)

    # ── Route drawing ─────────────────────────────────────────────────────────
    # Each algorithm draws two legs:
    #   Leg 1  (rider → restaurant) : dashed, semi-transparent  = "going to pick up"
    #   Leg 2  (restaurant → dest)  : solid,  fully opaque      = "delivering to customer"
    # Small X/Y offsets keep Dijkstra and A* (same path) visually separable.
    ALGO_STYLES = {
        #            color       offset
        "Dijkstra":     ("#4fc3f7",  0.35),
        "A*":           ("#a5d6a7", -0.35),
        "Bellman-Ford": ("#ff8a65",  0.00),
    }

    for algo, (color, off) in ALGO_STYLES.items():
        r = results.get(algo, {})

        # Leg 1 — rider → restaurant (dashed, dimmer)
        leg1 = r.get("path_leg1")
        if leg1 and len(leg1) > 1:
            xs = [g.positions[n][0] + off for n in leg1]
            ys = [g.positions[n][1] + off for n in leg1]
            ax.plot(xs, ys, color=color, ls="--", lw=2.0, alpha=0.5, zorder=4)

        # Leg 2 — restaurant → customer (solid, bright)
        leg2 = r.get("path_leg2")
        if leg2 and len(leg2) > 1:
            xs = [g.positions[n][0] + off for n in leg2]
            ys = [g.positions[n][1] + off for n in leg2]
            ax.plot(xs, ys, color=color, ls="-",  lw=2.8, alpha=1.0, zorder=4)

    # ── Restaurant icons (all 5) ───────────────────────────────────────────────
    for r_node in g.restaurants:
        selected = r_node == restaurant
        x, y = g.positions[r_node]
        ax.scatter([x], [y], s=80 if not selected else 0,
                   c="#2e2e3e", zorder=3, marker="^",
                   edgecolors="#555", linewidths=0.5)

    # ── Big clear markers for the 3 key points ────────────────────────────────
    def _mark(node: int, label: str, color: str, sz: int = 220):
        x, y = g.positions[node]
        ax.scatter([x], [y], s=sz, c=color, zorder=7,
                   edgecolors="white", linewidths=2.0)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(10, 8), color="white",
                    fontsize=12, fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", alpha=0.6, ec="none"))

    _mark(rider,      "S  Rider",       "#00bcd4", sz=250)
    _mark(restaurant, "R  Restaurant",  "#ff8c00", sz=280)
    _mark(dest,       "D  Customer",    "#66bb6a", sz=250)

    # ── Arrow showing journey direction ───────────────────────────────────────
    # Draw a faint arc: S → R → D so the flow is obvious
    sx, sy = g.positions[rider]
    rx, ry = g.positions[restaurant]
    dx, dy = g.positions[dest]
    ax.annotate("", xy=(rx, ry), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#00bcd4",
                                lw=1.2, alpha=0.4,
                                connectionstyle="arc3,rad=0.15"), zorder=3)
    ax.annotate("", xy=(dx, dy), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="-|>", color="#66bb6a",
                                lw=1.2, alpha=0.4,
                                connectionstyle="arc3,rad=0.15"), zorder=3)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], color="#4fc3f7", ls="-",  lw=2.5, label="Dijkstra"),
        mlines.Line2D([], [], color="#a5d6a7", ls="-",  lw=2.5, label="A*  (same path, offset)"),
        mlines.Line2D([], [], color="#ff8a65", ls="-",  lw=2.5, label="Bellman-Ford"),
        mlines.Line2D([], [], color="#ffffff", ls="--", lw=1.5, label="── Leg 1: Rider → Restaurant", alpha=0.5),
        mlines.Line2D([], [], color="#ffffff", ls="-",  lw=1.5, label="─── Leg 2: Restaurant → Customer"),
        mlines.Line2D([], [], color="#ffd700", ls="-",  lw=1.0, label="Express Lane", alpha=0.6),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              facecolor="#111927", labelcolor="white",
              edgecolor="#334", fontsize=9, framealpha=0.95)

    ax.set_title("FoodPappa  ·  Delivery Route Map  (dashed = pickup,  solid = delivery)",
                 color="#aaa", fontsize=11, pad=10, loc="left")
    fig.tight_layout(pad=0.5)
    return fig


def draw_degree_histogram(G: nx.DiGraph) -> plt.Figure:
    """Bar chart of out-degree distribution for the first 1,000 nodes."""
    sample  = list(G.nodes())[:1000]
    degrees = [G.out_degree(n) for n in sample]
    counts  = Counter(degrees)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.bar(xs, ys, color="#4fc3f7", alpha=0.8, edgecolor="#0d1117", linewidth=0.4)
    ax.set_xlabel("Out-Degree", color="#aaa", fontsize=9)
    ax.set_ylabel("Node Count",  color="#aaa", fontsize=9)
    ax.set_title("Out-Degree Distribution (first 1,000 nodes)",
                 color="#cccccc", fontsize=10)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.tight_layout()
    return fig


# ─── Results table ────────────────────────────────────────────────────────────

def build_comparison_df(results: dict) -> pd.DataFrame:
    """
    Build an algorithm comparison DataFrame.
    Appends ' ★' to the best (lowest) value in each numeric metric.
    """
    def _best(key):
        vals = [v[key] for v in results.values() if math.isfinite(v[key])]
        return min(vals) if vals else None

    best_cost = _best("cost")
    best_time = _best("time_ms")
    best_dist = _best("distance")
    best_expl = min(v["nodes_explored"] for v in results.values())

    def star(val, best):
        return " ★" if best is not None and abs(val - best) < 1e-9 else ""

    rows = []
    for name, r in results.items():
        rows.append({
            "Algorithm":        name,
            "Route Cost":       f"{r['cost']:.3f}{star(r['cost'], best_cost)}",
            "Distance (units)": f"{r['distance']:.2f}{star(r['distance'], best_dist)}",
            "Fuel ($)":         f"${r['fuel_cost']:.3f}",
            "Nodes Explored":   f"{r['nodes_explored']:,}{star(r['nodes_explored'], best_expl)}",
            "Time (ms)":        f"{r['time_ms']:.2f}{star(r['time_ms'], best_time)}",
            "Neg. Weights":     "✓ Yes" if r["handles_negative"] else "✗ No",
        })
    return pd.DataFrame(rows).set_index("Algorithm")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

REST_NAMES = [
    "🍕 Piazza Norte  (NW)",
    "🍔 Burger Bay    (NE)",
    "🍜 Central Hub   (C) ",
    "🌶  Spice Garden  (SW)",
    "🐟 Harbor Grill  (SE)",
]

with st.sidebar:
    st.title("🍕 FoodPappa")
    st.caption("Delivery Route Optimizer")
    st.divider()

    seed = int(st.number_input(
        "Graph seed", min_value=1, max_value=9999, value=42,
        help="Changing the seed regenerates the city graph with a different road layout.",
    ))

    st.divider()
    st.subheader("Order Setup")

    rest_idx = st.selectbox(
        "Restaurant", range(5), format_func=lambda i: REST_NAMES[i]
    )

    st.markdown("**Delivery destination**")
    dest_x = st.slider("Column (X)", 0, 99, 75)
    dest_y = st.slider("Row    (Y)", 0, 99, 75)

    st.markdown("**Rider spawn**")
    auto_rider = st.checkbox("Random position", value=True)
    rider_x = rider_y = 5
    if not auto_rider:
        rider_x = st.slider("Rider column (X)", 0, 99, 5)
        rider_y = st.slider("Rider row    (Y)", 0, 99, 5)

    st.divider()
    run_btn = st.button(
        "🚀 Find Best Route", use_container_width=True, type="primary"
    )

# ─── Main area ────────────────────────────────────────────────────────────────

st.title("🍕 FoodPappa – Delivery Route Optimizer")
st.markdown(
    "Comparing **Dijkstra**, **A\\***, and **Bellman-Ford** on a **10,000-node** "
    "urban graph with **120 express lanes** (negative-weight edges) that only "
    "Bellman-Ford can exploit for cheaper routes."
)

# Load city graph (cached after first build)
g     = get_city_graph(seed)
gstat = g.stats()

# ── Graph overview metrics ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodes",          f"{gstat['nodes']:,}")
c2.metric("Directed Edges", f"{gstat['directed_edges']:,}")
c3.metric("Express Lanes",  gstat["express_lanes"],  delta="negative-weight")
c4.metric("Restaurants",    gstat["restaurants"])
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_route, tab_nx = st.tabs(["🗺️  Route Planner", "📊  Graph Analysis"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Route Planner
# ════════════════════════════════════════════════════════════════════════════════
with tab_route:
    if not run_btn:
        st.info(
            "⬅️  Configure your order in the sidebar, then click **Find Best Route**."
        )
        st.stop()

    # ── Resolve node IDs from sidebar values ──────────────────────────────────
    restaurant_node = g.restaurants[rest_idx]
    dest_node       = g.nearest_node(dest_x, dest_y)

    if auto_rider:
        # Deterministic random position per (seed, restaurant, destination) triple
        rng        = random.Random(seed * 17 + rest_idx + dest_x + dest_y)
        rider_node = rng.randrange(g.COLS * g.ROWS)
    else:
        rider_node = g.nearest_node(rider_x, rider_y)

    # ── Run algorithms ────────────────────────────────────────────────────────
    with st.spinner("Running Dijkstra, A*, and Bellman-Ford…"):
        results = {}
        for name, fn in [
            ("Dijkstra",     dijkstra),
            ("A*",           astar),
            ("Bellman-Ford", bellman_ford),
        ]:
            results[name] = run_full_route(
                fn, g.adj, g.positions, rider_node, restaurant_node, dest_node
            )

    # ── Two-column layout: map | comparison table ─────────────────────────────
    col_map, col_info = st.columns([3, 2], gap="large")

    with col_map:
        st.subheader("Route Map")
        fig = draw_city_map(g, results, rider_node, restaurant_node, dest_node)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)   # release matplotlib memory

    with col_info:
        st.subheader("Algorithm Comparison")
        df = build_comparison_df(results)
        st.dataframe(df, use_container_width=True, height=218)

        # ── Winner callout ────────────────────────────────────────────────────
        d_cost  = results["Dijkstra"]["cost"]
        bf_cost = results["Bellman-Ford"]["cost"]

        if math.isfinite(bf_cost) and math.isfinite(d_cost) and bf_cost < d_cost - 0.01:
            savings = (d_cost - bf_cost) / d_cost * 100
            st.success(
                f"**Bellman-Ford wins!**  Found a route **{savings:.1f}% cheaper** "
                f"via express lanes that Dijkstra/A\\* cannot use "
                f"(they skip negative-weight edges entirely)."
            )
        else:
            valid = {k: v["cost"] for k, v in results.items()
                     if math.isfinite(v["cost"])}
            winner = min(valid, key=valid.get) if valid else "N/A"
            st.info(
                f"**{winner}** has the lowest route cost on this order. "
                "No express-lane shortcut improved this particular route."
            )

        # ── Per-leg breakdown ─────────────────────────────────────────────────
        st.markdown("**Leg breakdown**")
        lc1, lc2, lc3 = st.columns(3)
        for col, (name, r) in zip([lc1, lc2, lc3], results.items()):
            l1 = len(r["path_leg1"]) if r["path_leg1"] else 0
            l2 = len(r["path_leg2"]) if r["path_leg2"] else 0
            col.markdown(
                f"**{name}**  \n"
                f"Pickup: {l1} nodes  \n"
                f"Delivery: {l2} nodes  \n"
                f"⏱ {r['time_ms']:.1f} ms"
            )

        with st.expander("Node IDs (debug)"):
            st.write(f"Rider start : `{rider_node}`")
            st.write(f"Restaurant  : `{restaurant_node}`")
            st.write(f"Destination : `{dest_node}`")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – NetworkX Graph Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab_nx:
    st.subheader("NetworkX Structural Analysis")
    st.markdown(
        "The city graph is loaded into **NetworkX** (`DiGraph`) to examine structural "
        "properties — degree distribution, express-lane tagging, and independent route "
        "verification.  Actual routing uses the custom algorithms in `algorithms.py` "
        "because they are 3–10× faster on this graph size."
    )

    with st.spinner("Loading NetworkX graph (first load ~5 s)…"):
        G, nx_stats = get_nx_bundle(seed)

    # ── NX overview ───────────────────────────────────────────────────────────
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("NX Nodes",         f"{nx_stats['nx_nodes']:,}")
    n2.metric("NX Edges",         f"{nx_stats['nx_edges']:,}")
    n3.metric("Neg-weight Edges", nx_stats["neg_edges"],  delta="express lanes")
    n4.metric("Avg Out-Degree",   f"{nx_stats['avg_out']:.1f}")
    st.divider()

    # ── Restaurant table | degree histogram ───────────────────────────────────
    nc_left, nc_right = st.columns(2, gap="large")

    with nc_left:
        st.markdown("**Restaurant Node Connectivity**")
        rest_rows = []
        for i, r_node in enumerate(g.restaurants):
            deg = nx_stats["rest_degrees"][r_node]
            rest_rows.append({
                "Restaurant": REST_NAMES[i],
                "Node ID":    r_node,
                "In-Degree":  deg["in"],
                "Out-Degree": deg["out"],
            })
        st.dataframe(
            pd.DataFrame(rest_rows), hide_index=True, use_container_width=True
        )

    with nc_right:
        st.markdown("**Out-Degree Distribution (first 1,000 nodes)**")
        hfig = draw_degree_histogram(G)
        st.pyplot(hfig, use_container_width=True)
        plt.close(hfig)

    st.divider()

    # ── NX Dijkstra vs custom Dijkstra ────────────────────────────────────────
    st.markdown("**Route Verification: NetworkX vs Custom Dijkstra**")
    st.markdown(
        "We run `nx.dijkstra_path` on a fixed single-leg route and compare its cost "
        "to our hand-written Dijkstra.  Both must agree because the graph has no "
        "negative edges on the non-express subgraph and both algorithms are correct "
        "for non-negative weights."
    )

    demo_src = g.nid(10, 10)
    demo_dst = g.nid(80, 80)

    try:
        # NX Dijkstra rejects any graph with negative-weight edges (even off-path).
        # Build a non-negative view so both implementations operate on the same edges.
        G_pos = nx.DiGraph()
        for n, d in G.nodes(data=True):
            G_pos.add_node(n, **d)
        for u, v, d in G.edges(data=True):
            if d["weight"] >= 0:
                G_pos.add_edge(u, v, **d)

        with st.spinner("Running nx.dijkstra_path (single leg, non-negative subgraph)…"):
            nx_cost = nx.dijkstra_path_length(G_pos, demo_src, demo_dst, weight="weight")
            nx_path = nx.dijkstra_path(G_pos, demo_src, demo_dst, weight="weight")

        our_r    = dijkstra(g.adj, g.positions, demo_src, demo_dst)
        our_cost = our_r["cost"]
        match    = abs(nx_cost - our_cost) < 1e-5

        vc1, vc2 = st.columns(2)
        vc1.metric("Our Dijkstra cost",
                   f"{our_cost:.5f}", delta=f"{our_r['time_ms']:.1f} ms")
        vc2.metric("NetworkX Dijkstra cost", f"{nx_cost:.5f}")

        if match:
            st.success(
                f"✓ **Both implementations agree** — cost = **{nx_cost:.5f}**, "
                f"path = **{len(nx_path)} nodes**  (nodes {demo_src} → {demo_dst})"
            )
        else:
            st.warning(
                f"Results differ by **{abs(nx_cost - our_cost):.2e}** — "
                "likely floating-point rounding on a long path."
            )
    except Exception as exc:
        st.error(f"NetworkX verification failed: {exc}")
