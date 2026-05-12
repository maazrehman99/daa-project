# 🍕 FoodPappa – Delivery Route Optimizer

> A Design & Analysis of Algorithms (DAA) project comparing **Dijkstra**, **A\***, and **Bellman-Ford** on a 10,000-node urban road network for food delivery routing.

---

## What It Does

FoodPappa simulates a food delivery platform. Given a rider, a restaurant, and a customer location, it finds the optimal two-leg route (rider → restaurant → customer) using three graph algorithms and compares their performance in real time.

The city graph includes **120 express lanes** (negative-weight edges) that model fast shortcuts — only Bellman-Ford can exploit these, often finding routes **20–62% cheaper** than Dijkstra or A\*.

---

## Screenshots

| Route Planner | Graph Analysis |
|---|---|
| Dark map with all 3 routes overlaid | NetworkX degree stats + NX vs custom verification |

---

## Project Structure

```
FoodPappa/
├── app.py                  ← Streamlit dashboard (main entry point)
├── city_graph.py           ← 10,000-node urban graph generator + NetworkX export
├── algorithms.py           ← Dijkstra, A*, Bellman-Ford implementations
├── main_analysis.py        ← Terminal-only comparison tool
├── test_algorithms.py      ← 60-test unittest suite (all passing)
├── TECHNICAL_REPORT.md     ← Full academic-style technical report
├── requirements.txt        ← Python dependencies
└── index.html              ← Old HTML GUI (superseded by app.py)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### 3. Run terminal analysis (no browser needed)

```bash
python3 main_analysis.py
# with options:
python3 main_analysis.py --seed 123 --trials 5
```

### 4. Run tests

```bash
python3 test_algorithms.py
```

Expected output: **60 tests, 0 failures**

---

## The Graph

| Property | Value |
|---|---|
| Nodes | 10,000 (100×100 grid) |
| Undirected road edges | ~22,751 |
| Directed adjacency entries | ~45,500 |
| Express lanes (negative-weight) | 120 |
| Restaurants | 5 |
| Node jitter | ±0.35 world-units |
| Traffic factor range | 0.8 – 2.5 |
| Express lane weight range | −7.0 to −0.6 |

**No negative cycles:** Express lanes are strictly top-to-bottom (source row < destination row), making it structurally impossible to form a negative cycle.

---

## Algorithms

### Dijkstra
- **Complexity:** O((V + E) log V)
- **Speed:** ~12 ms per route
- **Limitation:** Skips negative edges — cannot use express lanes
- **Best for:** Standard road networks with non-negative weights

### A\* (A-Star)
- **Complexity:** O((V + E) log V) worst-case, faster in practice
- **Speed:** ~12 ms per route
- **Heuristic:** `h(n) = euclidean_distance(n, goal) × 0.78` (admissible)
- **Limitation:** Same as Dijkstra — no negative edges
- **Best for:** When destination direction is known (guided search)

### Bellman-Ford
- **Complexity:** O(V × E)
- **Speed:** ~380 ms per route
- **Advantage:** Handles negative-weight edges — exploits express lanes
- **Best for:** Graphs with special shortcut edges (tolls, fast lanes, discounts)

### Head-to-head

| Metric | Dijkstra | A\* | Bellman-Ford |
|---|---|---|---|
| Time | ~12 ms ★ | ~12 ms ★ | ~380 ms |
| Handles negative edges | ✗ | ✗ | ✓ |
| Route cost (with express) | Higher | Higher | Lower ★ |
| Nodes explored | ~5,000 ★ | ~4,500 ★ | 10,000 |
| Detects negative cycles | ✗ | ✗ | ✓ |

---

## Streamlit App Features

**Tab 1 — Route Planner**
- Select restaurant (5 named options across city quadrants)
- Set delivery destination via X/Y sliders
- Toggle random or manual rider position
- Dark-mode city map showing all 3 routes simultaneously
- Comparison table with ★ marking the best value per metric
- Winner callout with savings % when Bellman-Ford beats Dijkstra

**Tab 2 — Graph Analysis (NetworkX)**
- Node/edge counts verified against NetworkX `DiGraph`
- Restaurant node in-degree / out-degree table
- Out-degree distribution histogram (first 1,000 nodes)
- Live verification: NX `dijkstra_path_length` vs our custom Dijkstra

---

## NetworkX Integration

`CityGraph.to_networkx()` exports the graph as a `nx.DiGraph` with full attributes:

```python
from city_graph import CityGraph
import networkx as nx

g = CityGraph(seed=42)
G = g.to_networkx()

# Node attributes: x, y, col, row, is_restaurant
# Edge attributes: weight (float), express (bool)

print(G.number_of_nodes())   # 10000
print(G.number_of_edges())   # ~45500

# Count express lanes
express = sum(1 for _, _, d in G.edges(data=True) if d['express'])
print(express)  # 120
```

> **Note:** Use `algorithms.py` for actual routing — it's 3–10× faster than NetworkX built-ins on this graph size. NetworkX is used here for structural analysis and independent verification.

---

## Test Suite

60 tests across 5 classes:

| Class | Tests | What's Covered |
|---|---|---|
| `TestCityGraph` | 14 | Node/edge counts, express lanes, no negative cycles, nearest_node |
| `TestDijkstra` | 12 | Path correctness, endpoints, cost, timing, edge-in-adj check |
| `TestAstar` | 6 | Path, endpoints, agrees with Dijkstra (admissible heuristic) |
| `TestBellmanFord` | 8 | Path, BF ≤ Dijkstra cost, no negative cycle, iterations field |
| `TestRunFullRoute` | 10 | Two-leg journey, leg endpoints, fuel formula, time sum |
| `TestNetworkX` | 10 | NX export, express tags, restaurant attribute, weight match, NX vs custom |

---

## Dependencies

```
streamlit>=1.57.0
networkx>=3.6.1
matplotlib>=3.10.0
plotly>=6.0.0
pandas>=2.0.0
numpy>=2.0.0
```

---

## Real-World Relevance

This project models the core routing decisions made by platforms like **Uber Eats**, **DoorDash**, and **Amazon Logistics**:

- **Dijkstra / A\*** → Used in standard GPS routing (Google Maps, Apple Maps)
- **Bellman-Ford** → Used when road pricing, tolls, or rewards create negative-effective-weight edges
- **Two-leg routing** → Exactly the pickup + delivery model used by all major delivery apps

---

## Authors

Built as a Design & Analysis of Algorithms (DAA) academic project.

**Tech stack:** Python 3.12 · Streamlit · NetworkX · Matplotlib · Pandas
