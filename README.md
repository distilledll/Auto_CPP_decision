# CPP Route Planner & Segment Map Generator

The project turns the OpenStreetMap road network into a reproducible Euler route: it analyzes the balance of vertices, solves min-cost flow, and adds minimal artificial edges.
The result is an interactive HTML map and a set of GPX/HTML segments for turn-by-turn navigation and export.


---

> [README_RUS.md](README_RUS.md)

---

# Contents

1. [Brief description](#brief-description)
2. [Project files](#project-files)
3. [Requirements and installation](#requirements-and-installation)
4. [Quick start](#quick-start)
5. [Options and configuration](#options-and-configuration)
6. [Output files and folder structure](#output-files-and-folder-structure)
7. [License and contacts](#license-and-contacts)

---

# Brief description

The project converts the OpenStreetMap road network into a reproducible Euler route and generates convenient interactive maps + GPX for parts of the route. The main goal is to obtain a route that passes through all edges of the network (or a selected subpart) with minimal intervention (adding paths) and can be easily exported for inspections/patrols.

### Technical part (what the system does)

1. **Graph loading** — downloads a directed road graph for the specified region via osmnx and leaves the largest weakly connected component.
2. **Vertex balance analysis** — calculates the difference between out_degree and in_degree for each vertex, identifies nodes with excess and deficit (i.e., where inputs/outputs need to be added), and prepares pairs for correction.
3. **Calculation of pair costs** — for each pair (excess → deficit) calculates the shortest distance: first in a directed graph (Dijkstra), if there is no path — in an undirected graph with an additional penalty.
4. **Solving the min-cost flow problem** — forms an auxiliary directed graph with demands and capacities and solves the minimum-cost flow using OR-Tools; the result is which pairs and in what volume can be “connected” at minimum cost.
5. **Adding artificial edges** — based on the found flows, adds a sequence of edges (paths) to the working copy of the graph — marked as artificial, sometimes with the violates_direction flag (if a reverse/undirected traversal was used).
6. **Checking and obtaining an Eulerian cycle** — after additions, attempts to obtain an Eulerian circuit (via NetworkX) and extracts the order of the edges.
7. **Restoring geometry** — for each edge, extracts the geometry (if any) or uses the node coordinates to form a complete track polyline.
8. **Map and GPX generation** — saves the overall map (folium) and breaks the route into “large” segments; for each segment, creates an interactive HTML page with frame-by-frame navigation and a GPX file.

### Architecture and key components

- **GeoUtils** — utilities: haversine, densify (insertion of intermediate points), length calculation.
- **RoutePlanner** — main class: graph loading/processing, auxiliary graph construction, min-cost flow solution, working graph copy construction, Euler cycle retrieval, and complete geometry output.
- **SegmentMapGenerator** — responsible for breaking down the complete route into segments, “densifying” points, creating interactive HTML pages and GPX.
- **OR-Tools wrapper** — adaptation of input data NetworkX → SimpleMinCostFlow (cost scale, vertex indexing).
- **Visualization** — folium + AntPath and embedded JS for step-by-step navigation.

### Important design decisions / behavior

1. **Directed priority + undirected fallback:** the project prefers directed paths, but if there are no paths, it uses undirected ones with a large penalty so as not to leave pairs unsolved. This preserves the meaning of directed roads but makes the algorithm stable.
2. **Artificial edges as metadata:** added paths are marked, remain in the working graph, and are visually highlighted — this allows you to distinguish between real and added changes.
3. **Route caching:** routes between nodes are cached for reuse (saving Dijkstra/OSM nearest calls).
4. **Scalability:** Dijkstra for multiple sources is resource-intensive; recommendations are provided for narrowing the area, filtering by road type, and possible parallelization.

---

# Project files

* `main_script.py` — main file (can be considered as an example of use).
* `src/geoutils.py` — contains the `GeoUtils` class.
* `src/routeplanner.py` — contains the `CPPRoutePlanner` class.
* `src/segment_map_generator.py` — contains the `SegmentMapGenerator` class.
* `requirements.txt` — list of dependencies.
* `Maps_<name>/` — folder with interactive HTML and GPX (generated at startup).
* `Map_<name>.html` — final map of the complete route (generated at startup).

---

# Requirements and installation

Recommended Python version: **3.10+**.

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (`requirements.txt`):

```bash
pip install -r requirements.txt
```

---

# Quick start

1. Prepare the environment and dependencies (see previous section).
2. Change the settings in the `main_script.py` file if necessary:
   * `district_name` — name
* `custom_filter` — OSM filter for selecting road types.
   * Parameters for `SegmentMapGenerator.generate_segment_maps`: `max_km`, `step_m`, `densify_m`.
4. Run:

```bash
python3 main_script.py
```

---

# Parameters and configuration

The code uses several key settings. In short:

* `district_name` — string for `osmnx.graph_from_place(...)`. For example: `“China-city, Moscow, Russia”`.
* `custom_filter` — a string filter by OSM tags (by default, the main road types are selected).
* `undirected_penalty` (in `RoutePlanner`) — a penalty in meters added when using an undirected detour instead of a directed path (by default, it is large — to prefer directed routes).
* `max_km` (in `SegmentMapGenerator.generate_segment_maps`) — maximum length of a “large segment” in kilometers when splitting.
* `step_m` — step size (in meters) within a segment for turn-by-turn navigation (default is 400–500 m).
* `densify_m` — an option to insert intermediate points so that the distance between consecutive points is no more than the specified value (usually 20–50 m).

---

# Output files and folder structure

After successful execution, you will receive:

* `Map_<region>.html` — a single HTML map with the complete route and a mark indicating the added artificial edges.
* The `Maps_<region>/` folder with the files `seg_<n>.html` and `seg_<n>.gpx`.

---

# Further improvements (ideas)

* Logging instead of `print` for flexible configuration of log levels.
* New parameters for path segmentation.
* Optimization of the CPP construction algorithm.

---

# License and contacts

This project was implemented as a working utility due to the lack of the necessary functions and implementations of the CPP algorithm. 
Collaboration and suggestions: tg - `@distilledll`.