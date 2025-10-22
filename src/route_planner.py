import time
import math
from typing import List, Tuple, Dict, Optional

import osmnx as ox
import networkx as nx
import folium
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow


class CPPRoutePlanner:
    def __init__(self,
                 place_name: str,
                 custom_filter: Optional[str] = None,
                 undirected_penalty: float = 1e6,
                 address: Optional[str] = None
                 ):
        self.place_name = place_name
        self.custom_filter = custom_filter
        self.undirected_penalty = undirected_penalty
        self.user_address = address

        # graph attributes
        self.original_graph: Optional[nx.MultiDiGraph] = None
        self.directed_graph: Optional[nx.MultiDiGraph] = None
        self.undirected_graph: Optional[nx.Graph] = None

        # results / working graphs
        self.work_graph: Optional[nx.MultiDiGraph] = None
        self.center_node: Optional[int] = None
        self.unbalanced_nodes: List[Tuple[int, int]] = []
        self.positives: List[Tuple[int, int]] = []
        self.negatives: List[Tuple[int, int]] = []
        self.shortest_pairs: Dict[Tuple[int,int], Tuple[float,bool]] = {}

    @staticmethod
    def solve_min_cost_flow_with_ortools(digraph: nx.DiGraph, scale: int = 1000, default_capacity: int = 10 ** 9):
        """
        Replace networkx.network_simplex with OR-Tools SimpleMinCostFlow.
        Inputs:
          - digraph: directed graph with edge attribute 'weight' and optional 'capacity'
                     node attributes: 'demand' (networkx convention) or 'supply'.
        Returns:
          (total_cost_float, flow_dict) where flow_dict[u][v] = flow_amount (int).
        """
        #t0 = time.perf_counter()

        # map nodes to consecutive indices
        node_to_idx: Dict = {}
        idx_to_node: List = []
        for idx, node in enumerate(digraph.nodes()):
            node_to_idx[node] = idx
            idx_to_node.append(node)

        smcf = SimpleMinCostFlow()

        # add arcs (handle MultiDiGraph by iterating with keys)
        arcs = []  # list of tuples (arc_index, u_node, v_node)
        if digraph.is_multigraph():
            for u, v, key, data in digraph.edges(data=True):
                tail = node_to_idx[u]
                head = node_to_idx[v]
                capacity = int(data.get('capacity', default_capacity))
                w = data.get('weight', 0.0)
                if w is None or (isinstance(w, float) and (math.isnan(w) or math.isinf(w))):
                    w = 0.0
                cost = int(round(float(w) * scale))
                arc_idx = smcf.add_arc_with_capacity_and_unit_cost(tail, head, capacity, cost)
                arcs.append((arc_idx, u, v))
        else:
            for u, v, data in digraph.edges(data=True):
                tail = node_to_idx[u]
                head = node_to_idx[v]
                capacity = int(data.get('capacity', default_capacity))
                w = data.get('weight', 0.0)
                if w is None or (isinstance(w, float) and (math.isnan(w) or math.isinf(w))):
                    w = 0.0
                cost = int(round(float(w) * scale))
                arc_idx = smcf.add_arc_with_capacity_and_unit_cost(tail, head, capacity, cost)
                arcs.append((arc_idx, u, v))

        # set node supplies (OR-Tools: positive = supply, negative = demand).
        # NetworkX 'demand' convention: positive means demand; we map supply = -demand.
        for node, data in digraph.nodes(data=True):
            demand_attr = data.get('demand', None)
            supply_attr = data.get('supply', None)
            if supply_attr is not None:
                supply = int(supply_attr)
            elif demand_attr is not None:
                supply = -int(demand_attr)
            else:
                supply = 0
            smcf.set_node_supply(node_to_idx[node], int(supply))

        status = smcf.solve()
        if status != smcf.OPTIMAL:
            raise RuntimeError(f"OR-Tools SimpleMinCostFlow failed, status={status}")

        total_cost = smcf.optimal_cost() / scale

        # build flow dict {u: {v: flow}}
        flow_dict: Dict = {node: {} for node in digraph.nodes()}
        for arc_idx, u_node, v_node in arcs:
            f = smcf.flow(arc_idx)
            if f != 0:
                flow_dict[u_node].setdefault(v_node, 0)
                flow_dict[u_node][v_node] += int(f)

        #solve_time = time.perf_counter() - t0
        return total_cost, flow_dict

    def load_and_trim_graph(self):
        """Load graph from OSM and keep the largest weakly connected component."""
        print("[1] Loading directed graph from OSM...")
        graph = ox.graph_from_place(self.place_name, custom_filter=self.custom_filter, truncate_by_edge=True)
        print(f"    Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        wccs = list(nx.weakly_connected_components(graph))
        largest = max(wccs, key=len)
        graph = graph.subgraph(largest).copy()
        self.original_graph = graph.copy()
        self.directed_graph = graph
        self.undirected_graph = graph.to_undirected()
        print(f"    Trimmed to largest weakly connected component: Nodes={graph.number_of_nodes()}, Edges={graph.number_of_edges()}")

    def pick_center_node(self):
        """Pick a center node based on place name heuristics (keeps original logic)."""
        parts = self.place_name.split()
        #key = parts[0] if parts else ""
        # heuristics from original script: specific addresses for known districts
        try:
            if self.user_address is not None:
                address = self.user_address
                lat, lon = ox.geocode(address)[0], ox.geocode(address)[1]
                self.center_node = ox.distance.nearest_nodes(self.directed_graph, lon, lat)
            else:
                nodes = list(self.directed_graph.nodes(data=True))
                self.center_node = nodes[len(nodes) // 2][0]
        except Exception:
            # fallback to the graph centroid node
            nodes = list(self.directed_graph.nodes(data=True))
            self.center_node = nodes[len(nodes)//2][0]
        print(f"[3] Chosen center node: {self.center_node}")

    def analyze_balance(self):
        """Compute node balance (out-degree - in-degree) and prepare positive/negative lists."""
        print("[4] Analyzing node balances...")
        balance = {}
        unbalanced = []
        for n in self.directed_graph.nodes():
            b = self.directed_graph.out_degree(n) - self.directed_graph.in_degree(n)
            balance[n] = b
            if b != 0:
                unbalanced.append((n, b))
        self.unbalanced_nodes = unbalanced
        self.positives = [(n, b) for n, b in unbalanced if b > 0]   # needs incoming edges
        self.negatives = [(n, -b) for n, b in unbalanced if b < 0]  # needs outgoing edges (positive counts)

        print(f"    Unbalanced nodes: {len(self.unbalanced_nodes)}")
        pos_sum = sum(b for _, b in self.positives) if self.positives else 0
        neg_sum = sum(b for _, b in self.negatives) if self.negatives else 0
        print(f"    Sum positive demand: {pos_sum}, Sum negative demand: {neg_sum}")

    def build_pairwise_costs(self):
        """Compute shortest-path distances (directed; fallback to undirected with penalty) between negatives and positives."""
        print("[6] Building pairwise shortest-path costs between negatives and positives...")
        shortest_dist = {}
        dijkstra_calls = 0
        total_dijkstra_time = 0.0

        for source_node, need in self.negatives:
            t0 = time.perf_counter()
            dist_from_source = nx.single_source_dijkstra_path_length(self.directed_graph, source_node, weight='length')
            t1 = time.perf_counter()
            dijkstra_calls += 1
            total_dijkstra_time += (t1 - t0)

            for target_node, need_target in self.positives:
                if target_node in dist_from_source:
                    shortest_dist[(source_node, target_node)] = (dist_from_source[target_node], False)
                else:
                    try:
                        d_undir = nx.shortest_path_length(self.undirected_graph, source_node, target_node, weight='length')
                        shortest_dist[(source_node, target_node)] = (d_undir + self.undirected_penalty, True)
                    except nx.NetworkXNoPath:
                        shortest_dist[(source_node, target_node)] = (float('inf'), True)

        self.shortest_pairs = shortest_dist
        print(f"    Dijkstra calls: {dijkstra_calls}, total_dijkstra_time={total_dijkstra_time:.3f}s")
        pairs_without_directed = sum(1 for (u, v), (d, was_undir) in shortest_dist.items() if was_undir and math.isfinite(d) and d > self.undirected_penalty)
        print(f"    Pairs without directed path: {pairs_without_directed}")

    def build_auxiliary_flow_graph(self) -> nx.DiGraph:
        """
        Build auxiliary directed graph H used for min-cost flow:
        nodes = positives + negatives, node attribute 'demand' = balance,
        edges u->v with weight = shortest distance and large capacity.
        """
        aux = nx.DiGraph()
        if not self.positives and not self.negatives:
            return aux
        # add only nodes that participate
        for n, b in self.positives + self.negatives:
            aux.add_node(n)
        # set demand attribute: networkx convention: positive = demand
        balance = {n: self.directed_graph.out_degree(n) - self.directed_graph.in_degree(n) for n in aux.nodes()}
        for n in aux.nodes():
            aux.nodes[n]['demand'] = balance.get(n, 0)
        # add edges for feasible pairs
        for (u, v), (cost, was_undir) in self.shortest_pairs.items():
            if math.isfinite(cost):
                aux.add_edge(u, v, weight=float(cost), capacity=10**6)
        return aux

    def solve_flow_and_apply(self):
        """Solve min-cost flow on auxiliary graph and apply selected paths by adding artificial edges to work_graph."""
        aux = self.build_auxiliary_flow_graph()
        if aux.number_of_edges() == 0 or aux.number_of_nodes() == 0:
            print("[WARN] Auxiliary graph has no edges or nodes; skipping flow.")
            return None, None

        print("[DIAG] Aux flow graph nodes:", aux.number_of_nodes(), "edges:", aux.number_of_edges())
        try:
            t0 = time.perf_counter()
            total_cost, flow_dict = self.solve_min_cost_flow_with_ortools(aux)
            solve_time = time.perf_counter() - t0
            print(f"[TIMING] min-cost flow finished in {solve_time:.4f}s, cost={total_cost}")
        except Exception as exc:
            print("[ERROR] min-cost flow failed:", exc)
            return None, None

        # reconstruct paths for used pairs and add artificial edges to a working copy of the graph
        G_work = self.directed_graph.copy()
        pairs_used = []
        predicted_added_length = 0.0
        path_reconstruction_calls = 0
        total_path_time = 0.0
        added_edges = 0
        added_violations = 0

        for src in flow_dict or {}:
            for tgt, cnt in flow_dict[src].items():
                if cnt <= 0:
                    continue
                pairs_used.append((src, tgt, cnt))
                pair_cost, was_undir = self.shortest_pairs.get((src, tgt), (float('inf'), True))
                if math.isfinite(pair_cost):
                    predicted_added_length += pair_cost * cnt

                # find path (prefer directed)
                tpr0 = time.perf_counter()
                try:
                    path_nodes = nx.shortest_path(self.directed_graph, src, tgt, weight='length')
                    used_undir = False
                except nx.NetworkXNoPath:
                    path_nodes = nx.shortest_path(self.undirected_graph, src, tgt, weight='length')
                    used_undir = True
                tpr1 = time.perf_counter()
                path_reconstruction_calls += 1
                total_path_time += (tpr1 - tpr0)

                # add artificial edges for each unit of flow
                for _rep in range(int(cnt)):
                    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                        edge_data = G_work.get_edge_data(a, b)
                        violates = False
                        attrs = {}
                        if edge_data:
                            attrs = list(edge_data.values())[0].copy()
                        else:
                            rev = G_work.get_edge_data(b, a)
                            if rev:
                                attrs = list(rev.values())[0].copy()
                                violates = True
                            else:
                                attrs = {}
                                violates = True
                        attrs['artificial'] = True
                        if violates:
                            attrs['violates_direction'] = True
                        # add edge or update
                        G_work.add_edge(a, b, **attrs)
                        added_edges += 1
                        if violates:
                            added_violations += 1

        self.work_graph = G_work
        print("[RESULT] pairs chosen:", pairs_used)
        print(f"[RESULT] predicted added length (meters): {predicted_added_length:.1f}")
        print(f"    Path recon calls: {path_reconstruction_calls}, total_path_time={total_path_time:.3f}s")
        print(f"    Added edges: {added_edges}, added_violations: {added_violations}")

        return total_cost, flow_dict

    def ensure_eulerian_and_get_circuit(self) -> List[Tuple]:
        """Check if work_graph is Eulerian and produce Eulerian circuit (list of edges)."""
        if self.work_graph is None:
            raise RuntimeError("work_graph is not prepared; run solve_flow_and_apply first.")
        imbalanced_after = [(n, self.work_graph.out_degree(n) - self.work_graph.in_degree(n))
                            for n in self.work_graph.nodes() if (self.work_graph.out_degree(n) - self.work_graph.in_degree(n)) != 0]
        print("[8] Unbalanced nodes after additions:", len(imbalanced_after))
        if imbalanced_after:
            for n, b in imbalanced_after:
                print('   ', n, b)
        is_eulerian = nx.is_eulerian(self.work_graph)
        print("[9] Is graph Eulerian after correction?", is_eulerian)
        try:
            t0 = time.perf_counter()
            circuit = list(nx.eulerian_circuit(self.work_graph, source=self.center_node))
            t1 = time.perf_counter()
            print(f"    Eulerian circuit length (steps): {len(circuit)}, time={(t1-t0):.3f}s")
        except Exception as exc:
            print("[ERROR] eulerian_circuit failed:", exc)
            circuit = []
        return circuit

    def build_full_route_coords(self, circuit: List[Tuple]) -> List[Tuple[float,float]]:
        """From Eulerian circuit, reconstruct lat/lon route coords using original_graph geometry."""
        if self.original_graph is None:
            raise RuntimeError("original_graph not set.")
        full_route_coords: List[Tuple[float,float]] = []
        missed = 0
        t0 = time.perf_counter()
        for item in circuit:
            if len(item) == 3:
                u, v, key = item
                edge_data = self.original_graph.get_edge_data(u, v, key=key) if self.original_graph.has_edge(u, v) else self.original_graph.get_edge_data(v, u)
            else:
                u, v = item
                edge_data = self.original_graph.get_edge_data(u, v) or self.original_graph.get_edge_data(v, u)

            if not edge_data:
                edge_data = (self.work_graph.get_edge_data(u, v) if self.work_graph is not None else None)
                if not edge_data:
                    missed += 1
                    continue
            data = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data
            if 'geometry' in data:
                coords = list(data['geometry'].coords)
            else:
                coords = [(self.original_graph.nodes[u]['x'], self.original_graph.nodes[u]['y']),
                          (self.original_graph.nodes[v]['x'], self.original_graph.nodes[v]['y'])]
            # convert to (lat, lon)
            coords_latlon = [(lat, lon) for lon, lat in coords]
            if full_route_coords and full_route_coords[-1] == coords_latlon[0]:
                full_route_coords.extend(coords_latlon[1:])
            else:
                full_route_coords.extend(coords_latlon)
        t1 = time.perf_counter()
        print(f"    Geometry reconstruction missed edges: {missed}, time={(t1-t0):.3f}s")
        return full_route_coords

    def save_route_map(self, full_route_coords: List[Tuple[float,float]], output_name: Optional[str] = None):
        """Draw route and artificial edges on a folium map and save to HTML."""
        if not full_route_coords:
            print("[WARN] No route coords to draw.")
            return
        if output_name is None:
            prefix = self.place_name.split(',')[0].replace(' ', '_')
            output_name = f"Map_{prefix}.html"
        center_latlon = (self.original_graph.nodes[self.center_node]['y'], self.original_graph.nodes[self.center_node]['x'])
        m = folium.Map(location=center_latlon, zoom_start=12)
        folium.Marker(location=center_latlon,
                      popup=f"Center node: {self.center_node}",
                      tooltip="Administrative center",
                      icon=folium.Icon(color='red', icon='flag', prefix='fa')).add_to(m)
        folium.PolyLine(full_route_coords, weight=3).add_to(m)

        # draw artificial edges
        if self.work_graph is not None:
            for u, v, data in self.work_graph.edges(data=True):
                if data.get('artificial'):
                    orig = (self.original_graph.get_edge_data(u, v) or self.original_graph.get_edge_data(v, u))
                    if orig:
                        d = list(orig.values())[0]
                        if 'geometry' in d:
                            coords = [(lat, lon) for lon, lat in list(d['geometry'].coords)]
                            folium.PolyLine(coords, color="red", weight=4, opacity=0.6).add_to(m)

        t0 = time.perf_counter()
        m.save(output_name)
        t1 = time.perf_counter()
        print(f"    Saved route map to {output_name}, time={(t1-t0):.3f}s")