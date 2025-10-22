import gc

import networkx as nx

from src.route_planner import CPPRoutePlanner
from src.segment_map_generator import SegmentMapGenerator


if __name__ == "__main__":
    # Settings (edit as needed)
    district_name = "Краснинский район, Липецкая область, Россия"
    custom_filter = (
        '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|road|track"]'
        '["motor_vehicle"!~"no"]["access"!~"private"]["service"!~"private"]'
        '["surface"!~"dirt|sand|grass|ground|mud|earth|wood"]'
    )

    planner = CPPRoutePlanner(place_name=district_name, custom_filter=custom_filter)
    planner.load_and_trim_graph()
    planner.pick_center_node()
    planner.analyze_balance()

    if len(planner.unbalanced_nodes) == 0 and nx.is_strongly_connected(planner.directed_graph):
        print("[INFO] Graph is already balanced and strongly connected — no additions required.")
    else:
        planner.build_pairwise_costs()
        total_cost, flow_solution = planner.solve_flow_and_apply()
        circuit = planner.ensure_eulerian_and_get_circuit()
        full_route = planner.build_full_route_coords(circuit)
        planner.save_route_map(full_route, output_name=f"Карта_{district_name.split(',')[0].replace(' ', '_')}.html")

        # generate per-segment interactive maps and GPX
        if full_route:
            center_latlon = (planner.original_graph.nodes[planner.center_node]['y'],
                             planner.original_graph.nodes[planner.center_node]['x'])
            seg_gen = SegmentMapGenerator(planner.original_graph, center_latlon=center_latlon)
            seg_gen.generate_segment_maps(full_route,
                                          max_km=200.0,
                                          output_dir=f"Карты_{district_name.split(',')[0].replace(' ', '_')}",
                                          step_m=400,
                                          densify_m=30)

    # final cleanup
    del planner
    gc.collect()