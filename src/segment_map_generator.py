import os
import time
import json
from typing import List, Tuple, Dict, Optional

import osmnx as ox
import networkx as nx
import folium
from folium.plugins import AntPath

from src.geomutils import GeoUtils


class SegmentMapGenerator:
    def __init__(self, base_graph: nx.MultiDiGraph, center_latlon: Tuple[float,float]):
        self.graph = base_graph
        self.center_latlon = center_latlon
        self._route_cache: Dict[Tuple[int,int], List[Tuple[float,float]]] = {}

    def route_geometry_on_graph(self, a: Tuple[float,float], b: Tuple[float,float], weight='length') -> List[Tuple[float,float]]:
        """Return list of (lat, lon) coordinates for path between two points (a,b). Cache results by node pair."""
        na = ox.nearest_nodes(self.graph, a[1], a[0])
        nb = ox.nearest_nodes(self.graph, b[1], b[0])
        key = (na, nb)
        if key in self._route_cache:
            return self._route_cache[key]

        try:
            route_nodes = nx.shortest_path(self.graph, na, nb, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._route_cache[key] = [a, b]
            return [a, b]

        coords: List[Tuple[float,float]] = []
        last = None
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data is None:
                pt = (self.graph.nodes[u]['y'], self.graph.nodes[u]['x'])
                if pt != last:
                    coords.append(pt); last = pt
                continue
            edge = list(edge_data.values())[0]
            if 'geometry' in edge:
                for x, y in edge['geometry'].coords:
                    pt = (y, x)
                    if pt != last:
                        coords.append(pt); last = pt
            else:
                pt = (self.graph.nodes[u]['y'], self.graph.nodes[u]['x'])
                if pt != last:
                    coords.append(pt); last = pt
        # add final node
        end_pt = (self.graph.nodes[route_nodes[-1]]['y'], self.graph.nodes[route_nodes[-1]]['x'])
        if not coords or coords[-1] != end_pt:
            coords.append(end_pt)
        self._route_cache[key] = coords
        return coords

    def split_route_by_distance(self, route_coords: List[Tuple[float,float]], max_km: float = 200.0) -> List[List[Tuple[float,float]]]:
        """Split full route into large segments each no longer than max_km (kilometers)."""
        if not route_coords:
            return []
        if not self.center_latlon:
            lats = [c[0] for c in route_coords]; lons = [c[1] for c in route_coords]
            self.center_latlon = (sum(lats)/len(lats), sum(lons)/len(lons))
        segments: List[List[Tuple[float,float]]] = []
        # start with realistic path from center to first point
        current = self.route_geometry_on_graph(self.center_latlon, route_coords[0])[:]
        current_len = 0.0
        for i in range(1, len(route_coords)):
            a = route_coords[i-1]; b = route_coords[i]
            d_km = GeoUtils.haversine_m(a, b) / 1000.0
            if current_len + d_km + (GeoUtils.haversine_m(b, self.center_latlon) / 1000.0) <= max_km or len(current) == 1:
                current.append(b); current_len += d_km
            else:
                # close segment by adding path back to center
                try:
                    back = self.route_geometry_on_graph(current[-1], self.center_latlon)
                    if back:
                        if back[0] == current[-1]:
                            current.extend(back[1:])
                        else:
                            current.extend(back)
                except Exception:
                    current.append(self.center_latlon)
                segments.append(current)
                start_to_a = self.route_geometry_on_graph(self.center_latlon, a)
                current = start_to_a[:]
                if current[-1] != a:
                    current.append(a)
                current.append(b)
                current_len = (GeoUtils.haversine_m(self.center_latlon, a) + GeoUtils.haversine_m(a, b)) / 1000.0
        if current:
            segments.append(current)
        return segments

    @staticmethod
    def create_segment_map(full_route: List[Tuple[float,float]],
                           segment_coords: List[Tuple[float,float]],
                           seg_index: int,
                           total: int,
                           out_path: str,
                           step_m: int = 500,
                           densify_m: Optional[int] = None):
        """Create single HTML page for a segment with step navigation and GPX export."""
        if densify_m is None:
            densify_m = min(50, max(10, int(step_m/2)))
        dense = GeoUtils.densify_coords(segment_coords, max_segment_m=densify_m)
        # split into navigation steps of ~step_m meters
        steps: List[List[Tuple[float,float]]] = []
        if dense:
            cur = [dense[0]]
            cur_len = 0.0
            for i in range(1, len(dense)):
                a = dense[i-1]; b = dense[i]
                d = GeoUtils.haversine_m(a, b)
                if cur_len + d <= step_m or len(cur) == 1:
                    cur.append(b); cur_len += d
                else:
                    steps.append(cur)
                    cur = [dense[i-1], dense[i]]; cur_len = d
            if cur:
                steps.append(cur)
        if not steps and dense:
            steps = [dense]

        print(f"[Segment {seg_index}/{total}] original_pts={len(segment_coords)}, dense_pts={len(dense)}, steps={len(steps)}, step_m={step_m}m")

        # draw folium map
        lats = [c[0] for c in full_route]; lons = [c[1] for c in full_route]
        center = (sum(lats)/len(lats), sum(lons)/len(lons))
        m = folium.Map(location=center, zoom_start=12, control_scale=True)
        for idx, seg in enumerate(steps):
            folium.PolyLine(locations=seg, weight=5, opacity=0.6, tooltip=f"Segment {idx+1}").add_to(m)
        folium.PolyLine(locations=[(lat, lon) for lat, lon in full_route], weight=3, opacity=0.35, color="#888888").add_to(m)
        if len(dense) >= 2:
            AntPath(locations=[dense[0], dense[-1]], delay=1, weight=1, color="#000", pulse_color="#000", opacity=1).add_to(m)
        folium.CircleMarker(location=segment_coords[0], radius=5, color="green", fill=True).add_to(m)
        folium.CircleMarker(location=segment_coords[-1], radius=5, color="blue", fill=True).add_to(m)

        tmp_path = out_path + ".tmp.html"
        m.save(tmp_path)

        # insert interactive JS controls (prev/next/autoplay) and save final html + gpx
        steps_json = json.dumps(steps)
        js = f"""
        <script>
        var steps = {steps_json};
        (function(){{
            // Найдём объект карты (folium даёт map_<id> или иногда map)
            var mapVar = (typeof map !== 'undefined') ? map : null;
            if(!mapVar){{
                for(var k in window){{ if(k.indexOf('map_')===0){{ mapVar = window[k]; break; }} }}
            }}

            function initOnce(){{
                if(!mapVar){{
                    console.error('Map object not found (mapVar == null).');
                    return;
                }}
                var currentStep = 0;
                var ant = null;
                var info = document.getElementById('seginfo');
                var prevbtn = document.getElementById('prevbtn');
                var nextbtn = document.getElementById('nextbtn');
                var autoplaybtn = document.getElementById('autoplaybtn');

                function showStep(i){{
                    if(i<0 || i>=steps.length) return;
                    currentStep = i;
                    if(ant){{
                        try{{ mapVar.removeLayer(ant); }} catch(e){{ console.warn(e); }}
                        ant = null;
                    }}
                    // Создаём antPath для шага и добавляем на карту
                    ant = L.polyline.antPath(steps[i], {{
                        delay: 800, dashArray: [10,20], weight: 6, color: '#ff3333', pulseColor: '#ffffff'
                    }});
                    ant.addTo(mapVar);
                    if(info) info.innerHTML = 'Шаг ' + (currentStep+1) + ' / ' + steps.length;
                }}

                if(prevbtn) prevbtn.onclick = function(){{ if(currentStep>0) showStep(currentStep-1); }};
                if(nextbtn) nextbtn.onclick = function(){{ if(currentStep<steps.length-1) showStep(currentStep+1); }};
                document.addEventListener('keydown', function(e){{
                    if(e.key==='ArrowLeft'){{ if(currentStep>0) showStep(currentStep-1); }}
                    else if(e.key==='ArrowRight'){{ if(currentStep<steps.length-1) showStep(currentStep+1); }}
                }});

                var autoplay=false, autoplayInterval=null;
                if(autoplaybtn) autoplaybtn.onclick = function(){{
                    autoplay = !autoplay;
                    var el = autoplaybtn;
                    if(autoplay){{
                        el.innerText = 'Autoplay: ON';
                        autoplayInterval = setInterval(function(){{
                            if(currentStep<steps.length-1) showStep(currentStep+1);
                            else {{ clearInterval(autoplayInterval); autoplay=false; el.innerText='Autoplay: OFF'; }}
                        }}, 2500);
                    }} else {{ el.innerText = 'Autoplay: OFF'; if(autoplayInterval) clearInterval(autoplayInterval); }}
                }};

                showStep(0);
                console.log('[Segment page] init done. steps=', steps.length);
            }}

                if(mapVar){{
                    if(document.readyState === 'complete') initOnce(); else window.addEventListener('load', initOnce);
                }} else {{
                    window.addEventListener('load', function(){{
                        // при загрузке пробуем ещё раз найти map_... переменную
                        if(typeof map !== 'undefined') mapVar = map;
                        else {{
                            for(var k in window){{ if(k.indexOf('map_')===0){{ mapVar = window[k]; break; }} }}
                        }}
                        initOnce();
                    }});
                }}
            }})();
            </script>
            """
        nav_html = f"""
            <div style="position: fixed; bottom: 14px; left: 50%; transform: translateX(-50%); z-index:99999;
                        display:flex; gap:12px; align-items:center; background: rgba(255,255,255,0.92); padding:6px 10px; border-radius:10px;">
                <button id="prevbtn" style="font-size:18px;padding:8px 14px;border-radius:6px;">◀</button>
                <div id="seginfo" style="min-width:140px;text-align:center;font-weight:600;">Step 1 / {len(steps)}</div>
                <button id="nextbtn" style="font-size:18px;padding:8px 14px;border-radius:6px;">▶</button>
                <button id="autoplaybtn" style="font-size:12px;padding:6px 8px;border-radius:6px;">Autoplay: OFF</button>
                &nbsp; <a href="index.html">Index</a>
            </div>
        """
        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
        html = html.replace("</body>", nav_html + js + "</body>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        os.remove(tmp_path)

        # write GPX
        gpx_path = out_path.replace(".html", ".gpx")
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(gpx_path, "w", encoding="utf-8") as gf:
            gf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            gf.write('<gpx version="1.1" creator="export" xmlns="http://www.topografix.com/GPX/1/1">\n')
            gf.write(f'  <metadata><time>{now_iso}</time><name>Segment {seg_index}/{total}</name></metadata>\n')
            gf.write('  <trk><name>Segment</name><trkseg>\n')
            for lat, lon in dense:
                gf.write(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}"></trkpt>\n')
            gf.write('  </trkseg></trk>\n</gpx>\n')
        print(f"Saved {gpx_path} ({len(dense)} points)")
        print(f"Saved {out_path} (steps: {len(steps)})")

    def generate_segment_maps(self, full_route: List[Tuple[float,float]], output_dir: str = "output_maps",
                              max_km: float = 200.0, step_m: int = 500, densify_m: Optional[int] = None):
        """Main entry: split route and produce one HTML per big segment plus index."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # remove consecutive duplicate points
        cleaned: List[Tuple[float,float]] = [full_route[0]] if full_route else []
        for p in full_route[1:]:
            if p != cleaned[-1]:
                cleaned.append(p)
        big_segments = self.split_route_by_distance(cleaned, max_km=max_km)
        total = len(big_segments)
        print(f"Route total {GeoUtils.path_length_km(cleaned):.2f} km -> big segments: {total} (max_km={max_km})")
        for i, seg in enumerate(big_segments, start=1):
            out_path = os.path.join(output_dir, f"seg_{i:03d}.html")
            self.create_segment_map(cleaned, seg, i, total, out_path, step_m=step_m, densify_m=densify_m)
        # index
        with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write("<html><body><h2>Segments</h2><ul>")
            for i in range(1, total+1):
                f.write(f'<li><a href="seg_{i:03d}.html">Segment {i:03d}</a></li>')
            f.write("</ul></body></html>")
        print("Saved maps to", output_dir)