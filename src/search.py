import heapq
import math


def make_directions(graph, node_info, path):
    """
    Collapse a path (list of node IDs) into turn-by-turn directions.

    A new step is emitted whenever the road name changes. Consecutive
    segments on the same named road are merged into a single step.

    Returns
    -------
    steps : list[dict] with keys:
        step          – 1-based step number
        instruction   – "Start on" (first) or "Turn onto" (subsequent)
        road          – road name (or "unnamed road")
        segment_km    – distance of this step in km
        cumulative_km – total distance from start at end of this step
        nodes         – number of raw graph nodes consumed in this step
    """
    if len(path) < 2:
        return []

    steps = []
    current_road = None
    segment_len = 0.0
    segment_nodes = 0
    cumulative = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge = graph.get(u, {}).get(v) or graph.get(v, {}).get(u) or {}
        raw = edge.get("name")
        road = str(raw) if (raw and raw == raw) else "unnamed road"  # handle NaN float
        length = float(edge.get("length", 0.0))

        if road != current_road:
            if current_road is not None:
                steps.append({
                    "step": len(steps) + 1,
                    "instruction": "Start on" if not steps else "Turn onto",
                    "road": current_road,
                    "segment_km": round(segment_len / 1000, 2),
                    "cumulative_km": round(cumulative / 1000, 2),
                    "nodes": segment_nodes,
                })
            current_road = road
            segment_len = length
            segment_nodes = 1
        else:
            segment_len += length
            segment_nodes += 1

        cumulative += length

    steps.append({
        "step": len(steps) + 1,
        "instruction": "Start on" if not steps else "Turn onto",
        "road": current_road,
        "segment_km": round(segment_len / 1000, 2),
        "cumulative_km": round(cumulative / 1000, 2),
        "nodes": segment_nodes,
    })

    return steps


def astar(graph, node_info, source, targets, weight="length"):
    """
    A* shortest path from source to a set of target nodes.

    The heuristic is the Euclidean straight-line distance to the nearest
    target (admissible because road distance >= straight-line distance).
    For 'time' weight the heuristic is scaled by the maximum legal speed
    (130 km/h) so it remains admissible.

    Parameters
    ----------
    graph      : adjacency list from load_graph
    node_info  : node coordinate dict from load_graph
    source     : int – start node id
    targets    : int or list[int] – one or more destination node ids
    weight     : "length" (metres) or "time" (seconds)

    Returns
    -------
    dist         : float  – shortest cost to the first target reached
    path         : list[int] – node ids source → target
    visited_nodes: set[int]  – all nodes popped from the priority queue
    """
    if isinstance(targets, int):
        targets = [targets]
    target_set = set(targets)

    target_coords = [(node_info[t]["x"], node_info[t]["y"]) for t in target_set if t in node_info]
    max_speed_mps = 130 * 1000 / 3600  # admissible upper bound for time weight

    # Lazy cache: compute h(u) at most once per node.
    # Single-target fast path avoids the min() generator entirely.
    h_cache = {}
    if len(target_coords) == 1:
        tx0, ty0 = target_coords[0]
        if weight == "time":
            def h(u):
                if u in h_cache:
                    return h_cache[u]
                ux, uy = node_info[u]["x"], node_info[u]["y"]
                h_cache[u] = math.sqrt((ux - tx0) ** 2 + (uy - ty0) ** 2) / max_speed_mps
                return h_cache[u]
        else:
            def h(u):
                if u in h_cache:
                    return h_cache[u]
                ux, uy = node_info[u]["x"], node_info[u]["y"]
                h_cache[u] = math.sqrt((ux - tx0) ** 2 + (uy - ty0) ** 2)
                return h_cache[u]
    else:
        def h(u):
            if u in h_cache:
                return h_cache[u]
            ux, uy = node_info[u]["x"], node_info[u]["y"]
            val = min(math.sqrt((ux - x) ** 2 + (uy - y) ** 2) for x, y in target_coords)
            if weight == "time":
                val /= max_speed_mps
            h_cache[u] = val
            return val

    def edge_cost(data):
        if weight == "time":
            speed_mps = (data["maxspeed_kph"] or 50) * 1000 / 3600
            return data["length"] / speed_mps
        return data["length"]

    dist = {source: 0.0}
    prev = {}
    visited_nodes = set()
    pq = [(h(source), 0.0, source)]

    while pq:
        _, cost, u = heapq.heappop(pq)
        if u in visited_nodes:
            continue
        visited_nodes.add(u)

        if u in target_set:
            path = reconstruct_path(prev, source, u)
            return dist[u], path, visited_nodes

        for v, data in graph.get(u, {}).items():
            if v in visited_nodes:
                continue
            new_cost = cost + edge_cost(data)
            if new_cost < dist.get(v, math.inf):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost + h(v), new_cost, v))

    return math.inf, [], visited_nodes


def _reconstruct_ev_path(prev, start, end_state):
    path, s = [], end_state
    while s in prev:
        path.append(s)
        s = prev[s]
    path.append(start)
    path.reverse()
    return path


def dijkstra_ev_time(graph, charging, source, target,
                     battery_kwh, consumption_kwh_per_km=0.2,
                     initial_charge=100, min_final_charge=0,
                     default_speed_kph=80):
    """
    Shortest-TIME EV path with variable-speed charging (2.3.8).

    Cost = drive seconds + charging seconds. Charging stations have
    different speeds (max_kw), so charging time varies per stop.

    The expanded graph is NOT built upfront — states (node, charge_pct)
    are generated lazily as Dijkstra explores them.

    Charge edges advance 1 % at a time so Dijkstra automatically finds
    the optimal amount to charge at each stop (no need to charge to 100 %).

    Chargers with unknown power (NaN) are ignored.

    Parameters
    ----------
    graph                  : adjacency list from load_graph
    charging               : dict {node_id: max_kw} from load_charging
    source / target        : int – node ids
    battery_kwh            : float – battery capacity (e.g. 80 or 40)
    consumption_kwh_per_km : float – kWh/km (default 0.2)
    initial_charge         : int   – starting charge % (default 100)
    min_final_charge       : int   – minimum charge % at target (default 0)
    default_speed_kph      : float – fallback speed when edge has no maxspeed

    Returns
    -------
    total_seconds : float            – total travel+charge time (inf if unreachable)
    path_states   : list[(int, int)] – (node_id, charge_pct) sequence
    visited_count : int              – states expanded
    """
    kwh_per_pct = battery_kwh / 100

    def delta_w(length_m):
        return math.ceil(consumption_kwh_per_km * length_m / 1000 / battery_kwh * 100)

    def drive_time(data):
        spd = (data.get("maxspeed_kph") or default_speed_kph) * 1000 / 3600
        return data["length"] / spd

    start = (source, initial_charge)
    dist_map = {start: 0.0}
    prev = {}
    visited = set()
    pq = [(0.0, source, initial_charge)]

    while pq:
        cost, u, w = heapq.heappop(pq)
        state = (u, w)
        if state in visited:
            continue
        visited.add(state)

        if u == target and w >= min_final_charge:
            return cost, _reconstruct_ev_path(prev, start, state), len(visited)

        for v, data in graph.get(u, {}).items():
            dw = delta_w(data["length"])
            new_w = w - dw
            if new_w < 0:
                continue
            new_state = (v, new_w)
            if new_state in visited:
                continue
            new_cost = cost + drive_time(data)
            if new_cost < dist_map.get(new_state, math.inf):
                dist_map[new_state] = new_cost
                prev[new_state] = state
                heapq.heappush(pq, (new_cost, v, new_w))

        if w < 100 and u in charging:
            charger_kw = charging[u]
            if math.isnan(charger_kw):
                continue
            charge_cost = kwh_per_pct / charger_kw * 3600
            new_state = (u, w + 1)
            if new_state not in visited:
                new_cost = cost + charge_cost
                if new_cost < dist_map.get(new_state, math.inf):
                    dist_map[new_state] = new_cost
                    prev[new_state] = state
                    heapq.heappush(pq, (new_cost, u, w + 1))

    return math.inf, [], len(visited)


def astar_ev_time(graph, node_info, charging, source, target,
                  battery_kwh, consumption_kwh_per_km=0.2,
                  initial_charge=100, min_final_charge=0,
                  default_speed_kph=80, max_speed_kph=130):
    """
    A* shortest-TIME EV path with variable-speed charging.

    Same model as dijkstra_ev_time but guided by an admissible heuristic:

        h(u) = euclidean_distance(u, target) / max_speed_mps

    Admissible because road distance >= straight-line and actual speed <= max_speed.
    Charging time is omitted from h, so it can only underestimate actual cost.
    All 101 charge levels at the same node share the same h value.

    Parameters mirror dijkstra_ev_time; node_info is needed for coordinates.
    """
    kwh_per_pct = battery_kwh / 100
    max_speed_mps = max_speed_kph * 1000 / 3600
    tx, ty = node_info[target]["x"], node_info[target]["y"]
    # h(u) = straight-line distance to target / max speed.
    # Same value for all 101 charge levels at the same node.
    h_cache = {}
    def h(u):
        if u not in h_cache:
            ux, uy = node_info[u]["x"], node_info[u]["y"]
            h_cache[u] = math.sqrt((ux - tx) ** 2 + (uy - ty) ** 2) / max_speed_mps
        return h_cache[u]

    def delta_w(length_m):
        return math.ceil(consumption_kwh_per_km * length_m / 1000 / battery_kwh * 100)

    def drive_time(data):
        spd = (data.get("maxspeed_kph") or default_speed_kph) * 1000 / 3600
        return data["length"] / spd

    start = (source, initial_charge)
    dist_map = {start: 0.0}
    prev = {}
    visited = set()
    pq = [(h(source), 0.0, source, initial_charge)]

    while pq:
        _, cost, u, w = heapq.heappop(pq)
        state = (u, w)
        if state in visited:
            continue
        visited.add(state)

        if u == target and w >= min_final_charge:
            return cost, _reconstruct_ev_path(prev, start, state), len(visited)

        for v, data in graph.get(u, {}).items():
            dw = delta_w(data["length"])
            new_w = w - dw
            if new_w < 0:
                continue
            new_state = (v, new_w)
            if new_state in visited:
                continue
            new_cost = cost + drive_time(data)
            if new_cost < dist_map.get(new_state, math.inf):
                dist_map[new_state] = new_cost
                prev[new_state] = state
                heapq.heappush(pq, (new_cost + h(v), new_cost, v, new_w))

        if w < 100 and u in charging:
            charger_kw = charging[u]
            if math.isnan(charger_kw):
                continue
            charge_cost = kwh_per_pct / charger_kw * 3600
            new_state = (u, w + 1)
            if new_state not in visited:
                new_cost = cost + charge_cost
                if new_cost < dist_map.get(new_state, math.inf):
                    dist_map[new_state] = new_cost
                    prev[new_state] = state
                    heapq.heappush(pq, (new_cost + h(u), new_cost, u, w + 1))

    return math.inf, [], len(visited)


def summarize_ev_path(path_states, graph, charging, battery_kwh,
                      consumption_kwh_per_km=0.2, default_speed_kph=80):
    """
    Break a path_states sequence into human-readable drive segments and charge events.

    Returns
    -------
    segments : list[dict] – drive legs with distance, time, charge used
    stops    : list[dict] – charging events with charger kw, % added, time spent
    """
    segments, stops = [], []
    i = 0
    while i < len(path_states) - 1:
        cur_node, cur_w = path_states[i]
        nxt_node, nxt_w = path_states[i + 1]

        if nxt_node != cur_node:
            # Drive segment
            edge = graph.get(cur_node, {}).get(nxt_node) or graph.get(nxt_node, {}).get(cur_node) or {}
            length_m = edge.get("length", 0.0)
            spd = (edge.get("maxspeed_kph") or default_speed_kph) * 1000 / 3600
            segments.append({
                "from": cur_node, "to": nxt_node,
                "km": round(length_m / 1000, 2),
                "drive_min": round(length_m / spd / 60, 1),
                "charge_before": cur_w, "charge_after": nxt_w,
            })
            i += 1
        else:
            # Charging session — collect all consecutive 1 % steps at this node
            w_start = cur_w
            charger_kw = charging.get(cur_node, float("nan"))
            while i < len(path_states) - 1 and path_states[i + 1][0] == cur_node:
                i += 1
            w_end = path_states[i][1]
            kwh_added = (w_end - w_start) / 100 * battery_kwh
            charge_min = kwh_added / charger_kw * 60 if not math.isnan(charger_kw) else float("nan")
            stops.append({
                "node": cur_node,
                "charger_kw": charger_kw,
                "from_pct": w_start, "to_pct": w_end,
                "kwh_added": round(kwh_added, 1),
                "charge_min": round(charge_min, 1),
            })
            i += 1

    return segments, stops


def dijkstra_all(graph, source, weight="length"):
    """
    Single-source Dijkstra from source to all reachable nodes.

    Returns
    -------
    dist : dict[int, float] – shortest cost to every reachable node
    prev : dict[int, int]   – predecessor map for path reconstruction
    """
    def edge_cost(data):
        if weight == "time":
            speed_mps = (data["maxspeed_kph"] or 50) * 1000 / 3600
            return data["length"] / speed_mps
        return data["length"]

    dist = {source: 0.0}
    prev = {}
    pq = [(0.0, source)]

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist.get(u, math.inf):
            continue
        for v, data in graph.get(u, {}).items():
            new_cost = cost + edge_cost(data)
            if new_cost < dist.get(v, math.inf):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    return dist, prev


def reconstruct_path(prev, source, target):
    """Reconstruct path from predecessor map."""
    if target not in prev and target != source:
        return []
    path = []
    node = target
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(source)
    path.reverse()
    return path


def dijkstra(graph, source, target, weight="length"):
    """
    Find the shortest path from source to target using Dijkstra's algorithm.

    Parameters
    ----------
    graph      : dict[int, dict[int, dict]] – adjacency list from load_graph
    source     : int – start node id
    target     : int – end node id
    weight     : "length" (metres) or "time" (seconds, derived from length / maxspeed_kph)

    Returns
    -------
    dist       : float  – total cost (metres or seconds), math.inf if unreachable
    path       : list[int] – node ids from source to target (empty if unreachable)
    visited    : int    – number of nodes popped from the priority queue
    """
    def edge_cost(data):
        if weight == "time":
            speed_mps = (data["maxspeed_kph"] or 50) * 1000 / 3600
            return data["length"] / speed_mps
        return data["length"]

    dist = {source: 0.0}
    prev = {}
    pq = [(0.0, source)]
    visited = 0

    while pq:
        cost, u = heapq.heappop(pq)

        if cost > dist.get(u, math.inf):
            continue

        visited += 1

        if u == target:
            break

        for v, data in graph.get(u, {}).items():
            new_cost = cost + edge_cost(data)
            if new_cost < dist.get(v, math.inf):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    if target not in dist:
        return math.inf, [], visited

    path = []
    node = target
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(source)
    path.reverse()

    return dist[target], path, visited


def dijkstra_ev(graph, charging, source, target,
                battery_kwh, consumption_kwh_per_km=0.2,
                initial_charge=100, min_final_charge=0):
    """
    Shortest-distance path for an EV on the expanded state-space graph.

    Each state is (node_id, charge_pct) where charge_pct ∈ {0..100}.
    The graph is explored lazily — no explicit 279K-node structure is built.

    Transitions:
      Drive : (u, w) → (v, w − Δw)  cost = edge length,  only if w − Δw ≥ 0
      Charge: (u, w) → (u, 100)     cost = 0,             only at charging stations

    Δw = ceil(consumption_kwh_per_km × length_km / battery_kwh × 100)
    rounded up so we never underestimate energy use.

    Parameters
    ----------
    graph                  : adjacency list from load_graph
    charging               : dict {node_id: max_kw} from load_charging
    source                 : int – start node (fully charged by default)
    target                 : int – destination node
    battery_kwh            : float – battery capacity (e.g. 80 or 40)
    consumption_kwh_per_km : float – kWh per km (default 0.2)
    initial_charge         : int   – starting charge % (default 100)
    min_final_charge       : int   – minimum charge % required at target (default 0)

    Returns
    -------
    dist          : float         – shortest distance in metres (inf if unreachable)
    path_states   : list[(int,int)] – sequence of (node_id, charge_pct) states
    visited_count : int           – number of states expanded
    """
    def delta_w(length_m):
        kwh = consumption_kwh_per_km * length_m / 1000
        return math.ceil(kwh / battery_kwh * 100)

    start = (source, initial_charge)
    dist_map = {start: 0.0}
    prev = {}
    visited = set()
    pq = [(0.0, source, initial_charge)]

    while pq:
        cost, u, w = heapq.heappop(pq)
        state = (u, w)
        if state in visited:
            continue
        visited.add(state)

        if u == target and w >= min_final_charge:
            return cost, _reconstruct_ev_path(prev, start, state), len(visited)

        for v, data in graph.get(u, {}).items():
            dw = delta_w(data["length"])
            new_w = w - dw
            if new_w < 0:
                continue
            new_state = (v, new_w)
            if new_state in visited:
                continue
            new_cost = cost + data["length"]
            if new_cost < dist_map.get(new_state, math.inf):
                dist_map[new_state] = new_cost
                prev[new_state] = state
                heapq.heappush(pq, (new_cost, v, new_w))

        if u in charging and w < 100:
            new_state = (u, 100)
            if new_state not in visited and cost < dist_map.get(new_state, math.inf):
                dist_map[new_state] = cost
                prev[new_state] = state
                heapq.heappush(pq, (cost, u, 100))

    return math.inf, [], len(visited)
