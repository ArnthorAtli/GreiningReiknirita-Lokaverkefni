from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path):
    """Return path as-is if it exists, else resolve relative to project root."""
    p = Path(path)
    return p if p.exists() else _PROJECT_ROOT / p


def load_graph(nodes_path, edges_path):
    nodes_df = pd.read_csv(_resolve(nodes_path), sep="\t")
    edges_df = pd.read_csv(_resolve(edges_path), sep="\t")

    node_info = {
        row["id"]: {"x": row["x"], "y": row["y"], "lon": row["lon"], "lat": row["lat"]}
        for _, row in nodes_df.iterrows()
    }

    graph = {node_id: {} for node_id in node_info}

    for _, row in edges_df.iterrows():
        u, v = int(row["u"]), int(row["v"])
        edge_data = {
            "length": row["length"],
            "name": row["name"],
            "maxspeed_kph": row["maxspeed_kph"],
        }
        graph[u][v] = edge_data

        if int(row["oneway"]) == 0:  # 0 = bidirectional
            graph[v][u] = edge_data

    num_edges = sum(len(neighbors) for neighbors in graph.values())
    print(f"Nodes: {len(node_info)}")
    print(f"Edges (directed): {num_edges}")

    return graph, node_info, edges_df


def load_charging(charging_path):
    df = pd.read_csv(_resolve(charging_path), sep="\t")
    charging = df.groupby("nearest_node_id")["max_kw"].max().to_dict()
    print(f"Charging stations: {len(charging)}")
    return charging
