import os
import plotly.graph_objects as go

_PATH_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


def _layout():
    return go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
    )


def _edge_trace(node_info, edges_df, sample_frac=1.0):
    sampled = edges_df if sample_frac >= 1.0 else edges_df.sample(frac=sample_frac, random_state=42)
    xs, ys = [], []
    for _, row in sampled.iterrows():
        u, v = int(row["u"]), int(row["v"])
        if u in node_info and v in node_info:
            xs += [node_info[u]["x"], node_info[v]["x"], None]
            ys += [node_info[u]["y"], node_info[v]["y"], None]
    return go.Scattergl(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="#888888", width=1.2),
        hoverinfo="none",
        showlegend=False,
        name="Roads",
    )


def _node_trace(node_info):
    ids = list(node_info.keys())
    return go.Scatter(
        x=[node_info[n]["x"] for n in ids],
        y=[node_info[n]["y"] for n in ids],
        mode="markers",
        marker=dict(color="#555555", size=6, opacity=0.6),
        customdata=[[n] for n in ids],
        hovertemplate="id: %{customdata[0]}<br>x: %{x:.0f}<br>y: %{y:.0f}<extra></extra>",
        showlegend=False,
        name="Nodes",
    )


def _save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, stem + ".html")
    fig.write_html(out_path)
    print(f"Saved {out_path}")


def plot_network(node_info, edges_df, sample_frac=1.0):
    """Build the base network figure with edge and node traces."""
    fig = go.Figure(layout=_layout())
    fig.add_trace(_edge_trace(node_info, edges_df, sample_frac))
    fig.add_trace(_node_trace(node_info))
    fig.update_layout(title="Icelandic Road Network")
    return fig, None


def plot_charging(node_info, charging_dict, fig=None, output_dir="output"):
    """Overlay charging stations on an existing figure (or create one)."""
    if fig is None:
        raise ValueError("pass fig returned by plot_network")

    ids = [n for n in charging_dict if n in node_info]
    fig.add_trace(go.Scattergl(
        x=[node_info[n]["x"] for n in ids],
        y=[node_info[n]["y"] for n in ids],
        mode="markers",
        marker=dict(color="red", size=10, symbol="star"),
        text=[f"id: {n}<br>max_kw: {charging_dict[n]:.0f}" for n in ids],
        hoverinfo="text",
        name="Charging station",
    ))

    _save(fig, output_dir, "iceland_network")
    fig.show()
    return fig, None


def plot_single_path(node_info, edges_df, path, label, color, output_dir="output", filename="path.html"):
    """Build one full-network figure with a single highlighted path and save as HTML."""
    fig, _ = plot_network(node_info, edges_df, sample_frac=1.0)

    nodes_in_graph = [n for n in path if n in node_info]
    xs = [node_info[n]["x"] for n in nodes_in_graph]
    ys = [node_info[n]["y"] for n in nodes_in_graph]

    fig.add_trace(go.Scattergl(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=color, width=3),
        name=label,
        hoverinfo="none",
    ))
    fig.add_trace(go.Scattergl(
        x=[xs[0], xs[-1]], y=[ys[0], ys[-1]],
        mode="markers",
        marker=dict(color=color, size=10, symbol=["circle", "square"]),
        showlegend=False,
        hoverinfo="none",
    ))
    fig.update_layout(title=label)
    _save(fig, output_dir, filename)
    return fig


def show_side_by_side(figs, labels, height=500):
    """Display multiple Plotly figures side by side using make_subplots (works in VS Code)."""
    from plotly.subplots import make_subplots

    n = len(figs)
    combined = make_subplots(rows=1, cols=n, subplot_titles=labels)

    for col, fig in enumerate(figs, start=1):
        for trace in fig.data:
            combined.add_trace(trace, row=1, col=col)

    for col in range(1, n + 1):
        combined.update_xaxes(visible=False, scaleanchor=f"y{col if col > 1 else ''}", scaleratio=1, row=1, col=col)
        combined.update_yaxes(visible=False, row=1, col=col)

    combined.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    combined.show()


def plot_ev_path(node_info, edges_df, path_states, charging, label="",
                output_dir="output", filename="ev_path.html"):
    """
    Plot an EV path with charging stops highlighted.

    path_states : list of (node_id, charge_pct) tuples from dijkstra_ev
    """
    fig, _ = plot_network(node_info, edges_df, sample_frac=1.0)

    node_ids = [s[0] for s in path_states]
    charges  = [s[1] for s in path_states]

    xs = [node_info[n]["x"] for n in node_ids if n in node_info]
    ys = [node_info[n]["y"] for n in node_ids if n in node_info]

    # Path line
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="lines",
        line=dict(color="#377eb8", width=3),
        name="EV route", hoverinfo="none",
    ))

    stop_xs, stop_ys, stop_text = [], [], []
    i = 1
    while i < len(path_states):
        n, w = path_states[i]
        prev_n, prev_w = path_states[i - 1]
        if n == prev_n and w > prev_w:
            w_start = prev_w
            while i < len(path_states) - 1 and path_states[i + 1][0] == n:
                i += 1
            stop_xs.append(node_info[n]["x"])
            stop_ys.append(node_info[n]["y"])
            stop_text.append(f"node {n}<br>{w_start}% \u2192 {path_states[i][1]}%")
        i += 1

    if stop_xs:
        fig.add_trace(go.Scattergl(
            x=stop_xs, y=stop_ys, mode="markers",
            marker=dict(color="green", size=12, symbol="star"),
            text=stop_text, hoverinfo="text",
            name="Charging stop",
        ))

    # Start / end markers
    fig.add_trace(go.Scattergl(
        x=[xs[0], xs[-1]], y=[ys[0], ys[-1]], mode="markers",
        marker=dict(color=["blue", "red"], size=12, symbol=["circle", "square"]),
        text=[f"Start ({charges[0]}%)", f"End ({charges[-1]}%)"],
        hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(title=label)
    _save(fig, output_dir, filename)
    fig.show()
    return fig


def plot_visited_comparison(node_info, edges_df, path, dijkstra_visited, astar_visited,
                            label="", output_dir="output", filename="visited_comparison.html"):
    """
    Side-by-side comparison of nodes visited by Dijkstra vs A*.

    Grey = all nodes, orange = visited by the algorithm, red line = shortest path.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Dijkstra ({len(dijkstra_visited)} nodes visited)",
        f"A* ({len(astar_visited)} nodes visited)",
    ])

    all_ids = list(node_info.keys())
    path_set = set(path)

    for col, visited in enumerate([dijkstra_visited, astar_visited], start=1):
        unvisited = [n for n in all_ids if n not in visited]
        vis_ids    = [n for n in all_ids if n in visited and n not in path_set]

        fig.add_trace(go.Scattergl(
            x=[node_info[n]["x"] for n in unvisited],
            y=[node_info[n]["y"] for n in unvisited],
            mode="markers", marker=dict(color="#cccccc", size=4, opacity=0.5),
            showlegend=(col == 1), name="Not visited",
        ), row=1, col=col)

        fig.add_trace(go.Scattergl(
            x=[node_info[n]["x"] for n in vis_ids],
            y=[node_info[n]["y"] for n in vis_ids],
            mode="markers", marker=dict(color="#ff7f00", size=5, opacity=0.8),
            showlegend=(col == 1), name="Visited",
        ), row=1, col=col)

        fig.add_trace(go.Scattergl(
            x=[node_info[n]["x"] for n in path if n in node_info],
            y=[node_info[n]["y"] for n in path if n in node_info],
            mode="lines", line=dict(color="#e41a1c", width=2),
            showlegend=(col == 1), name="Shortest path",
        ), row=1, col=col)

        fig.update_xaxes(visible=False, scaleanchor=f"y{col if col > 1 else ''}", scaleratio=1, row=1, col=col)
        fig.update_yaxes(visible=False, row=1, col=col)

    fig.update_layout(
        title=f"Visited nodes – {label}",
        paper_bgcolor="white", plot_bgcolor="white",
        height=500, margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
    )
    _save(fig, output_dir, filename)
    fig.show()
    return fig


def plot_paths(node_info, edges_df, paths, labels=None, output_dir="output", filename="paths.png"):
    """Plot network with one or more highlighted shortest paths."""
    fig, _ = plot_network(node_info, edges_df, sample_frac=1.0)

    for i, path in enumerate(paths):
        color = _PATH_COLORS[i % len(_PATH_COLORS)]
        label = labels[i] if labels else f"Path {i + 1}"
        nodes_in_graph = [n for n in path if n in node_info]
        xs = [node_info[n]["x"] for n in nodes_in_graph]
        ys = [node_info[n]["y"] for n in nodes_in_graph]

        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=3),
            name=label,
            hoverinfo="none",
        ))
        fig.add_trace(go.Scattergl(
            x=[xs[0], xs[-1]], y=[ys[0], ys[-1]],
            mode="markers",
            marker=dict(color=color, size=10, symbol=["circle", "square"]),
            showlegend=False,
            hoverinfo="none",
        ))

    fig.update_layout(title="Shortest paths – Icelandic Road Network")
    _save(fig, output_dir, filename)
    fig.show()
    return fig, None
