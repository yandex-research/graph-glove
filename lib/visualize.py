from collections import defaultdict

import numpy as np
import torch
from bokeh import plotting as pl, models as bm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import functional as F

from lib import GraphEmbedding, check_numpy, training_mode


def draw_graph(
        x,
        y,
        edges,
        vertex_text=None,
        radius=10,
        vertex_alpha=0.5,
        vertex_color="blue",  # vertex params
        edge_width=1,
        edge_alpha=0.5,
        edge_color="gray",  # edge params
        width=600,
        height=400,
        show=True,
        grid=True,
        **kwargs
):
    """ draws an interactive plot for task points with auxilirary info on hover """
    fig = pl.figure(active_scroll="wheel_zoom", width=width, height=height)
    if not grid:
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.axis.visible = False
    # edges
    edges_ij = np.array([(from_i, to_i) for from_i, to_ix in edges.items() for to_i in to_ix])

    if vertex_text is None:
        vertex_text = ["" for i in range(len(x))]

    def _select_edges(field):
        if isinstance(field, dict):
            return [field[from_i][to_i] for from_i, to_i in edges_ij]
        else:
            return [field] * len(edges_ij)

    edge_source = bm.ColumnDataSource(
        {
            "xx": x[edges_ij].tolist(),
            "yy": y[edges_ij].tolist(),
            "alpha": _select_edges(edge_alpha),
            "color": _select_edges(edge_color),
            "width": _select_edges(edge_width),
        }
    )
    fig.multi_line(
        "xx", "yy", color="color", line_width="width", alpha="alpha", source=edge_source
    )

    # vertices
    def _maybe_repeat(x, size):
        if not hasattr(x, "__len__") or len(x) != size:
            x = [x] * size
        return x

    vertex_source = bm.ColumnDataSource(
        {
            "x": x,
            "y": y,
            "color": _maybe_repeat(vertex_color, len(x)),
            "alpha": _maybe_repeat(vertex_alpha, len(x)),
            "vertex_text": _maybe_repeat(vertex_text, len(x)),
            **kwargs,
        }
    )
    fig.scatter(
        "x",
        "y",
        size=radius,
        color="color",
        alpha="alpha",
        name="vertices",
        source=vertex_source,
    )
    fig.text("x", "y", "vertex_text", text_align="center", name="texts", y_offset=-8, text_font='helvetica',
             source=vertex_source)

    fig.add_tools(
        bm.HoverTool(
            tooltips=[(key, "@" + key) for key in kwargs.keys()], names=["vertices"]
        )
    )
    if show:
        pl.show(fig)
    return fig


def rgba_to_hex(rgba_matrix):
    assert rgba_matrix.min() >= 0.0 and rgba_matrix.max() <= 1.0
    assert rgba_matrix.ndim == 2 and rgba_matrix.shape[1] in (3, 4)
    colors_hex = []
    hexify = lambda x: '0' * max(0, 4 - len(hex(int(round(x * 255))))) + hex(int(round(x * 255))).replace('0x', '')
    for rgba in rgba_matrix:
        colors_hex.append(
            "#" + "".join(map(hexify, rgba))
        )
    return colors_hex


def visualize_embeddings(emb: GraphEmbedding, coords=None, vertex_labels=None,
                         deterministic=None, edge_probability_threshold=0.5, weighted=False, scale_factor=3.0,
                         cmap=plt.get_cmap('nipy_spectral_r'), **kwargs):
    """
    Draws learned graph using bokeh and some magic. Please set bokeh output (notebook / file / etc.) in advance
    :type emb: GraphEmbedding
    :param coords: a matrix[num_vertices, 2] of 2d point vertex coordinates, defaults to TSNE on pairwise distances
    :param vertex_labels: if given, assigns a label to each vertex and paints it to the respective color
    :param deterministic: if True, only use edges with p >= 0.5, otherwise sample edges with learned probability
    :param weighted: if True, edge widths are inversely proportional to their weights, default = all widths are equal
    :param scale_factor: multiplies edge widths by this number
    :param cmap: a callable(array) -> rgb(a) matrix used to paint vertices if vertex_labels are specified
    :param kwargs: see utils.draw_graph
    """
    if deterministic is None:
        deterministic = emb.training

    # handle edges
    from_ix, to_ix = emb.edge_sources, emb.edge_targets
    weights = F.softplus(emb.edge_weight_logits).view(-1).data.numpy()
    mean_weight = weights[1:].mean()
    num_vertices, num_edges = len(emb.slices) - 1, len(from_ix)
    edge_probabilities = torch.sigmoid(emb.edge_adjacency_logits.view(-1)).data.numpy()
    if deterministic:
        existence = edge_probabilities >= edge_probability_threshold
    else:
        existence = np.random.rand(num_edges) < edge_probabilities

    edge_dict = defaultdict(list)
    edge_width = defaultdict(dict)
    for edge_i in range(1, num_edges):  # skip first "technical" loop edge
        if existence[edge_i]:
            from_i, to_i, weight = from_ix[edge_i], to_ix[edge_i], weights[edge_i]
            edge_dict[from_i].append(to_i)
            edge_width[from_i][to_i] = scale_factor / (weight / mean_weight + 1e-3) \
                if weighted else 1.0

    # handle vertices
    if coords is None:
        pairwise_distances = emb.compute_pairwise_distances(edge_threshold=edge_probability_threshold)
        pairwise_distances[np.isinf(pairwise_distances)] = np.max(pairwise_distances[np.isfinite(pairwise_distances)])
        # ^-- [num_vertices x num_vertices]
        coords = TSNE(metric='precomputed').fit_transform(pairwise_distances)

    if vertex_labels is not None:
        vertex_color = (vertex_labels - np.min(vertex_labels)) * 1.0 / np.max(vertex_labels)
        vertex_color = rgba_to_hex(cmap(vertex_color)[:, :3])
    else:
        vertex_color = 'blue'

    vertex_stats = dict(
        vertex_id=np.arange(num_vertices),
        num_edges=np.array([np.sum(check_numpy(emb.get_edges(i).p_adjacent) >= edge_probability_threshold)
                            for i in range(emb.num_vertices)], dtype='int32')
    )

    assert coords.shape == (num_vertices, 2)
    if vertex_labels is not None:
        assert vertex_labels.shape == (num_vertices,)
        vertex_stats['label'] = vertex_labels

    return draw_graph(*coords.T, edges=edge_dict, edge_width=edge_width,
                      vertex_color=vertex_color, **vertex_stats, **kwargs)


def visualize_cluster(emb: GraphEmbedding, cluster_vertex_ids, dictionary, coords=None, vertex_labels=None,
                      deterministic=None, edge_probability_threshold=0.5, weighted=False, scale_factor=3.0,
                      cmap=plt.get_cmap('nipy_spectral_r'), default_color='#c8eda8', **kwargs):
    """ Like visualize_embeddings, but visualizes only a single cluster of vertices """

    cluster_vertices = list(cluster_vertex_ids.nodes)

    # 1. assemble in-cluster edges

    from_ix, to_ix = emb.edge_sources, emb.edge_targets
    weights = F.softplus(emb.edge_weight_logits).view(-1).data.numpy()
    mean_weight = weights[1:].mean()
    num_vertices, num_edges = len(emb.slices) - 1, len(from_ix)
    edge_probabilities = torch.sigmoid(emb.edge_adjacency_logits.view(-1)).data.numpy()
    if deterministic:
        existence = edge_probabilities >= edge_probability_threshold
    else:
        existence = np.random.rand(num_edges) < edge_probabilities

    edge_dict = defaultdict(list)
    edge_width = defaultdict(dict)
    for edge_i in range(1, num_edges):  # skip first "technical" loop edge
        if existence[edge_i]:
            from_i, to_i, weight = from_ix[edge_i], to_ix[edge_i], weights[edge_i]
            if from_i in cluster_vertices and to_i in cluster_vertices:
                from_c, to_c = map(cluster_vertices.index, [from_i, to_i])
                edge_dict[from_c].append(to_c)
                edge_width[from_c][to_c] = scale_factor / (weight / mean_weight + 1e-3) \
                    if weighted else 1.0

    # 2. compute pairwise distances
    vertex_ids = torch.as_tensor(list(cluster_vertex_ids.nodes), dtype=torch.int32)
    targets = torch.as_tensor(np.repeat(vertex_ids[None], len(vertex_ids), axis=0), dtype=torch.int32)

    with training_mode(emb, is_train=False):
        pairwise_distances = check_numpy(
            GraphEmbedding.forward(emb, vertex_ids, targets, soft=False)['target_distances'])

    pairwise_distances[np.isinf(pairwise_distances)] = np.max(pairwise_distances[np.isfinite(pairwise_distances)])
    # ^-- [num_vertices x num_vertices]
    if coords is None:
        coords = TSNE(metric='precomputed').fit_transform(pairwise_distances)

    # 3. assemble graph metadate
    cluster_size = len(vertex_ids)

    if vertex_labels is not None:
        vertex_color = (vertex_labels - np.min(vertex_labels)) * 1.0 / np.max(vertex_labels)
        vertex_color = rgba_to_hex(cmap(vertex_color)[:, :3])
    else:
        vertex_color = default_color

    vertex_stats = dict(
        vertex_id=np.arange(cluster_size),
        num_edges=np.array([np.sum(check_numpy(emb.get_edges(i).p_adjacent) >= edge_probability_threshold)
                            for i in vertex_ids], dtype='int32')
    )

    assert coords.shape == (cluster_size, 2)
    if vertex_labels is not None:
        assert vertex_labels.shape == (cluster_size,)
        vertex_stats['label'] = vertex_labels

    ix_to_token = {i: t for t, i in dictionary.items()}
    tokens = list(map(ix_to_token.get, map(int, vertex_ids)))

    # ... and finally, draw the resulting graph
    return draw_graph(
        *coords.T, edges=edge_dict, vertex_text=tokens, edge_width=edge_width, token=tokens, vertex_color=vertex_color,
        **vertex_stats, vertex_alpha=1.0, edge_alpha=0.25, **kwargs)

