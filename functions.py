import numpy as np
import numpy.typing as npt
import rustworkx as rx
from scipy.spatial.distance import jensenshannon as JSD


W1: float = 0.45
W2: float = 0.45
W3: float = 0.1


def network_node_dispersion(graph: npt.NDArray) -> np.float64:
    return 0.1


def node_distance_distribution(graph: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
    G: rx.PyGraph = rx.PyGraph(multigraph=False).from_adjacency_matrix(
        graph.astype(np.float64)
    )
    dist: npt.NDArray[np.int_] = rx.distance_matrix(G, parallel_threshold=300).astype(
        np.int_
    )
    dist[dist < 0] = dist.shape[0]
    N: np.int_ = dist.max() + 1
    dist_offsets: npt.NDArray[np.int_] = dist + np.arange(dist.shape[0])[:, None] * N
    return np.delete(
        np.bincount(dist_offsets.ravel(), minlength=dist.shape[0] * N).reshape(-1, N)
        / (dist.shape[0] - 1),
        0,
        axis=1,
    )
