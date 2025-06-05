from scipy.spatial import cKDTree
import torch
from torch_cluster import knn_graph
# from torch_geometric.nn import knn_graph


def enforce_exact_k_cuda(edge_index: torch.LongTensor, k: int, N: int) -> torch.LongTensor:
    """
    Given edge_index of shape [2, E] on CUDA, produce a new [2, E'] where each dst node has <= k incoming edges.
    Assumes edge_index is on the same device (e.g. CUDA).
    """
    device = edge_index.device
    src = edge_index[0]         # [E]
    dst = edge_index[1]         # [E]

    # 1) sort by dst
    sorted_dst, perm = dst.sort()
    sorted_src = src[perm]

    # 2) find unique dst values and counts
    unique_vals, counts = torch.unique(sorted_dst, return_counts=True)
    # counts is [#unique], sums to E

    # 3) compute start index of each group via cumsum
    #    “group_starts[i]” is the first index in sorted_dst where unique_vals[i] appears
    group_ends = counts.cumsum(dim=0)           # ends of each group
    group_starts = group_ends - counts          # starts of each group

    # 4) assign each sorted element a “group index” by doing unique on sorted_dst
    #    (we know sorted_dst is sorted, so unique preserves order)
    _, inverse = torch.unique(sorted_dst, return_inverse=True)
    #  `inverse[j]` tells us which unique_vals group sorted_dst[j] belongs to

    # 5) for each position j in [0..E), compute its rank within its group:
    idx = torch.arange(sorted_dst.size(0), device=device)
    start_per_elem = group_starts[inverse]     # broadcast: for each j, get the group’s start index
    rank_within_group = idx - start_per_elem   # 0,1,2,... within each block of identical dst

    # 6) mask = rank < k  → keep only the first k of each group
    keep_mask = rank_within_group < k          # [E] boolean

    # 7) apply that mask to the original permuted indices to recover final edges
    final_perm = perm[keep_mask]
    return edge_index[:, final_perm]

def knn_neighbors(x, args):
    B, N, C = x.shape
    x_flat = x.reshape(B * N, C)
    k = args.knn
    batch = torch.arange(B, device=x.device).repeat_interleave(N)
    edge_index = knn_graph(x_flat, k=k, batch=batch, loop=False, flow = 'source_to_target') # 'target_to_source')
    edge_index = enforce_exact_k_cuda(edge_index, k, N)
    src, dst = edge_index
    dst_batch = batch[dst]  
    dst_idx_local = dst % N
    src_idx_local = src % N

    idx = torch.zeros((B, N, k), dtype=torch.long, device=x.device)
    arange_k = torch.arange(k, device=x.device).repeat(B * N)[:dst.size(0)]

    idx[dst_batch, dst_idx_local, arange_k] = src_idx_local

    neighbors = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1),
        dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, -1, C)
    )

    central = x.unsqueeze(2).expand(-1, -1, k, -1)
    edge_feats = torch.cat([central, neighbors - central], dim=-1)
    return edge_feats, neighbors


def compute_local_covariance(points):
    """
    Compute the local covariance matrix for each point cloud.

    Args:
        points: Tensor of shape [batch_size, num_points, k_nearest, 3].

    Returns:
        covariances: Tensor of shape [batch_size, num_points, 9].
    """
    # Calculate mean across neighbors (dim=-2)
    means = points.mean(dim=-2, keepdim=True)  # Shape: [batch_size, num_points, 1, 3]

    # Subtract mean from neighbors
    centered_points = points - means  # Shape: [batch_size, num_points, k_nearest, 3]

    # Compute outer product of centered points (batched matrix multiplication)
    # Reshape centered points for bmm
    centered_points_flat = centered_points.view(-1, points.size(-2), points.size(-1))  # Shape: [B*num_points, k_nearest, 3]
    cov_matrices = torch.bmm(centered_points_flat.transpose(1, 2), centered_points_flat)  # Shape: [B*num_points, 3, 3]

    # Normalize by the number of neighbors (k_nearest)
    cov_matrices /= points.size(-2)  # Normalize by K

    # Reshape back to batch structure and flatten the 3x3 matrix
    cov_matrices = cov_matrices.view(points.size(0), points.size(1), 9)  # Shape: [batch_size, num_points, 9]

    return cov_matrices
