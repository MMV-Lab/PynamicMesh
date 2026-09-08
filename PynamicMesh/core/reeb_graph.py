import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from tqdm.auto import tqdm
from pyFM.mesh import TriMesh
import seaborn as sns
import networkx as nx
import pickle
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse import linalg as splinalg
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import wasserstein_distance
import warnings
from PynamicMesh.utils.tools import mesh_mat2object

try:
    import cupy as xp
    GPU_AVAILABLE = True
    print("[INFO] CuPy detected. Utilizing GPU for Reeb Graph spectral computations and scalar fields.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")


def to_gpu(arr):
    """Moves a numpy array to the GPU if available."""
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return xp.asarray(arr)
    return arr


def to_cpu(arr):
    """Moves a CuPy array back to the CPU for NetworkX/PyVista/Saving."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


# ----------------------------------------------------------------------------- #
#  Dynamic graph analysis (unchanged behaviour)
# ----------------------------------------------------------------------------- #

def _frame_number(path):
    """Extracts the integer frame id from names like 'Reeb_T12.pkl' (falls back to name order)."""
    m = re.search(r'T(\d+)', Path(path).stem)
    return int(m.group(1)) if m else float('inf')


def graph_time_analysis(reeb_folder_path, single_file=True):
    if single_file:
        print("\nStarting Dynamic Graph Analysis...")
    reeb_folder = Path(reeb_folder_path)
    # Natural sort on the frame number: a plain string sort puts Reeb_T10 before Reeb_T2.
    reeb_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.suffix == '.pkl'],
                        key=_frame_number)
    frame_ids = [_frame_number(f) for f in reeb_files]

    if len(reeb_files) < 2:
        print("Not enough Reeb graphs found to perform time analysis.")
        return

    analysis_folder = reeb_folder.parent / 'Graph_analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    features_list = []

    with open(reeb_files[0], 'rb') as f:
        G_prev = pickle.load(f)

    for i in tqdm(range(1, len(reeb_files)), desc="Computing Graph Metrics", leave=single_file):
        with open(reeb_files[i], 'rb') as f:
            G_curr = pickle.load(f)

        v_prev, e_prev = G_prev.number_of_nodes(), G_prev.number_of_edges()
        v_curr, e_curr = G_curr.number_of_nodes(), G_curr.number_of_edges()

        c_prev = nx.number_connected_components(G_prev) if v_prev > 0 else 0
        c_curr = nx.number_connected_components(G_curr) if v_curr > 0 else 0
        betti_prev = e_prev - v_prev + c_prev
        betti_curr = e_curr - v_curr + c_curr

        if v_curr > 0 and c_curr > 0:
            largest_cc = max(nx.connected_components(G_curr), key=len)
            sub_G = G_curr.subgraph(largest_cc)
            diameter = nx.diameter(sub_G)
            radius = nx.radius(sub_G)
        else:
            diameter, radius = 0, 0

        degrees_prev = [d for n, d in G_prev.degree()] if v_prev > 0 else [0]
        degrees_curr = [d for n, d in G_curr.degree()] if v_curr > 0 else [0]

        deg_wasserstein = wasserstein_distance(degrees_prev, degrees_curr)

        if v_prev > 0 and v_curr > 0:
            lap_prev_np = np.asarray(nx.normalized_laplacian_matrix(G_prev).todense())
            lap_curr_np = np.asarray(nx.normalized_laplacian_matrix(G_curr).todense())

            lap_prev_gpu = to_gpu(lap_prev_np)
            lap_curr_gpu = to_gpu(lap_curr_np)

            evals_prev = xp.linalg.eigvalsh(lap_prev_gpu)
            evals_curr = xp.linalg.eigvalsh(lap_curr_gpu)

            max_len = max(len(evals_prev), len(evals_curr))
            e_p_pad = xp.pad(evals_prev, (0, max_len - len(evals_prev)))
            e_c_pad = xp.pad(evals_curr, (0, max_len - len(evals_curr)))

            spectral_dist = float(to_cpu(xp.linalg.norm(e_p_pad - e_c_pad)))
        else:
            spectral_dist = 0.0

        # NOTE: graph_edit_distance is exponential; with a timeout NetworkX returns the
        # best *upper bound* found so far, so this value is approximate and can vary run to run.
        ged = nx.graph_edit_distance(G_prev, G_curr, timeout=2)
        if ged is None:
            ged = abs(v_curr - v_prev) + abs(e_curr - e_prev)

        features_list.append({
            'Transition': f"T{frame_ids[i-1]} -> T{frame_ids[i]}",
            'Time_Step': frame_ids[i-1],
            'Nodes_T': v_curr,
            'Edges_T': e_curr,
            'Delta_Nodes': v_curr - v_prev,
            'Delta_Edges': e_curr - e_prev,
            'Betti_1_Cycles': betti_curr,
            'Graph_Density': nx.density(G_curr) if v_curr > 1 else 0,
            'LCC_Diameter': diameter,
            'LCC_Radius': radius,
            'Deg_Wasserstein_Dist': deg_wasserstein,
            'Spectral_Laplacian_Dist': spectral_dist,
            'Graph_Edit_Dist': ged
        })

        G_prev = G_curr

    df = pd.DataFrame(features_list)
    csv_out_path = analysis_folder / 'time_analysis.csv'
    df.to_csv(csv_out_path, index=False)

    if single_file:
        print(f"Graph analysis complete. Data saved to: {csv_out_path}")

    return str(csv_out_path)


def plot_dynamic_graph_analysis(csv_path, single_file=True):
    """
    Generates time-series reports visualizing the dynamic features of the Reeb graphs.
    Saves the plots as PNG files in a 'plots' subdirectory.
    """
    if single_file:
        print("\nGenerating visual reports for Graph Dynamics...")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Could not find CSV file at {csv_path}")
        return

    df = pd.read_csv(csv_file)
    plots_folder = csv_file.parent / 'plots'
    os.makedirs(plots_folder, exist_ok=True)

    time_steps = df['Time_Step']
    sns.set_theme(style="whitegrid")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(time_steps, df['Nodes_T'], label='Nodes', marker='o', color='#1f77b4', linewidth=2)
    ax1.plot(time_steps, df['Edges_T'], label='Edges', marker='s', color='#ff7f0e', linewidth=2)
    ax1.set_title('Reeb Graph Structural Size Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step (T)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(plots_folder / '1_Structural_Size.png', dpi=200)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes2[0].plot(time_steps, df['Deg_Wasserstein_Dist'], color='purple', marker='o')
    axes2[0].set_title('Degree Distribution Shift (Wasserstein Distance)')
    axes2[0].set_ylabel('Distance')

    axes2[1].plot(time_steps, df['Spectral_Laplacian_Dist'], color='teal', marker='D')
    axes2[1].set_title('Global Shape Shift (Spectral Laplacian Distance)')
    axes2[1].set_ylabel('L2 Norm Diff')

    axes2[2].plot(time_steps, df['Graph_Edit_Dist'], color='crimson', marker='X')
    axes2[2].set_title('Transformation Cost (Graph Edit Distance)')
    axes2[2].set_ylabel('Edit Cost')
    axes2[2].set_xlabel('Time Step (Transition $T_{n-1} \u2192 T_n$)')

    fig2.suptitle('Dynamic Graph Similarity Metrics', fontsize=16, fontweight='bold', y=0.98)
    fig2.tight_layout()
    fig2.savefig(plots_folder / '2_Graph_Distances.png', dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3_tw = ax3.twinx()

    l1 = ax3.plot(time_steps, df['Betti_1_Cycles'], color='darkgreen', marker='^', label='Betti-1 (Cycles)')
    l2 = ax3_tw.plot(time_steps, df['LCC_Diameter'], color='navy', marker='v', linestyle='--', label='Diameter (LCC)')

    ax3.set_title('Topology & Spatial Span', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Step (T)', fontsize=12)
    ax3.set_ylabel('Number of Cycles (Betti-1)', color='darkgreen', fontsize=12)
    ax3_tw.set_ylabel('Graph Diameter (Hops)', color='navy', fontsize=12)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    fig3.tight_layout()
    fig3.savefig(plots_folder / '3_Internal_Topology.png', dpi=200)
    plt.close(fig3)

    if single_file:
        print(f"Visual reports generated successfully in: {plots_folder}")


# ----------------------------------------------------------------------------- #
#  Shared helpers
# ----------------------------------------------------------------------------- #

def _as_faces(faces):
    """Return faces as an (F, 3) int64 numpy array, accepting pyvista's padded format too."""
    f = np.asarray(to_cpu(faces))
    if f.ndim == 1:
        if f.size % 4 != 0 or not np.all(f[::4] == 3):
            raise ValueError("Flat face array must be in pyvista [3, i, j, k, ...] format.")
        f = f.reshape(-1, 4)[:, 1:]
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3); got {f.shape}.")
    return f.astype(np.int64)


def _area_weighted_center(vertices, faces):
    """Center of mass of the surface (area weighted), independent of vertex density."""
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    tri = v[f]                                          # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)              # (F,)
    centroid = tri.mean(axis=1)                         # (F, 3)
    total = area.sum()
    if total <= 0:
        return v.mean(axis=0)
    return (centroid * area[:, None]).sum(axis=0) / total


def _resolve_vertex_indices(spec, vertices, faces, default):
    """
    Normalizes a user-provided vertex selection into a 1-D int array.

    Accepted: int, list/array of ints, 'mass_center' (or a list whose first element
    is 'mass_center'), or None (-> `default`).
    """
    if spec is None:
        spec = default
    if isinstance(spec, str):
        spec = [spec]
    spec = list(np.atleast_1d(spec)) if not isinstance(spec, list) else spec

    if len(spec) > 0 and isinstance(spec[0], str):
        if spec[0] != 'mass_center':
            raise ValueError(f"Unknown vertex selector '{spec[0]}'. Use an index, a list of indices or 'mass_center'.")
        v = np.asarray(to_cpu(vertices), dtype=np.float64)
        center = _area_weighted_center(v, faces)
        return np.array([int(np.argmin(np.linalg.norm(v - center, axis=1)))])

    idx = np.asarray(spec, dtype=np.int64).ravel()
    n = np.asarray(vertices).shape[0]
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
        raise IndexError(f"Vertex indices {idx} out of range for a mesh with {n} vertices.")
    return idx


def _rank_normalize(x):
    """
    Histogram equalization: maps values to their normalized rank in [0, 1].

    The Reeb graph only depends on the *ordering* of the scalar field, so this is a
    topology-preserving transform. It guarantees that the bins used by
    compute_approx_reeb_graph are equally populated, which is far more robust for
    heavy-tailed fields (curvature, kernels) than uniform bins in value space.
    Ties are broken deterministically by index (stable sort).
    """
    x_gpu = to_gpu(np.asarray(x, dtype=np.float64))
    order = xp.argsort(x_gpu, kind='stable')
    ranks = xp.empty(len(order), dtype=xp.float64)
    ranks[order] = xp.arange(len(order), dtype=xp.float64)
    denom = max(len(order) - 1.0, 1.0)
    return to_cpu(ranks / denom)


def _robust_clip(x, lower_pct=0.5, upper_pct=99.5):
    """Replaces non-finite values and clips extreme outliers (typical for discrete curvature)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.percentile(x[finite], [lower_pct, upper_pct])
    x = np.where(finite, x, np.median(x[finite]))
    return np.clip(x, lo, hi)


def _sanitize_field(field, num_vertices, method):
    """Final guard: correct shape, float64, finite, non-degenerate."""
    field = np.asarray(to_cpu(field), dtype=np.float64).ravel()
    if field.shape[0] != num_vertices:
        raise ValueError(f"Scalar field '{method}' has {field.shape[0]} values but mesh has {num_vertices} vertices.")
    if not np.all(np.isfinite(field)):
        n_bad = int(np.count_nonzero(~np.isfinite(field)))
        warnings.warn(f"Scalar field '{method}': {n_bad} non-finite values replaced (median / clipped).")
        field = _robust_clip(field, 0.0, 100.0)
    if np.ptp(field) <= 1e-12 * max(1.0, np.abs(field).max()):
        warnings.warn(f"Scalar field '{method}' is (numerically) constant; the Reeb graph will collapse to a single node.")
    return field


# ----------------------------------------------------------------------------- #
#  Scalar fields
# ----------------------------------------------------------------------------- #

def _get_mesh_adjacency(vertices, faces):
    """Sparse symmetric adjacency matrix weighted by Euclidean edge length."""
    v_gpu = to_gpu(np.asarray(to_cpu(vertices), dtype=np.float64))
    f_gpu = to_gpu(_as_faces(faces))

    edges_gpu = xp.vstack([f_gpu[:, [0, 1]], f_gpu[:, [1, 2]], f_gpu[:, [2, 0]]])
    weights_gpu = xp.linalg.norm(v_gpu[edges_gpu[:, 0]] - v_gpu[edges_gpu[:, 1]], axis=1)

    edges = to_cpu(edges_gpu)
    weights = to_cpu(weights_gpu)
    keep = edges[:, 0] != edges[:, 1]                    # drop degenerate edges
    edges, weights = edges[keep], weights[keep]

    n = np.asarray(vertices).shape[0]
    adj = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    return adj.maximum(adj.T)


def compute_geodesic_distance(vertices, faces, vertex_ref_index, solver="heat"):
    """
    Geodesic distance from a *set* of source vertices (distance to the nearest source).

    solver='heat'     : pyFM heat method (smooth, fast). Falls back to Dijkstra on failure
                        or if the result is not finite.
    solver='dijkstra' : exact shortest path on the edge graph (always finite, slightly
                        over-estimates true geodesics, but perfectly stable across frames).
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    refs = np.atleast_1d(np.asarray(vertex_ref_index, dtype=np.int64))
    n = v.shape[0]

    if solver == "heat":
        try:
            d = np.asarray(TriMesh(v, f).geod_from(refs.tolist() if refs.size > 1 else int(refs[0])))
            # pyFM returns (n,) for one source or (n, k) / (k, n) for several -> distance to the set
            if d.ndim == 2:
                d = d.min(axis=1) if d.shape[0] == n else d.min(axis=0)
            d = d.ravel()
            if d.shape[0] == n and np.all(np.isfinite(d)):
                return d
            warnings.warn("Heat-method geodesics returned an invalid result; falling back to Dijkstra.")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Heat-method geodesics failed ({exc}); falling back to Dijkstra.")

    adj = _get_mesh_adjacency(v, f)
    d = dijkstra(adj, directed=False, indices=refs, min_only=True)
    if np.isinf(d).any():
        # Disconnected components: give unreachable vertices a distance beyond the max.
        finite_max = d[np.isfinite(d)].max() if np.isfinite(d).any() else 0.0
        warnings.warn("Mesh has vertices unreachable from the geodesic source(s); assigning max+1.")
        d = np.where(np.isinf(d), finite_max + 1.0, d)
    return d


def compute_harmonic_field(trimesh_obj, source_idx, sink_idx):
    """
    Solves Δf = 0 with Dirichlet conditions f(source)=1, f(sink)=0 using exact
    constraint elimination:   W_II f_I = -W_IB f_B.
    This replaces the penalty formulation (1e8 on the diagonal), which is badly
    conditioned for the cotangent matrix and silently inaccurate on fine meshes.
    """
    W = csr_matrix(trimesh_obj.W)
    n = W.shape[0]
    src = np.atleast_1d(np.asarray(source_idx, dtype=np.int64))
    snk = np.atleast_1d(np.asarray(sink_idx, dtype=np.int64))
    if np.intersect1d(src, snk).size:
        raise ValueError("Harmonic field: source and sink sets overlap.")

    f = np.zeros(n)
    f[src] = 1.0
    boundary = np.union1d(src, snk)
    interior = np.setdiff1d(np.arange(n), boundary)

    W_II = W[interior][:, interior]
    W_IB = W[interior][:, boundary]
    rhs = -W_IB @ f[boundary]

    # A component without any constrained vertex makes W_II singular -> tiny Tikhonov term.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f[interior] = splinalg.spsolve(W_II.tocsc(), rhs)
            if not np.all(np.isfinite(f)):
                raise np.linalg.LinAlgError
        except Exception:  # noqa: BLE001
            eps = 1e-8 * (abs(W_II.diagonal()).mean() + 1e-12)
            f[interior] = splinalg.spsolve((W_II + eps * identity(W_II.shape[0])).tocsc(), rhs)
    return np.clip(f, 0.0, 1.0)     # discrete maximum principle; clip only removes round-off


def _spectral_basis(trimesh_obj):
    if trimesh_obj is None or getattr(trimesh_obj, "eigenvalues", None) is None:
        raise ValueError("trimesh_obj with a computed LB eigendecomposition (mesh.process(k=...)) is required.")
    evals = np.asarray(trimesh_obj.eigenvalues, dtype=np.float64)
    evecs = np.asarray(trimesh_obj.eigenvectors, dtype=np.float64)
    return evals, evecs


def _auto_heat_time(evals):
    """Geometric mean of the two extreme resolvable diffusion times, 1/λ_1 and 1/λ_K."""
    pos = evals[evals > 1e-12]
    if pos.size < 2:
        return 1.0
    return float(1.0 / np.sqrt(pos[0] * pos[-1]))


def compute_heat_diffusion(trimesh_obj, source_idx, t="auto"):
    """
    Heat kernel k_t(source, x) = Σ_k e^{-t λ_k} φ_k(source) φ_k(x), summed over the source set.

    t='auto' picks a scale-aware time (see _auto_heat_time). A fixed t (the old default
    was 10.0) is *not* mesh-independent: for un-normalized meshes with small eigenvalues
    it yields a nearly constant field, for large eigenvalues it collapses to a delta.
    """
    evals, evecs = _spectral_basis(trimesh_obj)
    src = np.atleast_1d(np.asarray(source_idx, dtype=np.int64))
    if t == "auto" or t is None:
        t = _auto_heat_time(evals)

    evals_gpu = to_gpu(evals)
    evecs_gpu = to_gpu(evecs)
    phi_src = evecs_gpu[to_gpu(src), :].sum(axis=0)             # (K,)
    weights = xp.exp(-float(t) * evals_gpu) * phi_src           # (K,)
    return to_cpu(evecs_gpu @ weights)


def compute_matern_field(trimesh_obj, source_idx, nu=1.5, lengthscale=1.0):
    """
    Matérn kernel on the surface via its Karhunen–Loève expansion in the LB basis:
        k(source, x) = Σ_k (2ν/ℓ² + λ_k)^{-(ν + d/2)} φ_k(source) φ_k(x),   d = 2.
    This is exactly what geometric_kernels computes internally; using the pyFM basis
    directly avoids a second eigendecomposition and the fragile private-attribute patching.
    """
    evals, evecs = _spectral_basis(trimesh_obj)
    src = np.atleast_1d(np.asarray(source_idx, dtype=np.int64))
    if np.isinf(nu):
        spectrum = np.exp(-0.5 * lengthscale ** 2 * evals)          # squared-exponential limit
    else:
        spectrum = (2.0 * nu / lengthscale ** 2 + evals) ** (-(nu + 1.0))

    evecs_gpu = to_gpu(evecs)
    phi_src = evecs_gpu[to_gpu(src), :].sum(axis=0)
    return to_cpu(evecs_gpu @ (to_gpu(spectrum) * phi_src))


def compute_curvature_field(vertices, faces, method):
    """
    Discrete curvatures via VTK (pyvista). VTK's estimates are noisy and can be ±inf on
    degenerate triangles / boundaries, so the result is robustly clipped. Note that these
    are scale dependent (units of 1/length), so compare frames only at a common scale.
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    faces_pv = np.hstack([np.full((f.shape[0], 1), 3, dtype=np.int64), f]).ravel()
    mesh_pv = pv.PolyData(v, faces_pv)

    H = _robust_clip(mesh_pv.curvature(curv_type="mean"))
    if method == "mean_curvature":
        return H
    K = _robust_clip(mesh_pv.curvature(curv_type="Gaussian"))
    if method == "gaussian_curvature":
        return K

    # Principal curvatures: κ1,2 = H ± sqrt(H² - K)   (K <= H² for a real surface)
    disc = np.sqrt(np.maximum(H ** 2 - K, 0.0))
    if method == "shape_index":
        # Koenderink shape index in [-1, 1]; undefined (0) on umbilical points where κ1 == κ2
        with np.errstate(divide="ignore", invalid="ignore"):
            S = (2.0 / np.pi) * np.arctan2(H, disc)
        return np.nan_to_num(S, nan=0.0)
    if method == "curvedness":
        return np.sqrt(np.maximum(2.0 * H ** 2 - K, 0.0))
    raise ValueError(method)


def compute_vertex_normals(vertices, faces):
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])   # area-weighted face normals
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(norms, 1e-12)


def compute_normal_displacement(vertices, faces, prev_vertices, p2p, signed=True):
    """
    Normal component of the displacement between the previous frame (mapped through the
    point-to-point map p2p: current vertex j -> previous vertex p2p[j]) and the current frame.
    Uses the project's flow decomposition if importable, otherwise a direct projection
    on area-weighted vertex normals.
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    pv_prev = np.asarray(to_cpu(prev_vertices), dtype=np.float64)
    p2p = np.asarray(to_cpu(p2p), dtype=np.int64).ravel()
    if p2p.shape[0] != v.shape[0]:
        raise ValueError(f"p2p has {p2p.shape[0]} entries but the current mesh has {v.shape[0]} vertices "
                         "(expected the mesh2 -> mesh1 map, p2p_21).")
    displacements = v - pv_prev[p2p]

    try:
        from PynamicMesh.core.physic_model import compute_flow_decomposition
        norm_mag, _ = compute_flow_decomposition(v, _as_faces(faces), displacements)
        return np.asarray(norm_mag, dtype=np.float64).ravel()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"compute_flow_decomposition unavailable ({exc}); using direct normal projection.")
        normals = compute_vertex_normals(v, faces)
        comp = np.einsum("ij,ij->i", displacements, normals)
        return comp if signed else np.abs(comp)


def get_scalar_field(vertices, faces, method="z", prev_vertices=None, p2p=None, trimesh_obj=None, **kwargs):
    """
    Computes a per-vertex scalar field f: V -> R used to build the Reeb graph.

    Common kwargs
    -------------
    equalize_histogram : bool   rank-normalize the field to [0,1] (topology preserving).
                                Default True for 'heat_diffusion'/'matern_kernel', False otherwise.
    Method-specific kwargs are documented inline.
    """
    method = method.lower()
    v_cpu = np.asarray(to_cpu(vertices), dtype=np.float64)
    num_vertices = v_cpu.shape[0]
    v_gpu = to_gpu(v_cpu)
    equalize_default = method in ("heat_diffusion", "matern_kernel")

    if method in ("x", "y", "z"):
        field = v_gpu[:, "xyz".index(method)]

    elif method in ("signed_dist_x", "signed_dist_y", "signed_dist_z"):
        axis = "xyz".index(method[-1])
        field = v_gpu[:, axis] - v_gpu[:, axis].mean()

    elif method in ("dist_x_axis", "dist_y_axis", "dist_z_axis"):
        # Distance to the axis (parallel to x, y or z) passing through the area-weighted center.
        axis = "xyz".index(method[5])
        others = [a for a in range(3) if a != axis]
        center = to_gpu(_area_weighted_center(v_cpu, faces))
        field = xp.linalg.norm((v_gpu - center)[:, others], axis=1)

    elif method == "dist_centroid":
        # Area-weighted center: the plain vertex mean is biased by vertex density.
        center = to_gpu(_area_weighted_center(v_cpu, faces))
        field = xp.linalg.norm(v_gpu - center, axis=1)

    elif method in ("geodesic", "mass_center_geodesic"):
        # kwargs: vertex_ref_index (int | list | 'mass_center'), geodesic_solver ('heat' | 'dijkstra')
        spec = ['mass_center'] if method == "mass_center_geodesic" else kwargs.get("vertex_ref_index", [0])
        refs = _resolve_vertex_indices(spec, v_cpu, faces, default=[0])
        field = compute_geodesic_distance(v_cpu, faces, refs, solver=kwargs.get("geodesic_solver", "heat"))

    elif method in ("mean_curvature", "gaussian_curvature", "shape_index", "curvedness"):
        field = compute_curvature_field(v_cpu, faces, method)

    elif method == "normal_displacement":
        if prev_vertices is None or p2p is None:
            warnings.warn("normal_displacement requested without prev_vertices/p2p (first frame?); returning zeros.")
            field = np.zeros(num_vertices)
        else:
            field = compute_normal_displacement(v_cpu, faces, prev_vertices, p2p, signed=kwargs.get("signed", True))

    elif method.startswith("lb_eigen_"):
        evals, evecs = _spectral_basis(trimesh_obj)
        idx = int(method.split("_")[-1])
        if idx >= evecs.shape[1]:
            raise ValueError(f"{method}: only {evecs.shape[1]} eigenfunctions were computed (mesh.process(k=...)).")
        if idx == 0:
            warnings.warn("lb_eigen_0 is the constant eigenfunction; the Reeb graph will be trivial. Use idx >= 1.")
        # Eigenvector sign is arbitrary between frames; the Reeb graph is invariant under f -> -f,
        # but we fix the sign (positive at the vertex of largest |value|) so bins are comparable.
        field = evecs[:, idx].copy()
        field *= np.sign(field[np.argmax(np.abs(field))]) or 1.0

    elif method == "heat_diffusion":
        # kwargs: source_idx (int | list | 'mass_center'), t (float | 'auto')
        src = _resolve_vertex_indices(kwargs.get("source_idx", 0), v_cpu, faces, default=[0])
        field = compute_heat_diffusion(trimesh_obj, src, t=kwargs.get("t", "auto"))

    elif method == "matern_kernel":
        # kwargs: source_idx, nu, lengthscale
        src = _resolve_vertex_indices(kwargs.get("source_idx", 0), v_cpu, faces, default=[0])
        field = compute_matern_field(trimesh_obj, src, nu=kwargs.get("nu", 1.5), lengthscale=kwargs.get("lengthscale", 1.0))

    elif method == "harmonic":
        # kwargs: source_idx / sink_idx (int | list | 'mass_center'); defaults: lowest / highest z
        if trimesh_obj is None or getattr(trimesh_obj, "W", None) is None:
            raise ValueError("trimesh_obj with stiffness matrix W is required for harmonic fields.")
        src = _resolve_vertex_indices(kwargs.get("source_idx"), v_cpu, faces, default=[int(np.argmin(v_cpu[:, 2]))])
        snk = _resolve_vertex_indices(kwargs.get("sink_idx"), v_cpu, faces, default=[int(np.argmax(v_cpu[:, 2]))])
        field = compute_harmonic_field(trimesh_obj, src, snk)

    elif method == "multi_pca":
        # kwargs: fields (list of method names). Each field is robust-standardized before PCA.
        fields = kwargs.get("fields", ["z", "mean_curvature", "gaussian_curvature"])
        sub_kwargs = {k: val for k, val in kwargs.items() if k not in ("fields", "equalize_histogram")}
        stacked = []
        for name in fields:
            val = get_scalar_field(v_cpu, faces, method=name, prev_vertices=prev_vertices, p2p=p2p,
                                   trimesh_obj=trimesh_obj, equalize_histogram=False, **sub_kwargs)
            med = np.median(val)
            mad = np.median(np.abs(val - med)) * 1.4826
            stacked.append((val - med) / (mad if mad > 1e-12 else (np.std(val) + 1e-8)))
        pca = PCA(n_components=1)
        field = pca.fit_transform(np.vstack(stacked).T).ravel()
        field *= np.sign(field[np.argmax(np.abs(field))]) or 1.0      # deterministic sign

    else:
        raise ValueError(f"Unknown scalar field method: {method}")

    field = _sanitize_field(field, num_vertices, method)
    if kwargs.get("equalize_histogram", equalize_default):
        field = _rank_normalize(field)
    return field


# ----------------------------------------------------------------------------- #
#  Reeb graph
# ----------------------------------------------------------------------------- #

def _bin_scalar_field(sf, num_bins):
    """Uniform bins over [min, max]; robust to constant fields and floating point at the ends."""
    sf = np.asarray(sf, dtype=np.float64).ravel()
    f_min, f_max = sf.min(), sf.max()
    if not np.isfinite(f_min) or not np.isfinite(f_max):
        raise ValueError("Scalar field contains non-finite values; sanitize it before building the Reeb graph.")
    if f_max - f_min <= 0.0:
        return np.zeros(sf.shape[0], dtype=np.int64), np.array([f_min, f_min])
    bin_edges = np.linspace(f_min, f_max, num_bins + 1)
    bin_idx = np.floor((sf - f_min) / (f_max - f_min) * num_bins).astype(np.int64)
    return np.clip(bin_idx, 0, num_bins - 1), bin_edges


def compute_approx_reeb_graph(vertices, faces, scalar_field, num_bins=20):
    """
    Discrete Reeb graph of (mesh, scalar_field) via slab decomposition.

    The range of f is cut into `num_bins` slabs. For each triangle and each slab it
    crosses, the piece "triangle ∩ slab" is one cell (it is convex, hence connected).
    Cells of neighbouring triangles are merged when they share a segment of the common
    edge inside that slab; cells around a vertex are merged through the vertex. Connected
    groups of cells are the nodes (one per connected component of f^{-1}(slab)); two nodes
    are joined when a triangle spans both consecutive slabs.

    Compared with binning *vertices* only (previous implementation), this handles mesh
    edges that jump over several slabs correctly: they no longer create shortcut edges
    between non-adjacent slabs, which produced hundreds of spurious cycles and inflated
    Betti-1 as soon as num_bins exceeded the mesh resolution. Complexity is O(F·s) where
    s is the mean number of slabs a triangle spans (≈1 for a well-resolved mesh).

    Node attributes: pos (3,), bin (int), f_value (slab center), n_vertices, vertices (array).
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    sf = np.asarray(to_cpu(scalar_field), dtype=np.float64).ravel()
    if sf.shape[0] != v.shape[0]:
        raise ValueError("scalar_field must have one value per vertex.")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")

    # Drop degenerate triangles (repeated vertex index).
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    F = f.shape[0]
    graph = nx.Graph()
    if F == 0:
        return graph

    bin_idx, bin_edges = _bin_scalar_field(sf, num_bins)
    fb = bin_idx[f]                                     # (F, 3) slab of each corner
    fb_min, fb_max = fb.min(axis=1), fb.max(axis=1)
    span = fb_max - fb_min + 1                          # slabs crossed by each triangle
    cell_offset = np.concatenate([[0], np.cumsum(span)])
    n_cells = int(cell_offset[-1])

    def cell_id(face_ids, slabs):
        return cell_offset[face_ids] + (slabs - fb_min[face_ids])

    def expand(starts, counts):
        """For groups with given start values and counts, returns (group_index, start+offset)."""
        total = int(counts.sum())
        grp = np.repeat(np.arange(len(counts)), counts)
        first = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        return grp, np.repeat(starts, counts) + (np.arange(total) - first)

    rows, cols = [], []

    # 1) Merge cells of triangles that share an edge, in every slab the edge crosses.
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e_face = np.tile(np.arange(F), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, e_face = e[order], e_face[order]
    same = np.all(e[1:] == e[:-1], axis=1)              # consecutive rows = same undirected edge
    p = np.flatnonzero(same)
    if p.size:
        fa, fb_ = e_face[p], e_face[p + 1]
        lo = np.minimum(bin_idx[e[p, 0]], bin_idx[e[p, 1]])
        hi = np.maximum(bin_idx[e[p, 0]], bin_idx[e[p, 1]])
        grp, slab = expand(lo, hi - lo + 1)
        rows.append(cell_id(fa[grp], slab))
        cols.append(cell_id(fb_[grp], slab))

    # 2) Merge all cells around a vertex in the vertex's own slab (covers pinched vertices).
    corner_vertex = f.ravel()
    corner_face = np.repeat(np.arange(F), 3)
    corner_cell = cell_id(corner_face, bin_idx[corner_vertex])
    order = np.argsort(corner_vertex, kind="stable")
    cv, cc = corner_vertex[order], corner_cell[order]
    same_v = cv[1:] == cv[:-1]
    rows.append(cc[:-1][same_v])
    cols.append(cc[1:][same_v])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    adj = coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n_cells, n_cells))
    n_nodes, cell_label = connected_components(adj, directed=False)

    # Slab of every cell, hence of every node.
    _, cell_slab = expand(fb_min, span)
    node_slab = np.empty(n_nodes, dtype=np.int64)
    node_slab[cell_label] = cell_slab

    # Relabel nodes so ids increase with slab (deterministic, nicer for downstream plots).
    relabel = np.empty(n_nodes, dtype=np.int64)
    relabel[np.lexsort((np.arange(n_nodes), node_slab))] = np.arange(n_nodes)
    cell_label = relabel[cell_label]
    node_slab = node_slab[np.argsort(relabel)]

    # 3) Reeb edges: consecutive slabs inside the same triangle.
    grp, slab = expand(fb_min, span - 1)
    if grp.size:
        a = cell_label[cell_id(grp, slab)]
        b = cell_label[cell_id(grp, slab + 1)]
        reeb_edges = np.unique(np.sort(np.stack([a, b], axis=1), axis=1), axis=0)
    else:
        reeb_edges = np.empty((0, 2), dtype=np.int64)

    # 4) Node geometry: vertex -> node (well defined thanks to step 2).
    vertex_node = np.full(v.shape[0], -1, dtype=np.int64)
    vertex_node[corner_vertex] = cell_label[corner_cell]
    has_node = vertex_node >= 0
    pos_sum = np.zeros((n_nodes, 3))
    counts = np.zeros(n_nodes)
    np.add.at(pos_sum, vertex_node[has_node], v[has_node])
    np.add.at(counts, vertex_node[has_node], 1.0)

    # Nodes without vertices (slabs cutting through a triangle strictly between its corners):
    # place them on the iso-line f = slab center, interpolated along the triangle edges.
    empty = counts == 0
    if empty.any():
        f_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:]) if len(bin_edges) > 2 else np.array([bin_edges[0]])
        grp_c, slab_c = expand(fb_min, span)             # (cell -> face, cell -> slab)
        lab_c = cell_label[np.arange(n_cells)]
        sel = empty[lab_c]
        fc, sc, lc = grp_c[sel], slab_c[sel], lab_c[sel]
        target = f_mid[sc]
        for k in range(3):
            i0, i1 = f[fc, k], f[fc, (k + 1) % 3]
            f0, f1 = sf[i0], sf[i1]
            cross = ((f0 <= target) & (target <= f1)) | ((f1 <= target) & (target <= f0))
            cross &= np.abs(f1 - f0) > 1e-15
            tpar = np.where(cross, (target - f0) / np.where(cross, f1 - f0, 1.0), 0.0)
            pts = v[i0] + tpar[:, None] * (v[i1] - v[i0])
            np.add.at(pos_sum, lc[cross], pts[cross])
            np.add.at(counts, lc[cross], 1.0)
        still_empty = counts == 0                        # numerical corner case: use face centroids
        if still_empty.any():
            cen = v[f[fc]].mean(axis=1)
            m = still_empty[lc]
            np.add.at(pos_sum, lc[m], cen[m])
            np.add.at(counts, lc[m], 1.0)

    pos = pos_sum / counts[:, None]
    n_vertices = np.bincount(vertex_node[has_node], minlength=n_nodes)
    f_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) if len(bin_edges) > 2 else np.array([bin_edges[0]])
    vert_order = np.argsort(vertex_node[has_node], kind="stable")
    vert_sorted = np.flatnonzero(has_node)[vert_order]
    vert_split = np.split(vert_sorted, np.cumsum(n_vertices)[:-1])

    for node in range(n_nodes):
        graph.add_node(node, pos=pos[node], bin=int(node_slab[node]), f_value=float(f_centers[node_slab[node]]),
                       n_vertices=int(n_vertices[node]), vertices=vert_split[node])
    graph.add_edges_from(map(tuple, reeb_edges.tolist()))
    return graph


def create_reeb_polydata(graph):
    if len(graph.nodes) == 0:
        return pv.PolyData()

    nodes = list(graph.nodes(data=True))
    node_map = {n: i for i, (n, data) in enumerate(nodes)}
    pts = np.array([data['pos'] for n, data in nodes])

    lines = []
    for u, v in graph.edges():
        lines.extend([2, node_map[u], node_map[v]])

    mesh = pv.PolyData(pts)
    if lines:
        mesh.lines = np.array(lines)
    # Node attributes for colouring in the viewers (legacy / manually added nodes -> bin / 0).
    mesh.point_data['f_value'] = np.array([float(data.get('f_value', data.get('bin', np.nan))) for _, data in nodes])
    mesh.point_data['bin'] = np.array([int(data.get('bin', 0)) for _, data in nodes])
    mesh.point_data['n_vertices'] = np.array([int(data.get('n_vertices', 0)) for _, data in nodes])
    mesh.point_data['degree'] = np.array([graph.degree(n) for n, _ in nodes])
    return mesh