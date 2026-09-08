import trimesh
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyFM.mesh import TriMesh
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from scipy.sparse import coo_matrix, diags

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[INFO] CuPy detected. Utilizing GPU for Basic Geometry.")
except ImportError:
    CUPY_AVAILABLE = False
    print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")


# Metric groups -> CSV columns they produce
METRIC_COLUMNS = {
    'n_vertices': ['n_vertices'],
    'n_faces': ['n_faces'],
    'area': ['area'],
    'volume': ['volume', 'is_watertight'],
    'sphericity': ['sphericity'],
    'convexity': ['convexity'],
    'center_mass': ['cm_x', 'cm_y', 'cm_z'],
    'gaussian_curvature': ['mean_gaussian_curvature', 'mean_abs_gaussian_curvature', 'total_abs_gaussian_curvature', 'gaussian_curvature_std'],
    'mean_curvature': ['mean_mean_curvature', 'mean_abs_mean_curvature', 'willmore_energy'],
    'topology': ['euler_number', 'genus', 'n_boundary_edges', 'n_components'],
    'bounding_box': ['bbox_dx', 'bbox_dy', 'bbox_dz', 'bbox_diagonal'],
    'shape': ['elongation', 'flatness', 'surface_to_volume', 'radius_of_gyration'],
}
AVAILABLE_METRICS = set(METRIC_COLUMNS)


# ----------------------------------------------------------------------------- #
#  Geometry helpers (numpy only)
# ----------------------------------------------------------------------------- #

def _faces(mesh):
    f = np.asarray(mesh.faces)
    if f.ndim == 1:                                     # pyvista padded format
        f = f.reshape(-1, 4)[:, 1:]
    return f.astype(np.int64)


def face_areas(v, f):
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def vertex_areas(v, f):
    fa = face_areas(v, f)
    va = np.zeros(v.shape[0])
    for k in range(3):
        np.add.at(va, f[:, k], fa / 3.0)
    return va


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def signed_volume(v, f):
    """Enclosed volume by the divergence theorem (exact for closed, consistently oriented meshes)."""
    return float(np.einsum('ij,ij->i', v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0)


def gaussian_curvature(v, f):
    """Discrete Gaussian curvature K_i = (2π - Σ angles_i) / A_i (angle deficit per vertex area)."""
    angles_sum = np.zeros(v.shape[0])
    for k in range(3):
        p0, p1, p2 = v[f[:, k]], v[f[:, (k + 1) % 3]], v[f[:, (k + 2) % 3]]
        a, b = p1 - p0, p2 - p0
        cosang = np.einsum('ij,ij->i', a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-16)
        np.add.at(angles_sum, f[:, k], np.arccos(np.clip(cosang, -1.0, 1.0)))
    va = vertex_areas(v, f)
    used = va > 0
    K = np.zeros(v.shape[0])
    K[used] = (2.0 * np.pi - angles_sum[used]) / va[used]
    return K, va, used


def mean_curvature(v, f):
    """Signed discrete mean curvature H_i = (W x)_i · n_i / (2 A_i) with the cotangent stiffness W."""
    n = v.shape[0]
    I, J, Wv = [], [], []
    for k in range(3):
        i, j, l = f[:, k], f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        a, b = v[i] - v[l], v[j] - v[l]
        cot = np.einsum('ij,ij->i', a, b) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-16)
        I += [i, j]; J += [j, i]; Wv += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(Wv), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr()
    W = diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm
    va = vertex_areas(v, f)
    used = va > 0
    H = np.zeros(n)
    H[used] = np.einsum('ij,ij->i', (W @ v)[used], vertex_normals(v, f)[used]) / (2.0 * va[used])
    return H, va, used


def boundary_edge_count(f):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def to_trimesh(v, f):
    """
    Trimesh WITHOUT vertex merging. The default `process=True` welds coincident vertices; when two
    parts of the shape touch (legs, belly, ...) this creates non-manifold edges, `is_watertight`
    becomes False and every volume-based metric was lost, although the original mesh was closed.
    Only the face orientation is made consistent (outward normals).
    """
    m = trimesh.Trimesh(vertices=v, faces=f, process=False, validate=False)
    if not m.is_winding_consistent:
        trimesh.repair.fix_winding(m)
    if m.is_watertight:
        trimesh.repair.fix_inversion(m)
    return m


# ----------------------------------------------------------------------------- #
#  Per-mesh metrics
# ----------------------------------------------------------------------------- #

def compute_mesh_geometry(pyfm_mesh, metrics='all'):
    """
    Computes geometric properties of a single pyFM TriMesh.

    :param pyfm_mesh: A pyFM.mesh.TriMesh object (or any object with .vertices / .faces).
    :param metrics: 'all', a single string, or a list of strings among
        'n_vertices', 'n_faces', 'area', 'volume', 'sphericity', 'convexity', 'center_mass',
        'gaussian_curvature', 'mean_curvature', 'topology', 'bounding_box', 'shape'.
    :return: Dictionary of metrics (see METRIC_COLUMNS for the columns each metric produces).

    Volume-based metrics are always computed with the divergence theorem on the original faces;
    the boolean `is_watertight` tells whether the mesh is closed and manifold (exact volume) or not
    (volume is then the approximation for the almost closed surface).
    """
    if isinstance(metrics, str):
        metrics_to_compute = set(AVAILABLE_METRICS) if metrics.lower() == 'all' else {metrics.lower()}
    else:
        metrics_to_compute = {str(m).lower() for m in metrics}
    unknown = metrics_to_compute - AVAILABLE_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}. Available: {sorted(AVAILABLE_METRICS)}")

    v = np.asarray(pyfm_mesh.vertices, dtype=np.float64)
    f = _faces(pyfm_mesh)
    results = {}

    if 'n_vertices' in metrics_to_compute:
        results['n_vertices'] = int(v.shape[0])
    if 'n_faces' in metrics_to_compute:
        results['n_faces'] = int(f.shape[0])

    fa = face_areas(v, f)
    area = float(fa.sum())
    if 'area' in metrics_to_compute:
        results['area'] = area

    needs_volume = metrics_to_compute & {'volume', 'sphericity', 'convexity', 'center_mass', 'shape'}
    vol, watertight, t_mesh = np.nan, False, None
    if needs_volume:
        t_mesh = to_trimesh(v, f)
        watertight = bool(t_mesh.is_watertight)
        vol = abs(signed_volume(np.asarray(t_mesh.vertices), np.asarray(t_mesh.faces)))

    if 'volume' in metrics_to_compute:
        results['volume'] = vol
        results['is_watertight'] = watertight

    if 'sphericity' in metrics_to_compute:
        results['sphericity'] = (np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / area if (vol > 0 and area > 0) else np.nan

    if 'convexity' in metrics_to_compute:
        try:
            hull_vol = abs(float(t_mesh.convex_hull.volume))
            results['convexity'] = vol / hull_vol if hull_vol > 0 else np.nan
        except Exception:  # noqa: BLE001 - degenerate hull
            results['convexity'] = np.nan

    if 'center_mass' in metrics_to_compute:
        # Volume-based center of mass for closed meshes, area-weighted surface centroid otherwise
        # (trimesh.center_mass is meaningless on open surfaces).
        if watertight:
            cm = np.asarray(t_mesh.center_mass, dtype=np.float64)
        else:
            centroids = v[f].mean(axis=1)
            cm = (centroids * fa[:, None]).sum(axis=0) / max(area, 1e-16)
        results['cm_x'], results['cm_y'], results['cm_z'] = float(cm[0]), float(cm[1]), float(cm[2])

    if 'gaussian_curvature' in metrics_to_compute:
        K, va, used = gaussian_curvature(v, f)
        w = va[used] / va[used].sum()
        results['mean_gaussian_curvature'] = float(np.sum(w * K[used]))          # = 2πχ / A for closed meshes (Gauss-Bonnet)
        results['mean_abs_gaussian_curvature'] = float(np.sum(w * np.abs(K[used])))
        results['total_abs_gaussian_curvature'] = float(np.sum(va[used] * np.abs(K[used])))   # bending complexity, scale invariant
        results['gaussian_curvature_std'] = float(np.sqrt(np.sum(w * (K[used] - results['mean_gaussian_curvature']) ** 2)))

    if 'mean_curvature' in metrics_to_compute:
        H, va, used = mean_curvature(v, f)
        w = va[used] / va[used].sum()
        results['mean_mean_curvature'] = float(np.sum(w * H[used]))
        results['mean_abs_mean_curvature'] = float(np.sum(w * np.abs(H[used])))
        results['willmore_energy'] = float(np.sum(va[used] * H[used] ** 2))      # ∫H² dA, scale invariant (4π for a sphere)

    if 'topology' in metrics_to_compute:
        n_boundary, n_nonmanifold = boundary_edge_count(f)
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        n_edges = np.unique(e, axis=0).shape[0]
        euler = int(v.shape[0] - n_edges + f.shape[0])
        from scipy.sparse.csgraph import connected_components
        adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(v.shape[0], v.shape[0]))
        n_comp = int(connected_components(adj, directed=False)[0])
        results['euler_number'] = euler
        results['genus'] = int(round((2 * n_comp - euler) / 2)) if n_boundary == 0 else np.nan
        results['n_boundary_edges'] = n_boundary
        results['n_components'] = n_comp
        if n_nonmanifold:
            results['n_nonmanifold_edges'] = n_nonmanifold

    if 'bounding_box' in metrics_to_compute:
        ext = np.ptp(v, axis=0)
        results['bbox_dx'], results['bbox_dy'], results['bbox_dz'] = map(float, ext)
        results['bbox_diagonal'] = float(np.linalg.norm(ext))

    if 'shape' in metrics_to_compute:
        # Area-weighted PCA of the surface: principal extents (std along principal axes).
        centroids = v[f].mean(axis=1)
        w = fa / max(area, 1e-16)
        c = (centroids * w[:, None]).sum(axis=0)
        cov = ((centroids - c) * w[:, None]).T @ (centroids - c)
        ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        s = np.sqrt(ev)
        results['elongation'] = float(s[0] / s[1]) if s[1] > 0 else np.nan     # 1 = isotropic in the two largest axes
        results['flatness'] = float(s[1] / s[2]) if s[2] > 0 else np.nan
        results['surface_to_volume'] = float(area / vol) if vol > 0 else np.nan
        results['radius_of_gyration'] = float(np.sqrt(ev.sum()))

    return results


# ----------------------------------------------------------------------------- #
#  Plotting
# ----------------------------------------------------------------------------- #

def generate_plots_from_csv(csv_path, dpi=200):
    """
    Reads the CSV, computes consecutive pairwise distances for the Center of Mass,
    and plots all properties inside a single integrated dashboard figure with
    independent, optimized y-axis scaling. Frames whose mesh is not watertight are
    marked with a red cross on the volume-based curves.
    """
    csv_path = Path(csv_path)
    plot_path = csv_path.parent
    os.makedirs(plot_path, exist_ok=True)

    df = pd.read_csv(csv_path)
    x = np.arange(len(df))

    cm_cols = {'cm_x', 'cm_y', 'cm_z'}
    flag_cols = {'is_watertight'}
    volume_based = {'volume', 'sphericity', 'convexity', 'surface_to_volume'}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scalar_metrics = [c for c in numeric_cols if c not in cm_cols | flag_cols and not df[c].isna().all()]

    not_watertight = None
    if 'is_watertight' in df.columns:
        flags = df['is_watertight'].astype(str).str.lower().isin(['false', '0', '0.0'])
        not_watertight = np.flatnonzero(flags.values)

    has_cm_displacement = False
    distances = None
    if cm_cols.issubset(df.columns):
        coords = df[['cm_x', 'cm_y', 'cm_z']].values
        if len(coords) >= 2:
            has_cm_displacement = True
            if CUPY_AVAILABLE:
                gpu_coords = cp.asarray(coords)
                distances = cp.asnumpy(cp.linalg.norm(cp.diff(gpu_coords, axis=0), axis=1))
            else:
                distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)

    num_plots = len(scalar_metrics) + (1 if has_cm_displacement else 0)
    if num_plots == 0:
        print("[INFO] No columns found to plot.")
        return

    cols = 2 if num_plots > 1 else 1
    rows = int(np.ceil(num_plots / cols))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes = axes.flatten()

    def pad_flat(ax, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        y_min, y_max = values.min(), values.max()
        if np.isclose(y_min, y_max):
            ax.set_ylim(-1, 1) if y_min == 0 else ax.set_ylim(min(y_min * 0.9, y_min * 1.1), max(y_min * 0.9, y_min * 1.1))

    plot_idx = 0
    for metric in scalar_metrics:
        ax = axes[plot_idx]
        ax.plot(x, df[metric], marker='o', linestyle='-', color='b')
        if metric in volume_based and not_watertight is not None and not_watertight.size:
            ax.scatter(x[not_watertight], df[metric].values[not_watertight], marker='x', s=90, color='red', zorder=3,
                       label='not watertight (approx.)')
            ax.legend(fontsize=8)
        ax.set_title(f'Evolution of {metric.replace("_", " ").title()}')
        ax.set_xlabel('Mesh Sequence Index')
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.1)
        pad_flat(ax, df[metric])
        plot_idx += 1

    if has_cm_displacement:
        ax = axes[plot_idx]
        ax.plot(np.arange(len(distances)), distances, marker='s', linestyle='-', color='b')
        ax.set_title('Center of Mass Consecutive Displacement')
        ax.set_xlabel('Sequence Transition Interval (i to i+1)')
        ax.set_ylabel('Euclidean Distance')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.1)
        pad_flat(ax, distances)
        plot_idx += 1

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fig.savefig(plot_path / 'mesh_evolution_summary.png', dpi=dpi)
    plt.close(fig)