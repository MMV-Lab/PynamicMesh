import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt


try:
    import cupy as xp
    GPU_AVAILABLE = True
    print("[INFO] CuPy detected. Utilizing GPU for Spectral computations.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")


def to_cpu(arr):
    """Safely moves a CuPy array/scalar back to the CPU."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


def _frame_number(path):
    """Integer frame id from names like 'Reeb_T0012.pkl' / 'Reeb_T12.pkl' (name order as fallback)."""
    m = re.search(r'T(\d+)', Path(path).stem)
    return int(m.group(1)) if m else float('inf')


def node_function_value(data):
    """
    Reeb function value of a node.

    Nodes produced by compute_approx_reeb_graph store 'f_value' (center of the slab in the
    units of the scalar field). Older pickles only have 'bin'. The z coordinate of 'pos' is
    used as a last resort only: it equals the Reeb function solely when reeb_scalar == 'z'.
    """
    if 'f_value' in data:
        return float(data['f_value'])
    if 'bin' in data:
        return float(data['bin'])
    if 'pos' in data and len(data['pos']) >= 3:
        return float(data['pos'][2])
    return 0.0


def graph_similarity(reeb_folder_path, metrics_list, single_file=True):
    """
    Computes pairwise metrics between consecutive Reeb graphs of a sequence.

    NOTE: 'interleaving_distance', 'labeled_interleaving_distance', 'function_distortion_distance'
    and 'branch_decomposition_distance' are cheap *proxies* (sorted-value / sorted-hop-distance
    L-infinity comparisons and feature-vector norms), not the exact TDA distances of the same name.
    'labeled_interleaving_distance' coincides with 'interleaving_distance' unless nodes carry a
    'label' attribute.

    Args:
        reeb_folder_path (str/Path): folder containing the .pkl Reeb graphs.
        metrics_list (list of str | 'all'): requested metrics.
    Returns:
        str: path to the generated CSV file (None if fewer than two graphs are found).
    """
    reeb_folder = Path(reeb_folder_path)
    tag = 'Reeb' if 'reeb' in str(reeb_folder).lower() else reeb_folder.name

    if single_file:
        print("Starting metric similarity among graphs")

    if isinstance(metrics_list, str):
        metrics_list = [
            'degree_wasserstein',
            'spectral_laplacian',
            'interleaving_distance',
            'labeled_interleaving_distance',
            'function_distortion_distance',
            'branch_decomposition_distance'
        ]

    # Sort on the frame number: a plain string sort puts Reeb_T10 before Reeb_T2 (un-padded files).
    reeb_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.suffix == '.pkl'],
                        key=_frame_number)
    frame_ids = [_frame_number(f) for f in reeb_files]

    if len(reeb_files) < 2:
        print("Error: Not enough Reeb graphs found to perform pairwise similarity analysis.")
        return None

    analysis_folder = reeb_folder.parent / 'Graph_analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    with open(reeb_files[0], 'rb') as f:
        G_prev = pickle.load(f)

    results = []

    def calc_degree_wasserstein(g1, g2):
        v1, v2 = g1.number_of_nodes(), g2.number_of_nodes()
        deg1 = [d for n, d in g1.degree()] if v1 > 0 else [0]
        deg2 = [d for n, d in g2.degree()] if v2 > 0 else [0]
        return wasserstein_distance(deg1, deg2)

    def calc_spectral_distance(g1, g2):
        """Dense spectral distance between normalized Laplacians (spectra zero-padded to equal length)."""
        v1, v2 = g1.number_of_nodes(), g2.number_of_nodes()
        if v1 == 0 or v2 == 0:
            return 0.0

        lap_prev_np = np.asarray(nx.normalized_laplacian_matrix(g1).todense())
        lap_curr_np = np.asarray(nx.normalized_laplacian_matrix(g2).todense())

        evals_prev = xp.linalg.eigvalsh(xp.asarray(lap_prev_np))
        evals_curr = xp.linalg.eigvalsh(xp.asarray(lap_curr_np))

        max_len = max(len(evals_prev), len(evals_curr))
        e_p_pad = xp.pad(evals_prev, (0, max_len - len(evals_prev)))
        e_c_pad = xp.pad(evals_curr, (0, max_len - len(evals_curr)))

        return float(to_cpu(xp.linalg.norm(e_p_pad - e_c_pad)))

    def extract_scalars(G):
        if G.number_of_nodes() == 0:
            return np.array([0.0])
        return np.array([node_function_value(data) for _, data in G.nodes(data=True)])

    def _sorted_linf(a, b):
        a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 0.0
        a = np.pad(a, (0, max_len - len(a)), mode='edge')
        b = np.pad(b, (0, max_len - len(b)), mode='edge')
        return float(np.max(np.abs(a - b)))

    def calc_interleaving_proxy(g1, g2):
        return _sorted_linf(extract_scalars(g1), extract_scalars(g2))

    def calc_function_distortion_proxy(g1, g2):
        if g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
            return np.nan
        try:
            d1 = [d for _, dists in nx.all_pairs_shortest_path_length(g1) for d in dists.values()]
            d2 = [d for _, dists in nx.all_pairs_shortest_path_length(g2) for d in dists.values()]
            return _sorted_linf(d1, d2)
        except Exception:
            return np.nan

    def calc_branch_decomposition_proxy(g1, g2):
        def branch_features(g):
            v, e = g.number_of_nodes(), g.number_of_edges()
            c = nx.number_connected_components(g) if v > 0 else 0
            betti = max(0, e - v + c)
            branch_nodes = sum(1 for n, d in g.degree() if d > 2)
            return np.array([betti, branch_nodes])

        return float(np.linalg.norm(branch_features(g1) - branch_features(g2)))

    def calc_labeled_interleaving_proxy(g1, g2):
        def extract_grouped_scalars(G):
            groups = {}
            for n, data in G.nodes(data=True):
                label = str(data.get('label', 'unlabeled'))
                groups.setdefault(label, []).append(node_function_value(data))
            return groups

        g1_groups = extract_grouped_scalars(g1)
        g2_groups = extract_grouped_scalars(g2)

        all_labels = set(g1_groups) | set(g2_groups)
        if not all_labels:
            return 0.0

        return max(_sorted_linf(g1_groups.get(label, [0.0]), g2_groups.get(label, [0.0]))
                   for label in all_labels)

    metric_dispatch = {
        'degree_wasserstein': calc_degree_wasserstein,
        'spectral_laplacian': calc_spectral_distance,
        'interleaving_distance': calc_interleaving_proxy,
        'labeled_interleaving_distance': calc_labeled_interleaving_proxy,
        'function_distortion_distance': calc_function_distortion_proxy,
        'branch_decomposition_distance': calc_branch_decomposition_proxy,
    }

    for i in tqdm(range(1, len(reeb_files)), desc="Computing Fast Pairwise Similarities", leave=single_file):
        with open(reeb_files[i], 'rb') as f:
            G_curr = pickle.load(f)

        row_data = {
            'Transition': f"T{frame_ids[i-1]} -> T{frame_ids[i]}",
            'Time_Step': frame_ids[i]
        }

        for metric in metrics_list:
            metric_clean = metric.lower().strip()
            if metric_clean in metric_dispatch:
                val = metric_dispatch[metric_clean](G_prev, G_curr)
            else:
                print(f"[Warning] Unknown metric '{metric}'. Skipping.")
                val = np.nan
            row_data[metric] = val

        results.append(row_data)
        G_prev = G_curr

    df = pd.DataFrame(results)
    csv_out_path = analysis_folder / f'{tag}_pairwise_graph_similarity.csv'
    df.to_csv(csv_out_path, index=False)

    if single_file:
        print(f"Similarity analysis complete. Data saved to: {csv_out_path}")

    return str(csv_out_path)


def plot_graph_similarity(csv_path, single_file=True):
    """
    Reads the pairwise similarity CSV and generates a plot tracking the evolution
    of the chosen metrics over time. Each subplot has its own x-axis label.
    """
    if csv_path is None:
        return
    csv_file = Path(csv_path)

    tag = str(csv_file.name).split('_')[0]

    if single_file:
        print("\nGenerating visual reports for Graph Similarity...")

    if not csv_file.exists():
        print(f"Error: Could not find CSV file at {csv_path}")
        return

    df = pd.read_csv(csv_file)
    plots_folder = csv_file.parent / 'plots'
    os.makedirs(plots_folder, exist_ok=True)

    time_steps = df['Time_Step']

    exclude_cols = ['Transition', 'Time_Step']
    valid_metrics = [col for col in df.columns if col not in exclude_cols and not df[col].isna().all()]

    if not valid_metrics:
        print("No valid metric data found in the CSV to plot.")
        return

    sns.set_theme(style="whitegrid")
    num_metrics = len(valid_metrics)

    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics), sharex=False)

    if num_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("husl", num_metrics)

    for ax, metric, color in zip(axes, valid_metrics, colors):
        ax.plot(time_steps, df[metric], color=color, marker='D', linewidth=2, markersize=6)

        formatted_title = metric.replace('_', ' ').title()
        ax.set_title(f'Evolution of {formatted_title}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance / Shift', fontsize=10)
        ax.set_xlabel('Time Step (Transition $T_{n-1} \\rightarrow T_n$)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()

    output_img_path = plots_folder / f'{tag}_Pairwise_Similarity_Evolution.png'
    fig.savefig(output_img_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    if single_file:
        print(f"Visual report generated successfully in: {output_img_path}")