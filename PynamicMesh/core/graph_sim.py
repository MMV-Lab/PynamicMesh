import os
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

def graph_similarity(reeb_folder_path, metrics_list, single_file=True):

    """
    Computes pairwise metrics from a list of Reeb graphs across sequential time steps.
    Includes custom proxy solvers for advanced TDA metrics and GPU-accelerated spectral distance.
    
    Args:
        reeb_folder_path (str/Path): Path to the folder containing the computed .pkl Reeb graphs.
        metrics_list (list of str): List of requested metrics.
    Returns:
        str: Path to the generated CSV file.
    """
    if 'Reeb' in str(reeb_folder_path):
        tag = 'Reeb'

    if single_file:
        print("Starting metric similarity among graphs")
    
    if isinstance(metrics_list,str):
        metrics_list = [
            'degree_wasserstein',
            'spectral_laplacian',
            'interleaving_distance',
            'labeled_interleaving_distance', 
            'function_distortion_distance',
            'branch_decomposition_distance'
        ]
    
    reeb_folder = Path(reeb_folder_path)
    reeb_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.suffix == '.pkl'])
    
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
        """GPU-Accelerated Dense Spectral Laplacian distance."""
        v1, v2 = g1.number_of_nodes(), g2.number_of_nodes()
        if v1 == 0 or v2 == 0:
            return 0.0
            
        lap_prev_np = np.asarray(nx.normalized_laplacian_matrix(g1).todense())
        lap_curr_np = np.asarray(nx.normalized_laplacian_matrix(g2).todense())
        

        lap_prev_xp = xp.asarray(lap_prev_np)
        lap_curr_xp = xp.asarray(lap_curr_np)
        

        evals_prev = xp.linalg.eigvalsh(lap_prev_xp)
        evals_curr = xp.linalg.eigvalsh(lap_curr_xp)
        
        max_len = max(len(evals_prev), len(evals_curr))
        e_p_pad = xp.pad(evals_prev, (0, max_len - len(evals_prev)))
        e_c_pad = xp.pad(evals_curr, (0, max_len - len(evals_curr)))
        

        return float(to_cpu(xp.linalg.norm(e_p_pad - e_c_pad)))

    def extract_scalars(G):
        if G.number_of_nodes() == 0: return np.array([0.0])
        scalars = []
        for n, data in G.nodes(data=True):
            if 'pos' in data and len(data['pos']) >= 3:
                scalars.append(data['pos'][2]) 
            elif 'bin' in data:
                scalars.append(float(data['bin']))
            else:
                scalars.append(0.0)
        return np.array(scalars)

    def calc_interleaving_proxy(g1, g2):
        s1 = np.sort(extract_scalars(g1))
        s2 = np.sort(extract_scalars(g2))
        
        max_len = max(len(s1), len(s2))
        if max_len == 0: return 0.0
        
        s1_pad = np.pad(s1, (0, max_len - len(s1)), mode='edge')
        s2_pad = np.pad(s2, (0, max_len - len(s2)), mode='edge')
        return float(np.max(np.abs(s1_pad - s2_pad)))

    def calc_function_distortion_proxy(g1, g2):
        if g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
            return np.nan
        try:

            d1 = np.sort([d for _, dists in nx.all_pairs_shortest_path_length(g1) for d in dists.values()])
            d2 = np.sort([d for _, dists in nx.all_pairs_shortest_path_length(g2) for d in dists.values()])
            
            max_len = max(len(d1), len(d2))
            d1_pad = np.pad(d1, (0, max_len - len(d1)), mode='edge')
            d2_pad = np.pad(d2, (0, max_len - len(d2)), mode='edge')
            return float(np.max(np.abs(d1_pad - d2_pad)))
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
            if G.number_of_nodes() == 0: return groups
            
            for n, data in G.nodes(data=True):
                # Look for a label, default to 'unlabeled' if none exists
                label = str(data.get('label', 'unlabeled'))
                
                # Extract the scalar value just like the original function
                if 'pos' in data and len(data['pos']) >= 3:
                    val = data['pos'][2]
                elif 'bin' in data:
                    val = float(data['bin'])
                else:
                    val = 0.0
                    
                if label not in groups:
                    groups[label] = []
                groups[label].append(val)
            return groups

        g1_groups = extract_grouped_scalars(g1)
        g2_groups = extract_grouped_scalars(g2)
        
        all_labels = set(g1_groups.keys()).union(set(g2_groups.keys()))
        if not all_labels: return 0.0
        
        max_dist = 0.0
        # Compare distances strictly within matching label groups
        for label in all_labels:
            s1 = np.sort(g1_groups.get(label, [0.0]))
            s2 = np.sort(g2_groups.get(label, [0.0]))
            
            max_len = max(len(s1), len(s2))
            if max_len == 0: continue
            
            s1_pad = np.pad(s1, (0, max_len - len(s1)), mode='edge')
            s2_pad = np.pad(s2, (0, max_len - len(s2)), mode='edge')
            
            dist = float(np.max(np.abs(s1_pad - s2_pad)))
            max_dist = max(max_dist, dist) 
            
        return max_dist


    metric_dispatch = {
        'degree_wasserstein': calc_degree_wasserstein,
        'spectral_laplacian': calc_spectral_distance,
        'interleaving_distance': calc_interleaving_proxy,
        'labeled_interleaving_distance': calc_labeled_interleaving_proxy, 
        'function_distortion_distance': calc_function_distortion_proxy,
        'branch_decomposition_distance': calc_branch_decomposition_proxy,
    }

    for i in tqdm(range(1, len(reeb_files)), desc="Computing Fast Pairwise Similarities",leave=single_file):
        with open(reeb_files[i], 'rb') as f:
            G_curr = pickle.load(f)
            
        row_data = {
            'Transition': f"T{i-1} -> T{i}",
            'Time_Step': i
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