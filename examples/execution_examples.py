from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.core.reeb_graph import graph_time_analysis, plot_dynamic_graph_analysis  
from pathlib import Path
from PynamicMesh.core.graph_sim import graph_similarity, plot_graph_similarity
from PynamicMesh.utils.visualizers import (
    visualize_reeb_graphs, 
    edit_graph,
    visualize_physics,
    visual_selection_edition, 
    precompute_landmarks,
    visualize_obj_sequence
)

####################################################################################################### Paths reference list ##########################################################################################################################################

####################################################################################################### Linux #########################################################################################################################################################
base_mesh_path = '/PynamicMesh/Mesh_models'
mesh_path = '/PynamicMesh/Mesh_models/scene1'
matrix_path = '/PynamicMesh/Results/scene1/Transform_Matrices'
reeb_path = '/PynamicMesh/Results/scene1/Reeb_Graphs'
csv_file_path = '/PynamicMesh/Results/scene1/Graph_analysis/time_analysis.csv'
mesh_objs_folder = '/Results/scene1/active_surface_lab/01_Passive_Relaxation'

###################################################################################################### Windows #########################################################################################################################################################
base_mesh_path = r'\PynamicMesh\Mesh_models'
mesh_path = r'\PynamicMesh\Mesh_models\scene1'
matrix_path = r'\PynamicMesh\Results\scene1\Transform_Matrices'
reeb_path = r'\PynamicMesh\Results\scene1\Reeb_Graphs'
csv_file_path = r'\Results\scene1\Graph_analysis\time_analysis.csv'
mesh_objs_folder = r'\Results\scene1\active_surface_lab\01_Passive_Relaxation'

###########################################################################################################################################################################################################################################################################

####################################################################################################### Precomputing Visual helpers ####################################################################################################################################

###################################################################################################### Landmarks Precompute Visual Launcher ############################################################################################################################
print('Visualizing or editing landmarks for FM...')
visual_selection_edition(mesh_path,'FM')

print('Precomputing landmarks for FM...')
precompute_landmarks(base_mesh_path,'FM') 

###################################################################################################### Vertex index reference Precompute Visual Launcer for 'geodesic' in Reebs ########################################################################################
print('Visualizing or editing Vertex index for RG...')
visual_selection_edition(mesh_path,'geodesic')

print('Precomputing Vertex index for RG...')
precompute_landmarks(base_mesh_path,'geodesic') 

###################################################################################################### Sources Vertex index  Precompute Visual Launcher for 'heat_diffusion' in Reebs ###################################################################################
print('Visualizing or editing Sources Vertex index for RG...')
visual_selection_edition(mesh_path,'heat_diffusion')

print('Precomputing Sources Vertex index for RG...')
precompute_landmarks(base_mesh_path,'heat_diffusion')

######################################################################################################  Source-sink Vertex index  Precompute Visual Launcher for 'harmonic' in Reebs ###################################################################################
print('Visualizing or editing Source-sink Vertex index index for RG...')
visual_selection_edition(mesh_path,'harmonic')

print('Precomputing Source-sink Vertex index for RG...')
precompute_landmarks(base_mesh_path,'harmonic')

#########################################################################################################################################################################################################################################################################


####################################################################################################### Runing Full pipeline over folder system ##########################################################################################################################

####################################################################################################### Functional Map Settings ##########################################################################################################################################
compute_FM = False
compute_FMdiagonal_analysis = True
compute_isometric_analysis = True
FM_k_eigenfunctions = (10,10)
FM_k_eigenvalues = 100
FM_descriptors = 'WKS+HKS+MKS'
FM_landmarks = 'precomputed'
compute_physic_fields = True
FM_symmetri = {'symmetry_mode': 'landmarks+orientation+extrinsic'}

####################################################################################################### Reeb Graph Settings ###############################################################################################################################################
compute_RG = False
compute_graph_time_analysis = True
reeb_scalar_field = 'geodesic' #'mass_center_geodesic' 
bins = 30
vertex_ref_index = [4896]

####################################################################################################### Basic Geometry Settings ############################################################################################################################################
compute_BasicGeo = True
plot_basicGeo = True
metrics = 'all' # ['n_vertices', 'n_faces', 'area', 'volume', 'sphericity', 'gaussian_curvature', 'convexity', 'center_mass']

####################################################################################################### Graph Similairty Meetrics ##########################################################################################################################################
compute_Graphsimilarity = True
graph_metrics = 'all' # ['degree_wasserstein','spectral_laplacian','interleaving_distance','labeled_interleaving_distance','function_distortion_distance','branch_decomposition_distance']


####################################################################################################### Pipeline Runing ###################################################################################################################################################
print('Executing pipeline ...')
run_pipeline(
    path_str=base_mesh_path,
    compute_basicGeo = compute_BasicGeo,
    metrics = metrics,
    plot_basicGeo = plot_basicGeo,
    matrix_tranformation=compute_FM,
    diagonal_analysis=compute_FMdiagonal_analysis,
    isometric_analysis=compute_isometric_analysis,
    k_eigenfunctions=FM_k_eigenfunctions,
    k_eigenvalues=FM_k_eigenvalues,
    descriptor=FM_descriptors,
    landmarks=FM_landmarks,
    fm_params=FM_symmetri,
    compute_physic_fields=compute_physic_fields, 
    compute_reeb=compute_RG,
    time_graph_analysis=compute_graph_time_analysis,
    reeb_scalar=reeb_scalar_field,
    bins=bins,
    vertex_ref_index=vertex_ref_index,
    graph_sim = compute_Graphsimilarity,
    graph_metrics = graph_metrics
    )

###########################################################################################################################################################################################################################################################################


#######################################################################################################  Executions Outside of the loop ###################################################################################################################################


####################################################################################################### Edit Created Reeb Graph  ##########################################################################################################################################
print('Editing reeb graph...') 
edit_graph(mesh_path, reeb_path)

####################################################################################################### Running Reeb Graph Time Analysis  ##################################################################################################################################
print('Graph path analysis...')
graph_time_analysis(reeb_path)

####################################################################################################### Generating Reeb Graph Time Analysis Plots  #########################################################################################################################
print('Plotting dynamic graph analysis...')
plot_dynamic_graph_analysis(csv_file_path)

####################################################################################################### Reeb Graph Visualizer Launcher  #####################################################################################################################################
print('Reeb visualizations...') 
visualize_reeb_graphs(mesh_path, reeb_path)

####################################################################################################### Functional Map Visualizer Launcher  #################################################################################################################################
print('Multi-Physics Mapping visualizations...') 
visualize_physics(mesh_path, matrix_path, on_time=False)


####################################################################################################### Mesh sequence Visualizer Launcher  #################################################################################################################################
print('Mesh sequence Visualization...') 
visualize_obj_sequence(mesh_objs_folder)

###################################################################################################### Similarity Metrics Among Graphs and ploting  ##########################################################################################################################
csv_sim_path = graph_similarity(reeb_folder_path=reeb_path,metrics_list=graph_metrics)
plot_graph_similarity(csv_sim_path)


################################################################################################################################################################################################################################################################################

####################################################################################################### .mat image/mesh visualizer #############################################################################################################################################
from PynamicMesh.utils.mat_files import MatViewer 
from pathlib import Path

folder_path = Path(r'C:\Users\jair.sanchez\Downloads\Projects\PynamicMesh\Real_cell\Dendetric\Dendetric1\Img')
viewer = MatViewer(folder_path)
viewer.show()

####################################################################################################### .mat image/mesh converter .obj/tiff #####################################################################################################################################
from PynamicMesh.utils.mat_files import mat_file_converter 
from pathlib import Path

folder_path = Path(r'C:\Users\jair.sanchez\Downloads\Projects\PynamicMesh\Real_cell\Dendetric\Dendetric1\Img')
mat_file_converter(folder_path)

#################################################################################################################################################################################################################################################################################

####################################################################################################### Paths reference list ####################################################################################################################################################

from PynamicMesh.utils.batch import run_batch
from PynamicMesh.utils.tools import extract_yaml

config_path = '/PynamicMesh/examples/config_batch.yaml' # vlinux
config_path = r'\PynamicMesh\examples\config_batch.yaml' # windows

config = extract_yaml(config_path)
data_cfg = config.get("Data", {})
path_str = data_cfg.get("path_str")
run_batch(config,path_str)

##################################################################################################################################################################################################################################################################################
####################################################################################################### FM use of cases example ###################################################################################################################################################

#   'a' : FM without landmarks, plain intrinsic descriptors            (baseline; symmetric flips possible)
#   'b' : FM with precomputed (manually selected) landmarks only
#   'c' : FM with automatic landmarks only
#   'd' : FM with symmetry-aware descriptors only, no landmarks at all  (orientation term + extrinsic XYZ block)
EXAMPLE = 'd'

compute_FM = True
compute_FMdiagonal_analysis = True
compute_isometric_analysis = True
FM_k_eigenfunctions = (10, 10)
FM_k_eigenvalues = 100
compute_physic_fields = True

common_fm_params = {
    'n_descr': 100,
    'subsample_step': 4,
    'descr_params': {'nu': 1.5, 'k_smooth': 30, 'xyz_weight': 0.25},
    'fit_params': {'w_descr': 1e-1, 'w_lap': 1e-3, 'w_dcomm': 1.0},
    'refine': 'auto',
    'dt': 1.0,
    'verbose': False,        # prints the descriptor plan and the landmarks kept for each pair
}

FM_EXAMPLES = {
    # ---------------------------------------------------------------------------------------------------------------
    # a) No landmarks. Only intrinsic point signatures: the map is determined up to the intrinsic symmetries
    #    of the shape (left/right legs can be swapped between frames).
    'a': dict(
        descriptor='WKS+HKS+MKS',           # equal energy shares; or e.g. '0.5*WKS + 0.3*HKS + 0.2*MKS'
        landmarks=None,
        fm_params={**common_fm_params, 'symmetry_mode': 'none'},
    ),
    # ---------------------------------------------------------------------------------------------------------------
    # b) Precomputed landmarks only. Uses the selections stored by precompute_landmarks() /
    #    visual_selection_edition() (mood='FM'); landmark-localized WKS descriptors break the symmetry.
    'b': dict(
        descriptor='WKS+HKS+MKS',
        landmarks='precomputed',
        fm_params={**common_fm_params, 'symmetry_mode': 'landmarks',
                   'landmark_params': {'weight': 1.0, 'descriptor': 'WKS'}},
    ),
    # ---------------------------------------------------------------------------------------------------------------
    # c) Automatic landmarks only. Farthest-point samples on frame t (extremities first), matched on frame t+1
    #    (descriptor match within a spatial radius), outliers rejected by displacement / duplicates / geodesic
    #    consistency. Landmarks used are saved in Results\<scene>\Landmarks\.
    'c': dict(
        descriptor='WKS+HKS+MKS',
        landmarks='auto',
        fm_params={**common_fm_params, 'symmetry_mode': 'landmarks',
                   'landmark_params': {'n_landmarks': 12, 'match': 'hybrid', 'max_rel_dist': 0.15,
                                       'max_distortion': 0.25, 'min_landmarks': 4, 'search_rel_radius': 0.10,
                                       'weight': 1.0}},
    ),
    # ---------------------------------------------------------------------------------------------------------------
    # d) Symmetry-aware descriptors only, no landmarks. 'extrinsic' adds an aligned-coordinates (XYZ) descriptor
    #    block (valid between aligned consecutive frames), 'orientation' adds the orientation-preserving operator
    #    term (w_orient) that penalizes the mirrored map. The XYZ block can also be requested explicitly in the
    #    descriptor string, e.g. '0.5*WKS + 0.3*MKS + 0.2*XYZ' (then 'extrinsic' is redundant).
    'd': dict(
        descriptor='WKS+HKS+MKS',
        landmarks=None,
        fm_params={**common_fm_params, 'symmetry_mode': 'orientation+extrinsic',
                   'fit_params': {**common_fm_params['fit_params'], 'w_orient': 1.0}},
    ),
}
fm_settings = FM_EXAMPLES[EXAMPLE]

run_pipeline(
    path_str=base_mesh_path,
    compute_basicGeo=compute_BasicGeo,
    metrics=metrics,
    plot_basicGeo=plot_basicGeo,
    matrix_tranformation=compute_FM,
    diagonal_analysis=compute_FMdiagonal_analysis,
    isometric_analysis=compute_isometric_analysis,
    k_eigenfunctions=FM_k_eigenfunctions,
    k_eigenvalues=FM_k_eigenvalues,
    descriptor=fm_settings['descriptor'],
    landmarks=fm_settings['landmarks'],
    fm_params=fm_settings['fm_params'],
    compute_physic_fields=compute_physic_fields
)

##################################################################################################################################################################################################################################################################################