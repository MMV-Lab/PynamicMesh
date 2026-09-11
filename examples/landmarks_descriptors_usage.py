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


####################################################################################################### Functional Map Landmarks and Descriptors usage example  ###########################################################################################################

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