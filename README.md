# PynamicMesh: Dynamic Mesh Modeling & Analysis Tool 

<img src="./assets/logo.png" width="400" height ='250'/>


# How to Install
[Creating a new conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or a [virtual environment](https://docs.python.org/3/library/venv.html) with **Python 3.10+**.

```bash
conda create -y -n PynamicMesh -c conda-forge python=3.11
conda activate PynamicMesh
```

Clone repo and install dependencies :

```bash
git clone https://github.com/MMV-Lab/PynamicMesh
cd PynamicMesh
pip install -r requirements.txt
```

# Example Data

All the following examples can be replicated using the [meshes](./examples/Mesh_models/) and the generated example files provided [here](./examples) and all the code usage sintax are sumarized in the provided [code](./PynamicMesh.py)

The provided tools were developed independtly into pure computations and visualization/graphical tools in order to keep the flexibility of run the computings into pure non graphic node or high-performance computing cluster.

# Project overview

Given a family of meshes $ \mathscr{M} = \{ M_{t_i} | 0\leqslant i \leqslant T \}$ where each mesh $M_{t_i}$ encodes the spatial deformation of the shape on a specific time $t_i$, we deal with a shape deformation that provide a full geometrical description of the dynamic related to the related deformation prun_pipeline.

<img src="./assets/transf.gif" height ='180'/>


PynamicMesh offer a full genaral range opipelines based on Topology, Differential Geometry and Physics in order of modeling the complex dynamic encoded on the tranformation, allowing to extraction features that helps to characterize and understand the dynamical prun_pipeline.

For a detailed and applied understanding of meshes as Manifolds and triangulations the following [Jupyter Notebook](https://github.com/JairMathAI/Understanding_Persistent_Homology/blob/main/Persistent_Homology.ipynb) might interest you.

<details>
<summary><strong><span style="font-size:25px;">Functional Map</span></strong></summary>


<details>
<summary><span style="font-size:23px;">Understanding Functional Map Construction</span></summary>

The functional map $(\mathscr{FM})$ allows to compute a matrix representation of  a unknown tranformation function between meshes  $\mathscr{FM}: \mathcal{F}(M_1,M_2)\to \mathbb{M_{k\times k}(\mathbb{R})}$

Given two consecutive  time step meshes $M_{t_{i-1}}$  and  $M_{t_{i}}$, we can think abut them in terms of their respective vertices (points) and faces (triangles): $\{\mathcal{V},\mathcal{F}\}_{t_{i-1}}$  and  $\{\mathcal{V},\mathcal{F}\}_{t_{i}}$


The goal it's to find a representation of the unknown biyective transformation function $\varphi_n : M_{t_{i-1}} \to M_{t_{i}}$  that describe how the mesh is transformed on the space.



<img src="./assets/FM.PNG" height=200/>


We can use a scalar function defined over each mesh $\psi_{t_{i-1}}: M_{t_{i-1}} \to \mathbb{R}$ and $\psi_{t_i}: M_{t_i} \to \mathbb{R}$ which produce the relation $\psi_{t_i} = \psi_{t_{i-1}} \circ \varphi_n^{-1} = \psi_{t_{i-1}}(\varphi_n^{-1})$

<img src="./assets/scalar_map.PNG" height=200/>

This composition induce a linear functional, such that for every fuction $f:M_{t_{i-1}} \to \mathbb{R}$ we have $\mathcal{F}_{\varphi_n}(f) = f(\varphi_n^{-1})$ so we have the functional transformation $\mathcal{F}_{\varphi_n} : \mathcal{L}(M_{t_{i-1}},\mathbb{R})\to\mathcal{L}(M_{t_{i}},\mathbb{R}) $ where the task to find $\varphi_n$ now means find a represetantion for the functional $\mathcal{F}_{\varphi_n}$

As the linear function spaces  $\mathcal{L}(M_{t_{i-1}},\mathbb{R})$ and $\mathcal{L}(M_{t_{i}},\mathbb{R})$ are vectorial spaces we can use the Laplace-Beltrami eigenfunctions basis and apply a prun_pipeline of optimization to determine the functional map  (Matrix representation) of the functional $\mathcal{F}_{\varphi_n} = C_{t_{i-1}\to t_{i}}\in \mathbb{M_{k\times k}(\mathbb{R})}$ that encodes the spectral representation of the functional and the vector $\vec{v}$ that encode the explicit (spatial) vertex tranformation where $v_j=k$ means that the vertex $j$ it's transformed into the vertex $j$.

<b>Notes:</b>
- This prun_pipeline works for Meshes with diferent number of vertex because is based on a prestablished eigenfunction base of sixe $k$
- The respective $\psi_{t_{i-1}}: M_{t_{i-1}} \to \mathbb{R}$ maps used for computations are the Direchlet-Energy over the mesh of the Wave propagation function, the Heat diffusion function or the sum of both $\displaystyle E[\Psi]=\frac{1}{2}\int ||\Psi(x)||^2 dx$
- The Direchlet-Energy $E[\Psi]$ it's unsensitive to spatial symmetries so the maps could fail on symmetric regions, to solve this the use of landmarks as reference points it's allowed this induce condition on the optimization prun_pipeline in order to map correctly the symetric regions.  


Having this metric representation we can compute a big variety of descriptors  to characterize the dynamic.

The Full prun_pipeline can be sumarized through the following scheme:

<img src="./assets/FMComp.gif" width=900/>

</details>

<details>

<summary><span style="font-size:23px;">Functional Map Implementation Usage and Analysis</span></summary>

The computations are executed and managed throug the syntax:


```python
from core.pipelines import run_pipeline
run_pipeline(**args)
```

In order to compute the Functional Map tranformations run 

```python
from core.pipelines import run_pipeline
run_pipeline(
                        path_str='base/path',
                        matrix_tranformation=True,
                        diagonal_analysis=True,
                        isometric_analysis=True,
                        k_eigenfunc=100,
                        descriptor='WKS+HKS',
                        landmarks='precomputed',
                        compute_physic_fields=True,
                        )
```

<details>
<summary><span style="font-size:21px;"> Functional Map Parameters</span></summary>

Path to the root folder that contain the scenes
```python 
   path_str (str) 
```   

Flag to indicate the model execution
```python 
    matrix_tranformation (bool)
```  

Flag to indicate the isometry analysis execution within the loop
```python 
    isometric_analysis (bool)
```

Flag to indicate the diagonal analysis execution within the loop
```python 
    diagonal_analysis (bool)
```

Flag to indicate the computation and store of the physical fields (once) if it's false the visualizator will compute them on execution time every time. 
```python 
    compute_physic_fields (bool)
```
    
Energy function used on the prun_pipeline WKS (wave propagation kernel) , HKS (heat diffusion kernel) or WKS+HKS (both)
```python 
    descriptor (str): 'HKS' | 'WKS' |  'WKS+HKS'
``` 

    
Number of eigenfunction used on the matrix computation matrix of size $k_\text{eigenfunc} \times k_\text{eigenfunc}$

```python 
k_eigenfunc (int)
```     

Vertex indices indicators for symmetry restriction 

```python 
landmark (list|str):
``` 

<details>
<summary><span style="font-size:19px;">Landmark Options</span></summary>
No symmetry restrictions applied    
```python 
landmark (str) : None
``` 
A priori knowed indices of the $n$ symetrical vertex (when the same works to all the tranformations)

```python l
landmark (list) : [1,2,3,4,..,n] -> (n,)
``` 

A priori knowed indices of the symetrical vertex (one  per considered tranformations) if the list contain less sets of vertex than the pair of meshes the remain computings will performance without restrictions.

```python 
landmark (list) : [[1,..,n1],..,[1,..,nk]] -> (n,m)
``` 

A priori knowed pair indices of the  symetrical vertex here $[j,k]$ means that the $v_j$ vertex of the mesh $M_{t_{i-1}}$  its related to the $v_k$ vertex  of the mesh  $M_{t_i}$ (the same pair applied to every tranformation)

```python 
landmark (list) : [[1,2],...,[j,k]] -> (2,n)
``` 

A priori knowed pair indices of the  symetrical vertex  (one set of relation considered per tranformation) if the list contain less sets of vertex relations than the pair of meshes the remain computings will performance without restrictions.

```python 
landmark (list) : [[[1,2],...,[j,k]],...,[[1,2],...,[l,m]]]] -> (m,2,n)
```

<b>Note:</b>

The independent function ```precompute_landmarks(root_path,'FM')``` can be used and you use the graphical tool for click over the vertex selection over every escene on the project or the function ```visual_selection_editio(path_to_meshes,'FM')``` it's included  to precompute or edit the existed landmarks for a specific scene, both functions will generate landmark.npy file with the corresponding selected vertex relations, then this option will take a look for this precomputed .npy file during execution.

```python 
landmark (str) : 'precomputed'
``` 
</details>
</details>

<details>
<summary><span style="font-size:21px;">Landmarks Grafical Selection</span></summary>

For the Precomputed landmarks 

```python
from core.custom_fm import precompute_landmarks

precompute_landmarks("./PynamicMesh/Mesh_models",'FM')

```

Vertex are choosen doing click over it and unmarked click in again over the selected, when the selection is ready just  close the window in order to pass to the next mesh and the prun_pipeline is the same.

The vertex selection should be performed on the same order for each mesh so the right related pairs are formed, the selectd vertices should be aproximately corresponding on the same related regions of the meshes.

At the end the .npy file with our vertex selection on each frame will be stored in the path  ./PynamicMesh/Results/scene1/landmark.npy .

<img src="./assets/vertex_selection.gif" width=1600 height=1100/>

If we use the other function 

```python
from core.custom_fm import visual_selection_edition

visual_selection_editios("./PynamicMesh/Mesh_models/scene1",'FM')

```

The visualizer is going to show the specific mesh dynamic and the landmarks created previously given the chanche of edit them or also created them if there not exist.

In this case in order to change among meshes in the time use the arrow keys on the keyboard, when the edition is finished just cose the window and it will be stored. 

<img src="./assets/land_edit.gif" width=1600 height=1100/>

</details>

<details>
<summary><span style="font-size:21px;"> Physical Features and Map Transformation Models </span></summary>

We can run our matrix transformation computing, at the end of the run the pipeline will save one matrix per transformation (FMC_ij.npy) and the corresponding vector of vertex tranformations (FMV_ji.npy) within the folder ./PynamicMesh/Results/scene1/Transform_Matrices/

Let's run firts the pipeline without landmarks to see the results.

```python
from core.pipelines import run_pipeline
run_pipeline(base_mesh_path, matrix_tranformation=True, descriptor='WKS+HKS',landmarks=None, k_eigenfunc=100)
```


Compute the descriptors and visualize them with:

```python
from core.physic_model import  visualize_physics
visualize_physics("./PynamicMesh/Mesh_models/", "./PynamicMesh/Results/scene1/Transform_Matrices/",on_time=False)
```

Te parameter on time indicate if need to compute the physic field on the execution or just look for the precomputed and stored results created during ```run_pipeline```
```python
on_time(bool)
```

The program will compute and deploy the following:



<img src="./assets/no_landmarks.gif" width=1600 height=1100/>

<b>$\Delta$-Color tranfer</b>: 

Showing with different colors which region of the mesh $M_{t_{i-1}}$ is tranformed into which  over the mesh $M_{t_{i}}$ 

<b>$\Delta\vec{v}$ Vertex velocity displacement</b>: 

Showing the map obver the mesh about the velocites ratios of change respect the vertex maping (euclidean velocity of displacement) $||\vec{v}_{t_{i-1}}-\vec{v}_{t_{i}}||$, that means coloring the regions of faster change.


<b>Linear (edge) stain</b>: 

Plot the result of the finite elements prun_pipeline to compute the zones of strech and compresion with respect of the edges (1D),
that is where de coeficient $\varepsilon >0 \to$ strech edges and $\varepsilon < 0 \to$ contraction edges.

<b>Area (Face) stain</b>: 

Plot the result of the finite elements prun_pipeline to compute the zones of strech and compresion with respect of the faces (2D),
that is where de coeficient $\varepsilon >0 \to$ strech areas and $\varepsilon < 0 \to$ contraction areas.

<b>Normal prostrusion flow $\vec{v}|| \vec{N}$</b>: 

Descrive vertex-wise normals of the mesh geometry. That os the  "outward" or "inward" direction relative to the surface.

(Areas where the mesh is actively pushing out or pulling in)

<b>Tangent flow $\vec{v} \hookrightarrow \vec{T}$</b>: 

Subtracts the normal component from the total displacement to isolate the lateral movement, represents the lateral or crawling flow across the surface. This captures how the mesh material slides or shifts across the surface without necessarily changing the local thickness or volume.

(Surface flow, where the mesh might be rearranging its surface area laterally)

<b>Note:</b>

The Camel is almost symmetric by the middle, due this the $\Delta$-Color tranfer show that in the Fuctional maping capture well the transformation of the frontal legs that is, during all the times $t$ the right frontal leg is blue and the left frotal leg it's purple but on the case of the hind legs, we can see that in the transition $t_0\to t_1$ and $t_6\to t_7$ the legs and the half hind body swap colors, this because the Dirichlet energy can't capture the symmetries, this mean that our matrix representation it's not correct there, but using our landmarks we can see that this error is corrected, we can run:

The used landmarks in this examples are provided here [here](./examples/landmarks.npy)

```python
from core.pipelines import run_pipeline
from core.physic_model import  visualize_physics

run_pipeline(base_mesh_path, matrix_tranformation=True, descriptor='WKS+HKS',landmarks='precomputed', k_eigenfunc=100)
visualize_physics("./PynamicMesh/Mesh_models/", "./PynamicMesh/Results/scene1/Transform_Matrices/",on_time=False)
```

<img src="./assets/lanmarks_solve.gif" width=1600 height=1100/>


</details>

<details>
<summary><span style="font-size:21px;">Isometry Transformation Tracking</span></summary>

Beside the animations and computation within the folder (./PynamicMesh/Results/scene1/Diagonal_analysis) will be reported the Heatmap  of the matrix representation in each time step this heat map $C_{t_{i-1}\to t_{i}}$ codify the nature of the tranformation if the Heatmap it's diagonal the tranformation it's close to and isometry (The topology keeps intact and the tranformation is a rotation or shift) if the values of the Heatmap are sparse (far from the diagonal) means that during the tranformation the topology have significative deformations, in our camel example the changes on the topology are not to agresive almost only the legs change positions so the matrices are almost near to the diagonal, in order of track the diagonality of the Heatmaps during the time, we track three metrics an report the results on a csv file:


<b>Moment of Inertia Metric: </b> 

$\displaystyle MI(C_{t_{i-1}\to t_{i}})= 1-\frac{\sum_{ij}|i-j|^2|c_{ij}|}{n|k^2|^2}$ where $\displaystyle  n|k*k|^2 = \sum_{ij}d^2_{max}(c_ij)$ , this is 1 - inertia.  

If $MI(C_{t_{i-1}\to t_{i}})\to 1$ he energy is concentrated directly on or immediately next to the main diagonal.  

If $MI(C_{t_{i-1}\to t_{i}})\to 0$ Indicates energy has escaped to the upper or lower corners of the matrix.

<b>Note:</b>

This metric is highly sensitive to far-away outliers, meaning even a small amount of energy in the far corners will cause this metric to drop significantly.

<b>Exponential Decay Metric: </b> 

$\displaystyle ED(C_{t_{i-1}\to t_{i}})= \frac{\sum_{ij}|c_{ij}|e^{0.5|i-j|}}{\sum_{ij}|c_{ij}|}$ 

If $ED(C_{t_{i-1}\to t_{i}})\to 1$ Confirms that the energy is resting inside a tight, focused band along the diagonal.

If $MI(C_{t_{i-1}\to t_{i}})\to 0$ Indicates a highly diffused, blurry, or scattered functional map where the diagonal is poorly preserved.

<b>Cumulative Distribution Function Bandwidth: </b>

The function loops through every possible bandwidth radius k (from 0 up to the maximum dimension of the matrix). At each step, it accumulates the percentage of total energy contained within a diagonal band of thickness k: 

$\displaystyle CDF(C_{t_{i-1}\to t_{i}},k)= \frac{\sum_{|i-j|\leqslant k c_{ij}}}{\sum_{ij}|c_{ij}|^2} $ Exact percentage of total energy sitting directly on the core main diagonal line, 

If the plotted curve shoots up vertically and hits 1.0 at a very low bandwidth (k=2 or 3), the matrix is tightly bounded around the diagonal.

If the curve scales up gradually as a slow diagonal line, it indicates that the spectral energy is leaking into wide off-diagonal frequencies.


<img src="./assets/FM_Heatmap_Animation.gif" style="display: inline-block; width: 200; height:200"/>
<img src="./assets/Diagonal_metrics.png" style="display: inline-block; height: 200; "/>


</details>


<details>
<summary><span style="font-size:21px;">Heatmaps Similarity Metrics</span></summary>

A comparation among $C_{t_{i-1}\to t_{i}}$ vs  $C_{t_{i}\to t_{i+1}}$ trought the Corss Heatmaps Similarities is procided, this comparation means to look at the derivative of the deformation:

<b>Jensen-Shannon Divergence: </b> 

Treats the squared matrix as an "energy distribution" and measures how much the allocation of energy changes between the two mappings. A spike in JSD indicates a sudden phase shift in the physical deformation. 

For example, if a mesh was smoothly expanding over time, but suddenly starts twisting or twisting, the energy distribution across the matrix will dramatically change, and JSD will spike.

<b>Pearson & Spearman Correlation: </b> 

Measures how linearly aligned (Pearson) and structurally ranked (Spearman) the cells of $C_{t_{i-1}\to t_{i}}$ are to $C_{t_{i}\to t_{i+1}}$ 

High correlation means the "nature" or "pattern" of the deformation is steady and consistent. If a mesh is undergoing a continuous, prolonged stretch in one direction over several frames, the FMs will look structurally identical. A drop in correlation means the mesh has started a new, different movement.

<b>Manhattan $L_1$​ and Euclidean $L_2$ Distances: </b>

Measures the raw geometric difference between the specific coefficient values of the two matrices.

This acts as a measure of acceleration or intensity change. If the deformation is speeding up or becoming more drastic between frames, the coordinate distances will increase, even if the general shape of the matrix (the correlation) stays roughly the same.

<img src="./assets/Cross_Heatmap_Similarity.png" height =200 />

</details>
</details>
</details>



<details>
<summary><strong><span style="font-size:25px;">Reeb Graph</span></strong></summary>

<details>
<summary><span style="font-size:23px;"> Understanding Reeb Graph Construction</span></summary>

The Reeb Graph ($\mathscr{RG}$) is a power full tool that allows to compute a graph representation of a mesh (topology skelleton) using Morse theory (level curve or contour lines of the mesh ) $\mathscr{RG}:\mathcal{M}\to \{(V,E),\tau\}$.

Given a real scalar field over the mesh $f:M_{t_{i}} \to \mathbb{R}$ and the following equivalence relation  $x_1,x_2 \in M_{t_{i}}$ are related $x_1\sim x_2$ if and only if they belong to the same level set:  $x_1,x_2 \in f^{-1}(c)$.

Then the Reeb graph is the topological quotient space induced by the relation, endowed with the quotient topology  $(M_{t_{i}}/\sim,\tau_{\sim})$.

Even when the definition can be abstract the idea is very intuitive, think about the scalar field  $f:M_{t_{i}} \to \mathbb{R}$ as the map that tell how to travel trough the mesh.

The condition $x_1,x_2 \in f^{-1}(c_i)$ means that $f(x_1)=f(x_2)=c_i$ for a specific value $c_i$ this tells us that on the level $c_i$ of the travel we need to cut a slice of the surface $M_{t_{i}}$ this is the level curve or contour line of the meash on the lever $c_i$.

Defining the equivaent relation means that we need to look how many points of the surface $x \in M_{t_{i}}$ on that slice take the value $c_i$, that is $f(x)=c_i$, then supose that in that level we have $k$ different points that take this value $\displaystyle x^i_1,...,x^i_k$ taking the "equivalence relation" means that we are goin to think now about all this point as "the same single thing" that is as a single point $\displaystyle v_{c_i} = [c_i] =\{x^i_1,...,x^i_k\}$ that only means that we are identifying this pints with a single vertex on our graph construction.

Saying that the graph has the quotient topology, means that the graph capture the topology relationships of the mesh.


The idea is easy to follow grafically: 


<img src="./assets/reeb_T.png" height =300 />

The Full dynamical analysis prun_pipeline can be sumarized through the following scheme:

<img src="./assets/RGComp.gif" height =300 />

</details>

<details>
<summary><span style="font-size:23px;"> Reeb Graph Usage and Analysis</span></summary>

The computations are executed and managed throug the syntax:


```python
from core.pipelines import run_pipeline
run_pipeline(**args)
```

In order to compute the Reeb Graph run 

```python
from core.pipelines import run_pipeline
run_pipeline(
                        path_str='base/path', 
                        compute_reeb=True,
                        time_graph_analysis=True,
                        reeb_scalar="geodesic",
                        bins=30,
                        **scalar_fields_args 
                    )
```

<details>
<summary><span style="font-size:21px;"> Reeb Graph Parameters</span></summary>

Path to the root folder that contain the scenes
```python 
   path_str (str) 
```   

Flag to indicate the model execution
```
    compute_reeb (bool)
```  
Number of levels sets used for the graph computing
```python 
    bins (int)
```

Scalar field used for compute the graph
```python 
    reeb_scalar (str)
```

Parameters related to the selection of Sacalar Fiels ```param_name = param_value```
```python 
    **scalar_fields_args (dict)
```

Flag to indicate if run the analysis withing the loop, this will run the analysis over the raw computed graphs if you need to run the analysis on the graphs after a modification you can run it independently over the modificated graphs.
```python 
    time_graph_analysis (bool)
```

<details>
<summary><span style="font-size:19px;"> Available Scalar Fields</span></summary>

## Spatial Based

<b>$(x,y,z)$-Level sets :</b>  

```python 
reeb_scalar="x" | reeb_scalar="y" | reeb_scalar="z"
```

<b>Distance from the center of mass :</b> 

```python 
reeb_scalar="dist_centroid"
```

<b>Signed distance relative to parallel planes crossing the centroid :</b> 

```python 
reeb_scalar="signed_dist_x" | reeb_scalar="signed_dist_y" | reeb_scalar="signed_dist_z"
```

<b>Absolute distance to specified axis :</b> 

```python 
reeb_scalar="dist_x_axis"| reeb_scalar="dist_y_axis" | reeb_scalar="dist_z_axis" 
```

## Geometric/Topology based

<b>Geometric mean curvarture :</b> 

```python  
reeb_scalar="mean_curvature"
```

<b>Gaussian curvarture  :</b> 

```python 
reeb_scalar="gaussian_curvature"
```

<b>Shape base index  :</b> 

```python  
reeb_scalar="shape_index"
```

<b>Curve base index  :</b> 

```python  
reeb_scalar="curvedness"
```
 
<b>Protrusion mapping based on mesh $M_{t-1}$ :</b> 

```python 
reeb_scalar="normal_displacement"
```


<b>Spectral mapping based on $n$ Laplace-Beltrami eigenfunctions:</b> 

```python 
reeb_scalar="lb_eigen_n" 
```

<b>Multi scalar field maps combination  trough  Mapper Lens construction (PCA feature extraction)</b> 

```python  
reeb_scalar="multi_pca", fields=["f1","f2",..,"fn"] 
```

<b>Spectral mapping based on Heat diffusion , source point (vertex index $i$) $v_i$ and time $t$ :</b> 

```python 
reeb_scalar="heat_diffusion", source_idx=i, t=t 
```

<details>
<summary><span style="font-size:17px;"> source_idx options</span></summary>

```python 
source_idx (str|list|int)
t(int) 
```
While the time $t$ remains the same for every mesh of the scene ```source_idx``` can codify different options:

Apply the same heat source ($v_n$) for all the meshes

```python 
source_idx (int): n
```

Apply the heat sourse $v_i$ for the mesh $M_i$ if $n$ is less than the number of meshes for the rest  the default value ```source_idx=0``` will be applied
```python 
source_idx (list): [0,1,2,3,...,n]
```

Apply the heat sourse $v_i$ for the mesh $M_i$  for the meshes on the  None position the default value ```source_idx=0``` will be applied

```python 
source_idx (list): [0,None,None,3,...,n]
```

Visual selecction, in the same fashion as in the case of the functional map (see  section Landmarks grafical selection for functional maps)

```python 
source_idx (str): "precomputed"
```

</details>

<b>Spectral mapping based on Harmomic with boundary conditions  injection flow in vertex  $v_i$ and leaving flow in vertex  $v_t$: </b> 

```python 
reeb_scalar="harmonic", source_idx=i, sink_idx=t 
```

<details>
<summary><span style="font-size:17px;"> source_idx options</span></summary>

While  on the case of  ```source_idx=i,sink_idx=j```  we refere to the boundary conditions  injection flow in vertex  $v_i$ and leaving flow in vertex  $v_j$ ```source_idx`` can codify different options:

```python 
source_idx (str|list|int)
sink_idx (int) 
```

Use the pair $(i,j)$ as bundary condition $v_i$, $v_j$  for all the meshes.

```python 
source_idx (int)
sink_idx (int) 
```

Use every pair $(i,j)$ in the list as bundary condition $v_i$, $v_j$ ```source_idx=i , sink_idx=j ``` for each mesh, if there are less pairs than meshes for the rest the default value ```source_idx=min(index) , sink_idx=max(index) ``` will be applied

```python 
source_idx (list): [[1,2],[3,4],...,[i,j]]
```

Use every pair $(i,j)$ n the list as bundary condition $v_i$, $v_j$ ```source_idx=i , sink_idx=j ``` for each mesh, for the None position the default value ```source_idx=min(index) , sink_idx=max(index) ``` will be applied

```python 
source_idx (list): [[1,2],None,...,[i,j]]
```

Visual selecction, in the same fashion as in the case of the functional map (see  section Landmarks grafical selection for functional maps)

```python 
source_idx (str): "precomputed"
```
</details>

<b>Geodesic mapping based on vertex landmarks $[v_0,...,v_n]\in M_t$  :</b>

```python 
reeb_scalar="geodesic" vertex_ref_index=[0,1,2,n] 
```

<details>
<summary><span style="font-size:17px;"> vertex_ref_index options</span></summary>

The parameter ```vertex_ref_index``` can codify different options:

```python 
vertex_ref_index (str|list)
```

Apply the set of reference vertex $$[v_0,...,v_n]$ for every mesh  

```python 
vertex_ref_index (list): [0,1,2,3,...,n] 
```

Apply each set of reference vertex $$[v_0,...,v_n]$ for each mesh  in the case of have less reference set for the rest the default value ```vertex_ref_index=[0]``` will be applied.

```python 
vertex_ref_index (list): [[0,...,n1],[0,...,n2],...,[0,...,nk]] 
```

Apply each set of reference vertex $$[v_0,...,v_n]$ for each mesh  in the case None positions the default value ```vertex_ref_index=[0]``` will be applied.

```python 
vertex_ref_index (list): [[0,...,n1],None,...,[0,...,nk]]
```

Visual selecction, in the same fashion as in the case of the functional map (see  section Landmarks grafical selection for functional maps)

```python 
vertex_ref_index (str): "precomputed"
```
</details>

<details>
<summary><span style="font-size:17px;"> Visual reference notes</span></summary>

In each cases of the Heat diffusion, Harmonic and Geodesic based scalar maps, the optional reference point can be selected grafically with the excecution of the corresponding code :
```python 
from core.custom_fm import  visual_selection_editio, precompute_landmarks


################################ Sources Vertex index  precompute visual tools for 'heat_diffusion' in Reebs #############################################################
print('Visualizing or editing Sources Vertex index for RG...')
visual_selection_editio(mesh_path,'heat_diffusion')

print('Precomputing Sources Vertex index for RG...')
precompute_landmarks(base_mesh_path,'heat_diffusion')

################################ Source-sink Vertex index  precompute visual tools for 'harmonic' in Reebs #############################################################
print('Visualizing or editing Source-sink Vertex index index for RG...')
visual_selection_editio(mesh_path,'harmonic')

print('Precomputing Source-sink Vertex index for RG...')
precompute_landmarks(base_mesh_path,'harmonic')

################################# Vertex index reference precompute visual tools for 'geodesic' in Reebs #############################################################
print('Visualizing or editing Vertex index for RG...')
visual_selection_editio(mesh_path,'geodesic')

print('Precomputing Vertex index for RG...')
precompute_landmarks(base_mesh_path,'geodesic') 

```
Each one will generate and save the respective sources.npy, source_sink.npy, vert_ref_geo.npy file withing the folder ./PynamicMesh/Results/scene1

<b>Note:</b>

If durig the pipeline execution prun_pipeline the corresponding ```'precomputed'``` function is used but no .npy files are foun for a certain folder the default values will be used.

</details>
</details>
</details>
<details>
<summary><span style="font-size:21px;">Reeb Graph Visualization</span></summary>

After the modeling prun_pipeline execution the files Reeb_Ti.pkl and Scalar_Ti.npy (one for each time $t$) will be saved within the folder ./PynamicMesh/Results/scene1/Reeb_Graphs.

With this files we can visualize the evolution of the field and the graph over the time

```python 
from core.pipelines import run_pipeline
from core.reeb_graph import graph_time_analysis, visualize_reeb_graphs

print('Executing modeling ...')
run_pipeline(base_mesh_path, compute_reeb=True, bins=30 , reeb_scalar='geodesic', vertex_ref_index=[4896])

print('Reeb visualizations...') 
visualize_reeb_graphs(mesh_path, reeb_path)
```

<img src="./assets/RG_view.gif" width=1600 height=1100/>

</details>

<details>
<summary><span style="font-size:21px;">Reeb Graph Edition Tool</span></summary>

When the graphs are created we can use the grafical tool to edit the created graph:

Click over existing vertex (on the graph side) $to$ Delete the vertex and all the conected edges to it.
Click on the surface mesh vertex (on the mesh side) $to$ Create a vertex and click on a existing graph vertex (on the graph side) to create the edge among them.
Press key i $to$ Activate INNER mode and when you click on a vertx it moves ortogonal into the mesh (press again to deactivate)
Press key o $to$ Activate OUTER mode and when you click on a vertx it moves ortogonal closer to the mesh surface (press again to deactivate mode)
Press key c $to$ Activate LINK mode and when you click on a two existing vertex on the graph side the egde among them is created  (press again to deactivate mode)
Space Bar $to$ Undo the last change
Use s and w keys to activate/deactivate visible layer on the mesh
Use the arrow keys to change the graph on time.

When the edition is ready just close the window, only the corresponding modificated graphs will be saved.

<b>Note:</b>
The original computed graphs are no overrided the modificated graphs will be stored within the folder ./PynamicMesh/Results/scene1/Reeb_graph_manual_edit

<img src="./assets/editionRG.gif" width=1600 height=1100/>

</details>
<details>
<summary><span style="font-size:21px;">Time Graph analysis and plotting</span></summary>

When the desired graph are ready and saved , we can run a temporal analysis and generate the plot of the results adn the csv with the data:

```python 
from core.reeb_graph import graph_time_analysis, plot_dynamic_graph_analysis

print('Graph path analysis...')
graph_time_analysis(reeb_path)

print('Plotting dynamic graph analysis...')
plot_dynamic_graph_analysis(csv_file_path)
```

<b>Structural Complexity (Nodes & Edges)</b>

Encode the raw size of the Reeb graph skeleton. This tracks how "complex" or "branchy" the shape is.

A spike in nodes and edges indicates the mesh is growing new appendages, fragmenting, or wrinkling on time.

A drop indicates the mesh is smoothing out, shrinking, or parts are merging together on time.

<b>The "Intensity" of Deformation (The Distance Metrics)</b>

We calculates three distance metrics (Wasserstein, Spectral Laplacian, and Graph Edit Distance) between consecutive time steps. Together, these act as an "earthquake seismograph" for the meshes.

Encode how drastically the skeleton shifted from $t_{i−1}$​ to $t_i$​. 

Smooth, low values means that the mesh is experiencing stable, continuous deformation (e.g., simply moving or slowly expanding).

Sudden spikes, Indicates a critical topological event in the system. The mesh just underwent a sudden structural change, such as breaking apart (fission), colliding/merging (fusion), or suddenly collapsing.

The Graph Edit Distance highlights direct physical breakages/additions of branches, while Spectral Distance highlights global warping of the overall shape.

<b>Holes, Loops, and Fusions (Betti-1 Cycles)</b>

The Betti-1 number counts the number of 1D loops or cycles in the graph.  It detects when the shape folds back on itself to create a hole or a tunnel (like a donut). 

An increase in cycles means appendages have touched and fused together, creating a closed loop.

<b>Stretching and Elongation (LCC Diameter)</b>

The "Diameter" of the Largest Connected Component represents the longest shortest-path across the graph's skeleton. 
It measures the maximum spatial span of the object. If the diameter steadily increases while the number of nodes stays the same, it means your mesh is being stretched or elongated (like pulling a piece of taffy).

This analysis allow us to automatically pinpoint exactly when and how your 3D meshes undergo major structural changes without having to manually watch the 3D animation. It converts visual shape evolution into a dashboard of growth (size), drastic events (distances), stretching (diameter), and fusions (cycles).

<img src="./assets/1_Structural_Size.png" style="display: inline-block;  height:200"/>
<img src="./assets/2_Graph_Distances.png" style="display: inline-block; height: 450; "/>
<img src="./assets/3_Internal_Topology.png" style="display: inline-block; height: 200; "/>

</details>
</details>
</details>




<!-- <details>
<summary><strong><span style="font-size:25px;"> AAAAAAAAAAAAAAAAS</span></strong></summary>
</details> -->





