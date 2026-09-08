

import os
import re
from pathlib import Path
import h5py
import numpy as np
import pyvista as pv
import scipy.io
import tifffile  
from matplotlib.colors import LinearSegmentedColormap
from PynamicMesh.utils.tools import natural_sort_key

import sys


def create_4d_obj(obj_folder_path):
    """
    Reads all .obj files in a folder and combines them into a single 4d_mesh.obj.
    Each file is stored as a separate group/object frame.
    """
    folder_path = Path(obj_folder_path)
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    # Gather and sort the .obj files naturally
    obj_files = sorted(list(folder_path.glob("*.obj")), key=natural_sort_key)
    
    if not obj_files:
        print(f"No .obj files found in '{folder_path}'.")
        return
        
    # Set the output path to the parent directory, matching the 4D TIFF logic
    out_path = folder_path.parent / "4d_mesh.obj"
    
    print(f"Combining {len(obj_files)} .obj files into '{out_path}'...")
    
    vertex_offset = 0
    
    with open(out_path, 'w') as out_f:
        out_f.write("# 4D Combined OBJ File\n")
        
        for i, obj_file in enumerate(obj_files):
            # Label each mesh as a separate frame for 4D parsing
            out_f.write(f"\no frame_{i:04d}\n")
            out_f.write(f"g frame_{i:04d}\n")
            
            local_vertex_count = 0
            
            with open(obj_file, 'r') as in_f:
                for line in in_f:
                    if line.startswith('v '):
                        out_f.write(line)
                        local_vertex_count += 1
                        
                    elif line.startswith('f '):
                        # OBJ face indices are 1-based and global. 
                        # We must shift the indices by the total vertices loaded so far.
                        parts = line.strip().split()[1:]
                        new_face_parts = []
                        
                        for p in parts:
                            indices = p.split('/')
                            # Shift the vertex index (the first number in the face definition)
                            v_idx = int(indices[0])
                            new_v_idx = v_idx + vertex_offset
                            
                            # Reconstruct the face definition string (handles v, v/vt, or v/vt/vn)
                            new_p = str(new_v_idx)
                            if len(indices) > 1:
                                new_p += '/' + '/'.join(indices[1:])
                                
                            new_face_parts.append(new_p)
                            
                        out_f.write("f " + " ".join(new_face_parts) + "\n")
                        
                    # (Optional) If your source OBJs contain normals (vn) or textures (vt)
                    # you would need to track offsets for those as well. Assuming standard 
                    # generated geometry from your earlier script, 'v' and 'f' is sufficient.
            
            # Update the global offset for the next frame
            vertex_offset += local_vertex_count
            
    print(f"Successfully generated 4D mesh: {out_path}")



def inspect_mat(filepath):
    print(f"=== Inspecting file: {filepath} ===")
    
    # 1. Try standard scipy.io (MATLAB v4 through v7)
    try:
        mat = scipy.io.loadmat(filepath)
        print("\n[SUCCESS] Loaded using scipy.io (Standard v4-v7 format)")
        for k, v in mat.items():
            if not k.startswith('__'):
                print(f"  - Key: '{k}'")
                print(f"    Type: {type(v)}")
                if isinstance(v, np.ndarray):
                    print(f"    Shape: {v.shape}")
                    print(f"    Dtype: {v.dtype}")
                else:
                    print(f"    Value: {v}")
        return
    except Exception as e_scipy:
        print(f"\n[INFO] scipy.io failed (likely a v7.3 HDF5 file): {e_scipy}")

    # 2. Try h5py (MATLAB v7.3 format)
    try:
        print("\n--- Attempting load via h5py (MATLAB v7.3 format) ---")
        with h5py.File(filepath, 'r') as f:
            def recursive_inspect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  - Dataset: '{name}' | Shape: {obj.shape} | Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"  - Group: '{name}'")
            f.visititems(recursive_inspect)
    except Exception as e_h5:
        print(f"[ERROR] h5py also failed to read the file: {e_h5}")


VOLCANO_CMAP = LinearSegmentedColormap.from_list(
    "volcano_grey_red", 
    ["#1a1a1a", "#707070", "#d73027", "#f46d43", "#fee08b"]
)

def extract_arrays(d, prefix=""):
    """Recursively flattens nested MATLAB structs and HDF5 groups to find arrays."""
    arrays = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(k, str) and not k.startswith("__"):
                arrays.update(extract_arrays(v, k))
    elif isinstance(d, np.ndarray):
        if d.dtype.names is not None:
            val = d[0, 0] if d.size == 1 else d
            for name in val.dtype.names:
                arrays.update(extract_arrays(val[name], name))
        else:
            arrays[prefix] = d
    else:
        arrays[prefix] = d
    return arrays


def parse_mat_content(mat_dict):
    """Detects whether the loaded .mat dictionary contains a Mesh, Image/Volume, or 4D Volume."""
    flat_dict = extract_arrays(mat_dict)
    keys = list(flat_dict.keys())
    
    if not keys:
        return None, "No usable variables found in .mat file.", "unknown"

    vertices, faces = None, None
    v_keys = ["vertices", "vert", "nodes", "v", "pos", "p"]
    f_keys = ["faces", "triangles", "elements", "f", "cells", "t"]

    for k in keys:
        k_lower = k.lower()
        val = np.squeeze(flat_dict[k])
        
        if not isinstance(val, np.ndarray) or val.ndim != 2:
            continue
            
        is_v = k_lower in v_keys or 'vert' in k_lower or 'node' in k_lower
        is_f = k_lower in f_keys or 'face' in k_lower or 'elem' in k_lower or 'tri' in k_lower
        
        if is_v:
            vertices = val
        elif is_f:
            faces = val

    if vertices is not None and faces is not None:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)

        if vertices.shape[0] in [2, 3] and vertices.shape[1] > 3:
            vertices = vertices.T
        if faces.shape[0] in [3, 4] and faces.shape[1] > 4:
            faces = faces.T

        if vertices.shape[1] == 2:
            vertices = np.column_stack((vertices, np.zeros(vertices.shape[0])))

        if faces.max() == vertices.shape[0] or faces.min() >= 1:
            faces = faces - 1

        pad = np.full((faces.shape[0], 1), faces.shape[1], dtype=np.int64)
        pv_faces = np.hstack((pad, faces)).flatten()
        
        mesh = pv.PolyData(vertices, pv_faces)
        mesh.rotate_x(90, inplace=True)
        mesh.rotate_z(90, inplace=True)
        
        return (
            mesh,
            f"Mesh ({vertices.shape[0]} vertices, {faces.shape[0]} faces)",
            "mesh",
        )

    largest_key = max(keys, key=lambda k: np.asarray(flat_dict[k]).size)
    raw_arr = np.asarray(flat_dict[largest_key])
    
    # Squeeze singleton dimensions to avoid MATLAB structural nesting wrappers
    arr = np.squeeze(raw_arr)

    # If it's still 5D or higher due to cell arrays, try to find the actual numeric block
    if arr.ndim > 4:
        arr = np.squeeze(arr[0])

    if arr.ndim == 4:
        return (
            arr, 
            f"4D Volume array '{largest_key}' shape {arr.shape}", 
            "volume_4d"
        )
    elif arr.ndim == 3:
        grid = pv.ImageData(dimensions=arr.shape)
        grid.point_data["values"] = arr.flatten(order="F")  
        return (
            grid,
            f"3D Volume array '{largest_key}' shape {arr.shape}",
            "volume",
        )
    elif arr.ndim == 2:
        grid = pv.ImageData(dimensions=(arr.shape[0], arr.shape[1], 1))
        grid.point_data["values"] = arr.flatten(order="F")
        return grid, f"2D Image array '{largest_key}' shape {arr.shape}", "image"

    return (
        None,
        f"Could not parse shape/structure for variables (Detected dimensions: {arr.ndim}): {keys}",
        "unknown",
    )


def load_mat_file(filepath):
    """Loads MATLAB file handling both standard (v4-v7) and v7.3 (HDF5) formats."""
    try:
        return scipy.io.loadmat(filepath)
    except Exception:
        try:
            def load_h5(group):
                res = {}
                for k, v in group.items():
                    if isinstance(v, h5py.Group):
                        res[k] = load_h5(v)
                    else:
                        res[k] = np.array(v)
                return res
                
            with h5py.File(filepath, "r") as f:
                return load_h5(f)
        except Exception as e:
            print(f"Error reading {filepath.name}: {e}")
            return None


def _save_obj_manually(mesh, filepath):
    """Fallback manual OBJ writer robust to mixed mesh topologies."""
    with open(filepath, 'w') as f:
        for p in mesh.points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            
        faces = mesh.faces
        i = 0
        while i < len(faces):
            n = faces[i]
            face_indices = faces[i+1 : i+1+n]
            face_str = " ".join([str(idx + 1) for idx in face_indices])
            f.write(f"f {face_str}\n")
            i += n + 1


def mat_file_converter(path):
    """
    Processes either a single .mat file or a folder of .mat files. 
    - If a single .mat file is given and has a 4D array, it converts directly to 4D .tiff.
    - If a folder is given, saves meshes as .obj and stacks 3D volumes into a 4D .tiff.
    """
    input_path = Path(path)
    
    # --- ADDED LOGIC FOR A SINGLE FILE ---
    if input_path.is_file() and input_path.suffix.lower() == '.mat':
        print(f"Processing single file: '{input_path}'...")
        mat_dict = load_mat_file(input_path)
        if mat_dict is None:
            return
            
        obj, info_str, data_type = parse_mat_content(mat_dict)
        
        if data_type == "volume_4d":
            if np.issubdtype(obj.dtype, np.floating):
                obj = obj.astype(np.float32)
            elif obj.dtype == np.int64:
                obj = obj.astype(np.int32)
                
            tif_out_path = input_path.parent / f"{input_path.stem}_4D.tiff"
            tifffile.imwrite(str(tif_out_path), obj, imagej=True)
            print(f"Saved 4D Volume TIFF: {tif_out_path}")
        else:
            print(f"Expected a 4D volume, but found: {data_type}. Ignored.")
        return

    # --- ORIGINAL LOGIC FOR FOLDER PROCESSING ---
    if not input_path.is_dir():
        print(f"Invalid path provided: '{input_path}'")
        return
        
    mat_files = sorted(list(input_path.glob("*.mat")), key=natural_sort_key) 
    
    if not mat_files:
        print(f"No .mat files found in folder: '{input_path}'")
        return

    print(f"Processing {len(mat_files)} files in '{input_path}'...")
    
    volumes = []
    obj_dir = input_path.parent / "OBJ"
    
    for filepath in mat_files:
        mat_dict = load_mat_file(filepath)
        if mat_dict is None:
            continue
            
        obj, info_str, data_type = parse_mat_content(mat_dict)
        
        if data_type == "mesh":
            obj_dir.mkdir(parents=True, exist_ok=True)
            out_path = obj_dir / f"{filepath.stem}.obj"
            
            try:
                obj.save(str(out_path))
            except Exception:
                _save_obj_manually(obj, out_path)
            print(f"Saved Mesh: {out_path}")

        elif data_type == "volume":
            arr_3d = obj.point_data["values"].reshape(obj.dimensions, order="F")
            volumes.append(arr_3d)
            print(f"Loaded Volume for TIFF stacking: {filepath.name}")
        elif data_type == "volume_4d":
            print(f"Found 4D Volume in batch processing, ignoring for stack: {filepath.name}")

    if volumes:
        print("\nStacking sequential volumes into a 4D array...")
        volume_4d = np.stack(volumes, axis=0)

        if np.issubdtype(volume_4d.dtype, np.floating):
            volume_4d = volume_4d.astype(np.float32)
        elif volume_4d.dtype == np.int64:
            volume_4d = volume_4d.astype(np.int32)
        
        tif_out_path = input_path.parent / f"{input_path.name}_4D.tiff"
        tifffile.imwrite(str(tif_out_path), volume_4d, imagej=True)
        print(f"Saved 4D Volume TIFF: {tif_out_path}")
        
    print("Batch processing complete.")


class MatViewer:

    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        
        self.mat_files = sorted(
            list(self.folder_path.glob("*.mat")), 
            key=natural_sort_key
        )

        if not self.mat_files:
            raise FileNotFoundError(
                f"No .mat files found in folder: '{self.folder_path}'"
            )

        self.current_idx = 0
        self.plotter = pv.Plotter()

        self.plotter.add_key_event("Right", self.next_file)
        self.plotter.add_key_event("Left", self.prev_file)

        self.load_and_render()

    def load_and_render(self):
        self.plotter.clear()

        filepath = self.mat_files[self.current_idx]
        mat_dict = load_mat_file(filepath)

        header = (
            f"[{self.current_idx + 1}/{len(self.mat_files)}] {filepath.name}\n"
            f"Controls: Right Arrow (Next) | Left Arrow (Prev)"
        )

        if mat_dict is None:
            self.plotter.add_text(
                f"{header}\nStatus: Failed to load file.",
                position="upper_left",
                font_size=10,
            )
        else:
            obj, info_str, data_type = parse_mat_content(mat_dict)

            if obj is None:
                self.plotter.add_text(
                    f"{header}\nStatus: {info_str}",
                    position="upper_left",
                    font_size=10,
                )
            else:
                if data_type == "mesh":
                    self.plotter.add_mesh(
                        obj,
                        color="white",
                        show_edges=True,
                        edge_color="gray",
                        opacity=1.0,
                    )
                elif data_type == "volume":
                    self.plotter.add_volume(obj, cmap=VOLCANO_CMAP)
                elif data_type == "image":
                    self.plotter.add_mesh(obj, cmap=VOLCANO_CMAP)
                # --- ADDED SAFEGUARD FOR 4D VOLUMES ---
                elif data_type == "volume_4d":
                    self.plotter.add_text(
                        f"{header}\nStatus: 4D Volume loaded. Cannot render directly in 3D viewer.",
                        position="upper_left",
                        font_size=10,
                    )
                    return

                self.plotter.add_text(
                    f"{header}\nData: {info_str}",
                    position="upper_left",
                    font_size=10,
                )

        self.plotter.reset_camera()
        self.plotter.render()

    def next_file(self):
        self.current_idx = (self.current_idx + 1) % len(self.mat_files)
        self.load_and_render()

    def prev_file(self):
        self.current_idx = (self.current_idx - 1) % len(self.mat_files)
        self.load_and_render()

    def show(self):
        self.plotter.show()  