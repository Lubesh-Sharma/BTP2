import os
import glob
import numpy as np
from pyFM.mesh import TriMesh

def get_visu_colors(vertices):
    min_coord = np.min(vertices, axis=0, keepdims=True)
    max_coord = np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    # Return as 0-255 integers
    return (cmap * 255).astype(int)

def save_colored_off(filename, vertices, faces, colors):
    """
    Saves a mesh in COFF format (Colored OFF) which is standard for MeshLab.
    """
    with open(filename, 'w') as f:
        f.write("COFF\n")
        f.write(f"{vertices.shape[0]} {faces.shape[0]} 0\n")
        
        # Write vertices with RGB values
        for i in range(vertices.shape[0]):
            v = vertices[i]
            c = colors[i]
            f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]} 255\n") # 255 is Alpha
            
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def process_all_p2p(p2p_folder, mesh_folder, output_base):
    os.makedirs(output_base, exist_ok=True)
    p2p_files = glob.glob(os.path.join(p2p_folder, "p2p_*.txt"))
    
    if not p2p_files:
        print("No p2p files found. Check your paths!")
        return

    for p2p_path in p2p_files:
        filename = os.path.basename(p2p_path).replace(".txt", "")
        parts = filename.split("_to_")
        src_name = parts[0].replace("p2p_", "")
        tgt_name = parts[1]
        
        path1 = os.path.join(mesh_folder, f"{src_name}.off")
        path2 = os.path.join(mesh_folder, f"{tgt_name}.off")

        if not os.path.exists(path1) or not os.path.exists(path2):
            continue

        # Load Meshes
        mesh1 = TriMesh(path1, center=True, area_normalize=True)
        mesh2 = TriMesh(path2, center=True, area_normalize=True)

        # Load Mapping
        p2p_pairs = np.loadtxt(p2p_path, dtype=int)
        tgt_idx = p2p_pairs[:, 1]
        tgt_idx = np.clip(tgt_idx, 0, mesh2.vertlist.shape[0] - 1)

        # Generate Colors
        colors_src = get_visu_colors(mesh1.vertlist)
        colors_tgt = colors_src[tgt_idx]

        # Directory Management
        save_dir = os.path.join(output_base, f"{src_name}_to_{tgt_name}")
        os.makedirs(save_dir, exist_ok=True)

        # Save using the custom colored OFF writer
        save_colored_off(os.path.join(save_dir, f"{src_name}_colored.off"), 
                         mesh1.vertlist, mesh1.facelist, colors_src)
        
        save_colored_off(os.path.join(save_dir, f"{tgt_name}_colored.off"), 
                         mesh2.vertlist, mesh2.facelist, colors_tgt)

        print(f"Exported colored meshes for {src_name} -> {tgt_name}")

# --- Paths ---
P2P_RESULTS_DIR = "./p2p_results_st_te_SHREC/"
MESH_INPUT_DIR = "./input/SHREC/off/"
OUTPUT_DIR = "./colored_output/SHREC/"

process_all_p2p(P2P_RESULTS_DIR, MESH_INPUT_DIR, OUTPUT_DIR)