import os
import numpy as np
import trimesh
import scipy.io as sio
import gdist

def compute_geodesic_matrix(vertices, faces):
    V = vertices.shape[0]
    dist_matrix = np.zeros((V, V), dtype=np.float32)

    for i in range(V):
        # compute geodesic distance from vertex i to all others
        dist = gdist.compute_gdist(
            vertices.astype(np.float64),
            faces.astype(np.int32),
            source_indices=np.array([i], dtype=np.int32)
        )
        dist_matrix[i] = dist

        if i % 1000 == 0: # Increased interval for cleaner logs
            print(f"  > Vertex {i}/{V}")

    return dist_matrix

def process_last_n_files(input_dir, output_dir, n=20):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Get all .off files and sort them alphabetically
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.off')])
    
    # 2. Slice the list to get min(folder size, last_n_files)
    # Using negative indexing [-n:] automatically handles cases where len < n,
    # comment out this line in the case of the shrec_19
    files_to_process = all_files[-n:]

    print(f"Found {len(all_files)} files. Processing the last {len(files_to_process)}.")

    for f in files_to_process:
        mesh_path = os.path.join(input_dir, f)
        name = os.path.splitext(f)[0]
        out_path = os.path.join(output_dir, name + ".mat")

        # Skip if file already exists (optional, but recommended)
        if os.path.exists(out_path):
            print(f"Skipping {name} - already exists at {out_path}")
            continue

        print(f"\n--- Processing {name} ---")

        mesh = trimesh.load(mesh_path, process=False)
        vertices = mesh.vertices
        faces = mesh.faces

        dist_matrix = compute_geodesic_matrix(vertices, faces)

        sio.savemat(out_path, {'dist': dist_matrix})
        print(f"Successfully saved: {out_path}")

# ====== USAGE ======
# Update these paths to point to your FAUST or SCAPE data
input_folder = "./input/SCAPE/off"      
output_folder = "./input/SCAPE/mat"    

# This will only process the last 20 files in the folder
process_last_n_files(input_folder, output_folder, n=20)