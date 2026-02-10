import numpy as np
import argparse
import os
from scipy.sparse.linalg import eigsh
from hks import get_graph_laplacian, get_cotan_laplacian
from trimesh import load_obj, load_off

def compute_fps(VPos, k):
    """
    Selects k points using Farthest Point Sampling (FPS) with Euclidean distance.
    This ensures a diverse coverage of the shape surface.
    
    Args:
        VPos (np.ndarray): [N, 3] Vertex positions.
        k (int): Number of points to select.
        
    Returns:
        np.ndarray: [k] Indices of selected points.
    """
    N = VPos.shape[0]
    if k > N:
        raise ValueError(f"Cannot sample {k} points from {N} vertices")
        
    selected_indices = [0] # Start with first vertex (arbitrary)
    
    # Initialize distances from the first point
    # VPos is (N, 3)
    dists = np.linalg.norm(VPos - VPos[0], axis=1)
    
    for _ in range(1, k):
        # Select the point with the maximum minimum distance to the set
        next_idx = np.argmax(dists)
        selected_indices.append(next_idx)
        
        # Update distances: new distance is min(old_dist, dist_to_new_point)
        new_dists = np.linalg.norm(VPos - VPos[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
        
    return np.array(selected_indices)

def save_pp_file(filename, VPos, indices):
    """
    Saves selected points as a MeshLab .pp (PickedPoints) file.
    
    Args:
        filename (str): Output path.
        VPos (np.ndarray): [N, 3] Vertices.
        indices (np.ndarray): [k] Indices of points to save.
    """
    with open(filename, 'w') as f:
        f.write('<!DOCTYPE PickedPoints>\n')
        f.write('<PickedPoints>\n')
        for i, idx in enumerate(indices):
            p = VPos[idx]
            # Write point with name as 1-based index (1...k)
            f.write(f' <point x="{p[0]}" y="{p[1]}" z="{p[2]}" active="1" name="{i+1}"/>\n')
        f.write('</PickedPoints>\n')

def process_shape(obj_path, k, t, neigvecs):
    """
    Processes a single shape to extract HKS features seeded at FPS points.
    
    Steps:
    1. Load mesh.
    2. Perform FPS (k points).
    3. Compute Spectrum (Eigen decomposition).
    4. Compute HKS relative to k source(FPS) points.
    5. Save results to 'matrix_*.txt' and '*.pp'.
    
    Args:
        obj_path (str): Input path.
        k (int): Number of FPS points.
        t (float): HKS time.
        neigvecs (int): Number of eigenvectors used.
    """
    print(f"Processing {obj_path}...")
    input_path = obj_path
    if not os.path.exists(input_path) and os.path.exists(os.path.join("input", obj_path)):
        input_path = os.path.join("input", obj_path)
        
    if input_path.endswith('.obj'):
        VPos, VColors, Elements = load_obj(input_path)
    else:
        # Assuming .off
        VPos, VColors, Elements = load_off(input_path)
        
    print(f"  Loaded {len(VPos)} vertices and {len(Elements)} elements.")
    
    # 1. FPS Selection
    print(f"  Selecting {k} points via FPS...")
    fps_indices = compute_fps(VPos, k)
    print(f"  Selected indices: {fps_indices}")
    
    # 2. Compute Laplacian
    print("  Computing Laplacian...")
    is_graph = (Elements.shape[1] == 2)
    if is_graph:
        L = get_graph_laplacian(VPos, Elements)
    else:
        L = get_cotan_laplacian(VPos, Elements)
         
    # 3. Eigen decomposition
    # Ensure we don't ask for more eigenvectors than vertices
    n_eigs = min(VPos.shape[0] - 1, neigvecs)
    print(f"  Computing {n_eigs} eigenvectors...")
    
    # Use small sigma to avoid singularity at 0
    vals, vecs = eigsh(L, k=n_eigs, which='LM', sigma=1e-8)
    
    # 4. Compute HKS vectors from the k source points
    # HKS(x, s) = sum_i exp(-lambda_i * t) * phi_i(x) * phi_i(s)
    # Resulting matrix should be (N, k), where N is num vertices.
    # Each row x contains the heat kernel signature values from sources s_1...s_k
    
    print("  Computing HKS matrix...")
    vals_t = np.exp(-vals * t) # (M,)
    
    # Scale eigenvectors by sqrt(exp(-lambda*t)) effectively, 
    # but since it's symmetric product phi_i(x)*phi_i(s), we can just scale one side by exp or both by sqrt.
    # ScaledVecs = vecs * vals_t[None, :] corresponds to phi_i(x) * exp(-lambda_i * t)
    # Then dot with unscaled vecs[s, :] gives sum (phi_i(x) * exp...) * phi_i(s)
    
    ScaledVecs = vecs * vals_t[None, :]
    Sources = vecs[fps_indices, :]
    HKS_matrix = ScaledVecs.dot(Sources.T)
    
    # 5. Save outputs
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    txt_filename = os.path.join("output", f"matrix_{base_name}.txt")
    pp_filename = os.path.join("output", f"{base_name}.pp")
    
    print(f"  Saving matrix to {txt_filename}...")
    np.savetxt(txt_filename, HKS_matrix)
    
    print(f"  Saving picked points to {pp_filename}...")
    save_pp_file(pp_filename, VPos, fps_indices)
    print("  Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute HKS from k FPS points on two shapes.")
    parser.add_argument("--shape1", type=str, required=True, help="Path to first shape file")
    parser.add_argument("--shape2", type=str, required=True, help="Path to second shape file")
    parser.add_argument("--k", type=int, required=True, help="Number of points to sample")
    parser.add_argument("--t", type=float, required=True, help="Time parameter t for HKS")
    parser.add_argument("--neigvecs", type=int, default=200, help="Number of eigenvectors to compute")
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs("output", exist_ok=True)
    
    process_shape(args.shape1, args.k, args.t, args.neigvecs)
    process_shape(args.shape2, args.k, args.t, args.neigvecs)
