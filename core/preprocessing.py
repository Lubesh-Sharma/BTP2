import os
import numpy as np
from scipy.sparse.linalg import eigsh
from utils.mesh import load_obj, load_off
from utils.hks import get_graph_laplacian, get_cotan_laplacian
from utils.files import save_pp_file

def compute_fps(VPos, k):
    """
    Performs Farthest Point Sampling (FPS).
    """
    N = VPos.shape[0]
    if k > N: k = N
    selected_indices = [0]
    dists = np.linalg.norm(VPos - VPos[0], axis=1)
    for _ in range(1, k):
        next_idx = np.argmax(dists)
        selected_indices.append(next_idx)
        new_dists = np.linalg.norm(VPos - VPos[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    return np.array(selected_indices)

def compute_hks_features(VPos, Elements, k_indices, t, neigvecs=300):
    """
    Computes HKS features.
    """
    is_graph = (Elements.shape[1] == 2)
    if is_graph: L = get_graph_laplacian(VPos, Elements)
    else: L = get_cotan_laplacian(VPos, Elements)
    n_eigs = min(VPos.shape[0] - 1, neigvecs)
    vals, vecs = eigsh(L, k=n_eigs, which='SM')
    

    # norms = np.linalg.norm(vecs, axis=0)
    # print("Eigenvector norms:", norms[:10])

    vals_t = np.exp(-vals * t)
    ScaledVecs = vecs * vals_t[None, :]
    Sources = vecs[k_indices, :]
    HKS_matrix = ScaledVecs.dot(Sources.T)
    # print(HKS_matrix[1:].shape)
    # exit()
    return HKS_matrix

def normalize_pc(points):
    """
    Centers and rescales a point cloud.
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0: points /= scale
    return points
    
def normalize_descriptors(features, eps=1e-12):
    """
    L2-normalize each feature channel over vertices
    Equivalent to what Functional Maps do.
    """
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    features = features / (norms + eps)
    return features

def process_geometry(obj_path, k, t, output_dir="output"):
    """
    Orchestrates the geometric preprocessing.
    """
    print(f"[{obj_path}] Loading...")
    input_path = obj_path
    if not os.path.exists(input_path) and os.path.exists(os.path.join("input", obj_path)):
        input_path = os.path.join("input", obj_path)
    
    if input_path.endswith('.obj'): VPos, _, Elements = load_obj(input_path)
    else: VPos, _, Elements = load_off(input_path)
    
    print(f"[{obj_path}] Running FPS (k={k})...")
    fps_idx = compute_fps(VPos, k)
    
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    pp_path = os.path.join(output_dir, base_name + ".pp")
    save_pp_file(pp_path, VPos, fps_idx)
    
    print(f"[{obj_path}] Computing HKS (t={t})...")
    features = compute_hks_features(VPos, Elements, fps_idx, t)
    features = normalize_descriptors(features)
    features = np.log(np.abs(features) + 1e-10)
    mat_path = os.path.join(output_dir, "matrix_" + base_name + ".txt")
    np.savetxt(mat_path, features)
    return VPos, Elements, features, fps_idx
