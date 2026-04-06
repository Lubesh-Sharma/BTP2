import os
import numpy as np
from scipy.sparse.linalg import eigsh
from utils.mesh import load_obj, load_off
from utils.hks import get_graph_laplacian, get_cotan_laplacian
from utils.files import save_pp_file

def compute_fps(VPos, k):
    """
    Performs Farthest Point Sampling (FPS) smoothly on the GPU via PyTorch.
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VPos_tensor = torch.tensor(VPos, dtype=torch.float32, device=device)
    
    N = VPos_tensor.shape[0]
    if k > N: k = N
    selected_indices = [0]
    
    dists = torch.norm(VPos_tensor - VPos_tensor[0], dim=1)
    
    for _ in range(1, k):
        next_idx = torch.argmax(dists).item()
        selected_indices.append(next_idx)
        new_dists = torch.norm(VPos_tensor - VPos_tensor[next_idx], dim=1)
        dists = torch.minimum(dists, new_dists)
        
    return np.array(selected_indices)

def compute_hks_features(VPos, Elements, k_indices, t, neigvecs=300):
    """
    Computes HKS features efficiently by performing dense eigenvalue decomposition natively on the GPU.
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    is_graph = (Elements.shape[1] == 2)
    if is_graph: L = get_graph_laplacian(VPos, Elements)
    else: L = get_cotan_laplacian(VPos, Elements)
    
    n_eigs = min(VPos.shape[0] - 1, neigvecs)
    
    # Convert sparse Laplacian directly to dense GPU tensor
    L_dense = torch.tensor(L.toarray(), dtype=torch.float32, device=device)
    vals, vecs = torch.linalg.eigh(L_dense)
    
    # Sort purely by absolute magnitude to perfectly simulate ARPACK's 'which=SM' (Smallest Magnitude)
    abs_vals = torch.abs(vals)
    sorted_indices = torch.argsort(abs_vals)
    vals = vals[sorted_indices][:n_eigs]
    vecs = vecs[:, sorted_indices][:, :n_eigs]
    
    # Perform math on GPU
    vals_t = torch.exp(-vals * t)
    ScaledVecs = vecs * vals_t.unsqueeze(0)
    
    k_idx_tensor = torch.tensor(k_indices, dtype=torch.long, device=device)
    Sources = vecs[k_idx_tensor, :]
    
    # Fast matrix multiplication on the GPU
    HKS_matrix = torch.matmul(ScaledVecs, Sources.transpose(0, 1))
    
    return HKS_matrix.cpu().numpy()

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

def process_geometry(obj_path, k, t, neigvecs=300, output_dir="output"):
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
    features = compute_hks_features(VPos, Elements, fps_idx, t, neigvecs=neigvecs)
    features = normalize_descriptors(features)
    features = np.log(np.abs(features) + 1e-10)
    mat_path = os.path.join(output_dir, "matrix_" + base_name + ".txt")
    np.savetxt(mat_path, features)
    return VPos, Elements, features, fps_idx
