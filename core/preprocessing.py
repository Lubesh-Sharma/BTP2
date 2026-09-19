import os
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from utils.mesh import load_obj, load_off
from utils.hks import get_graph_laplacian, get_cotan_laplacian
from utils.files import save_pp_file

def compute_fps(VPos, k):
    """
    Performs Farthest Point Sampling (FPS) smoothly on the GPU via PyTorch.
    """
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

def compute_point_normals(VPos, k=15):
    """
    Computes outward-oriented surface normal vectors on an unstructured 3D point cloud
    using local covariance analysis with BFS normal propagation.
    
    1. Computes local covariance eigenvectors on k-NN neighborhoods (k=15).
    2. Seeds outward direction at the extreme point furthest from the shape center of mass.
    3. Propagates orientation across the k-NN graph ensuring neighboring normals align (n_i . n_j > 0).
    4. Verifies global outwardness.
    
    Guarantees seamless outward orientation across limbs (arms/legs/quadruped legs),
    torso, and head without flipping.
    """
    from collections import deque
    N = VPos.shape[0]
    k_nn = min(k, N)
    nn = NearestNeighbors(n_neighbors=k_nn).fit(VPos)
    _, idxs = nn.kneighbors(VPos)
    
    # [N, k, 3]
    neighbors = VPos[idxs]
    means = neighbors.mean(axis=1, keepdims=True)
    diffs = neighbors - means
    
    # Covariance matrices: [N, 3, 3] = diffs.T @ diffs / k
    covs = np.matmul(diffs.transpose(0, 2, 1), diffs) / k_nn
    
    # Eigenvalues and eigenvectors for symmetric 3x3 matrices (sorted ascending)
    _, vecs = np.linalg.eigh(covs)
    normals = vecs[:, :, 0].copy()  # [N, 3] Smallest eigenvector = normal axis
    
    # Normalize initial normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norms
    
    # Global centroid and radial distances
    centroid = VPos.mean(axis=0, keepdims=True)
    dists_from_center = np.linalg.norm(VPos - centroid, axis=1)
    
    # Seed point: the extreme point furthest from centroid (e.g. top of head or fingertip)
    # The normal at the extreme point unequivocally points outward (away from centroid)
    seed_idx = int(np.argmax(dists_from_center))
    seed_vec = VPos[seed_idx] - centroid[0]
    if np.dot(normals[seed_idx], seed_vec) < 0:
        normals[seed_idx] = -normals[seed_idx]
        
    # BFS propagation across k-NN graph to orient all normals consistently
    visited = np.zeros(N, dtype=bool)
    visited[seed_idx] = True
    queue = deque([seed_idx])
    
    # Build adjacency list from k-NN
    adj = idxs[:, 1:]  # exclude self
    
    while queue:
        curr = queue.popleft()
        n_curr = normals[curr]
        for neighbor in adj[curr]:
            if not visited[neighbor]:
                # If neighbor normal opposes current normal, flip it
                if np.dot(normals[neighbor], n_curr) < 0:
                    normals[neighbor] = -normals[neighbor]
                visited[neighbor] = True
                queue.append(neighbor)
                
    # Handle any disconnected components
    unvisited = np.where(~visited)[0]
    for idx in unvisited:
        ref_vec = VPos[idx] - centroid[0]
        if np.dot(normals[idx], ref_vec) < 0:
            normals[idx] = -normals[idx]
            
    # Final global check: total outward flux must be positive
    outward_ref = VPos - centroid
    if np.sum(normals * outward_ref) < 0:
        normals = -normals
        
    return normals.astype(np.float32)

def compute_hks_wks_features(VPos, Elements, k_indices, t, neigvecs=300, n_wks=50, sigma=0.06):
    """
    Computes both Heat Kernel Signatures (HKS) and Wave Kernel Signatures (WKS)
    efficiently on the GPU in a single eigen-decomposition pass.

    - HKS captures smooth diffusion / geometry at scale t.
    - WKS evaluates energy frequencies, separating thin extremities (hands) from thick limbs (legs).

    Args:
        VPos: [N, 3] Point cloud coordinates.
        Elements: [M, 2 or 3] Connectivity (Lines or Faces).
        k_indices: [k] Anchor point indices for HKS.
        t: float, diffusion time parameter for HKS.
        neigvecs: int, number of eigenvectors to use.
        n_wks: int, number of WKS energy bands.
        sigma: float, WKS Gaussian variance width.

    Returns:
        hks: np.ndarray [N, len(k_indices)]
        wks: np.ndarray [N, n_wks]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    is_graph = (Elements.shape[1] == 2)
    if is_graph: L = get_graph_laplacian(VPos, Elements)
    else: L = get_cotan_laplacian(VPos, Elements)
    
    n_eigs = min(VPos.shape[0] - 1, neigvecs)
    
    try:
        # Convert sparse Laplacian directly to dense GPU tensor
        L_dense = torch.tensor(L.toarray(), dtype=torch.float32, device=device)
        vals, vecs = torch.linalg.eigh(L_dense)
    except Exception:
        # Fallback to SciPy ARPACK sparse solver
        vals_np, vecs_np = eigsh(L, k=n_eigs, which='SM')
        vals = torch.tensor(vals_np, dtype=torch.float32, device=device)
        vecs = torch.tensor(vecs_np, dtype=torch.float32, device=device)
    
    # Sort by absolute magnitude
    abs_vals = torch.abs(vals)
    sorted_indices = torch.argsort(abs_vals)
    vals = vals[sorted_indices][:n_eigs]
    vecs = vecs[:, sorted_indices][:, :n_eigs]
    
    # 1. HKS computation on GPU
    vals_t = torch.exp(-vals * t)
    ScaledVecs = vecs * vals_t.unsqueeze(0)
    k_idx_tensor = torch.tensor(k_indices, dtype=torch.long, device=device)
    Sources = vecs[k_idx_tensor, :]
    HKS_matrix = torch.matmul(ScaledVecs, Sources.transpose(0, 1))
    hks_np = HKS_matrix.cpu().numpy()
    
    # 2. WKS computation on GPU (using positive eigenvalues)
    pos_mask = vals > 1e-6
    if pos_mask.sum() > 5:
        vals_pos = vals[pos_mask]
        vecs_pos = vecs[:, pos_mask]
        log_vals = torch.log(vals_pos)
        e_steps = torch.linspace(log_vals[0], log_vals[-1], n_wks, device=device)
        diff = e_steps.unsqueeze(1) - log_vals.unsqueeze(0)
        weights = torch.exp(- (diff ** 2) / (2.0 * (sigma ** 2)))
        wks_gpu = torch.matmul(vecs_pos ** 2, weights.transpose(0, 1))
        # Normalize per vertex so row sums to 1
        wks_gpu = wks_gpu / (torch.sum(wks_gpu, dim=1, keepdim=True) + 1e-12)
        wks_np = wks_gpu.cpu().numpy()
    else:
        wks_np = np.zeros((VPos.shape[0], n_wks), dtype=np.float32)
        
    return hks_np, wks_np

def normalize_pc(points):
    """
    Zero-centers and uniformly rescales a point cloud by its maximum radius.
    Preserves true anatomical coordinate centers (z=0 is coronal center).
    """
    centroid = np.mean(points, axis=0, keepdims=True)
    points_centered = points - centroid
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    if scale > 0:
        points_centered = points_centered / scale
    return points_centered
    
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
    Orchestrates geometric preprocessing for 3D point clouds without mesh dependence:
      1. FPS sampling of k anchor points.
      2. Joint HKS (50) + WKS (50) multi-energy spectral feature extraction.
      3. Outward-oriented PCA point normals (3) to break front-back reflection.
      4. Centered & radially normalized coordinates (3).
    
    Combined Feature Layout [N, 106]:
      [0 : 50]    -> HKS features (50)
      [50 : 100]  -> WKS features (50)
      [100 : 103] -> Centered XYZ coordinates (3)
      [103 : 106] -> Outward PCA Normals (3)
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
    
    print(f"[{obj_path}] Computing HKS & WKS (t={t})...")
    hks, wks = compute_hks_wks_features(VPos, Elements, fps_idx, t, neigvecs=neigvecs, n_wks=50)
    
    # Scale-harmonize HKS to [0, 1] per channel (prevents log scale from drowning other features)
    hks_min = hks.min(axis=0, keepdims=True)
    hks_max = hks.max(axis=0, keepdims=True)
    hks = (hks - hks_min) / (hks_max - hks_min + 1e-8)
    
    # Scale-harmonize WKS to [0, 1] per channel (elevates limb-distinguishing frequencies)
    wks_min = wks.min(axis=0, keepdims=True)
    wks_max = wks.max(axis=0, keepdims=True)
    wks = (wks - wks_min) / (wks_max - wks_min + 1e-8)

    print(f"[{obj_path}] Computing outward PCA normals...")
    normals = compute_point_normals(VPos, k=15)
    
    # Centered and radially normalized coordinates [-1, 1]
    pos_norm = normalize_pc(VPos.copy()).astype(np.float32)

    # Combine into 106-dim feature matrix with harmonized scales:
    # [HKS in [0, 1], WKS in [0, 1], pos_norm in [-1, 1], normals in [-1, 1]]
    features = np.concatenate([hks, wks, pos_norm, normals], axis=1).astype(np.float32)

    mat_path = os.path.join(output_dir, "matrix_" + base_name + ".txt")
    np.savetxt(mat_path, features)
    print(f"[{obj_path}] Feature dim: {features.shape[1]} (50 HKS + 50 WKS + 3 XYZ + 3 Normals)")
    return VPos, Elements, features, fps_idx
