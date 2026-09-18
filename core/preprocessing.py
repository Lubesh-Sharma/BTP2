import os
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
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

def compute_hks_features(VPos, Elements, k_indices, t, neigvecs=300, return_vecs=False):
    """
    Computes HKS features efficiently by performing dense eigenvalue decomposition natively on the GPU.
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
    
    hks_np = HKS_matrix.cpu().numpy()
    if return_vecs:
        return hks_np, vecs.cpu().numpy()
    return hks_np

def compute_chirality_features(VPos, Elements, vecs, pairs=((1, 2), (1, 3), (2, 3))):
    """
    Computes Laplace-Beltrami Eigenfunction Chirality / Cross-Gradient features
    (Property A) on a triangle surface mesh.

    For each triangle face f:
      1. Gradient of eigenfunction phi_i on face f:
         grad(phi_i)|_f = (1 / (2 * Area_f)) * sum_{k=1}^3 phi_i(v_k) (n_f x e_k)
      2. Face chirality:
         chi_{i,j}|_f = n_f . (grad(phi_i)|_f x grad(phi_j)|_f)
      3. Project face chirality back to vertices using area-weighted accumulation.
      4. Scale-normalize each channel to [-1, 1].

    Under reflection symmetry across the sagittal plane (an orientation-reversing map):
      chi_{i,j}(sigma(x)) = - chi_{i,j}(x)
    This breaks the bilateral symmetry ambiguity inherent to HKS.

    Args:
        VPos: np.ndarray [N, 3] vertex positions.
        Elements: np.ndarray [M, 3] triangle face vertex indices.
        vecs: np.ndarray [N, K] LBO eigenvectors (sorted by increasing eigenvalue magnitude).
        pairs: tuple of pairs of eigenvector indices (e.g. (1, 2), (1, 3), (2, 3)).

    Returns:
        chirality: np.ndarray [N, len(pairs)] float32 chirality descriptors in [-1, 1].
    """
    N = VPos.shape[0]
    n_pairs = len(pairs)
    if Elements.shape[1] != 3 or vecs.shape[1] < 3:
        return np.zeros((N, n_pairs), dtype=np.float32)

    # Face vertex positions
    v1 = VPos[Elements[:, 0]]
    v2 = VPos[Elements[:, 1]]
    v3 = VPos[Elements[:, 2]]

    # Directed edges opposite to vertices 1, 2, 3
    e1 = v3 - v2  # opposite v1
    e2 = v1 - v3  # opposite v2
    e3 = v2 - v1  # opposite v3

    # Face normal and area
    face_cross = np.cross(v2 - v1, v3 - v1)
    face_areas = 0.5 * np.linalg.norm(face_cross, axis=1)
    face_areas_safe = np.maximum(face_areas, 1e-12)
    face_normals = face_cross / (2.0 * face_areas_safe[:, None])

    # Rotated in-plane vectors: n_f x e_k
    n_cross_e1 = np.cross(face_normals, e1)
    n_cross_e2 = np.cross(face_normals, e2)
    n_cross_e3 = np.cross(face_normals, e3)

    # Helper to compute constant gradient of a scalar field across all faces [M, 3]
    def face_gradient(phi):
        p1 = phi[Elements[:, 0], None]
        p2 = phi[Elements[:, 1], None]
        p3 = phi[Elements[:, 2], None]
        return (p1 * n_cross_e1 + p2 * n_cross_e2 + p3 * n_cross_e3) / (2.0 * face_areas_safe[:, None])

    # Accumulate vertex total area for area-weighted averaging
    vert_areas = np.zeros(N, dtype=np.float64)
    for k in range(3):
        np.add.at(vert_areas, Elements[:, k], face_areas)
    vert_areas_safe = np.maximum(vert_areas, 1e-12)

    chirality = np.zeros((N, n_pairs), dtype=np.float32)

    # Cache gradients of requested eigenvectors
    unique_indices = sorted(list({idx for pair in pairs for idx in pair}))
    grad_cache = {}
    for idx in unique_indices:
        if idx < vecs.shape[1]:
            grad_cache[idx] = face_gradient(vecs[:, idx])

    for p_idx, (i, j) in enumerate(pairs):
        if i in grad_cache and j in grad_cache:
            grad_i = grad_cache[i]
            grad_j = grad_cache[j]
            # Face chirality = n_f . (grad_i x grad_j)
            face_chi = np.sum(face_normals * np.cross(grad_i, grad_j), axis=1)

            # Area-weighted vertex accumulation
            weighted_face_chi = face_chi * face_areas
            vert_chi = np.zeros(N, dtype=np.float64)
            for k in range(3):
                np.add.at(vert_chi, Elements[:, k], weighted_face_chi)
            vert_chi = vert_chi / vert_areas_safe

            # Robust scale normalization to [-1, 1] using 99th percentile
            scale_val = np.percentile(np.abs(vert_chi), 99.0)
            if scale_val < 1e-8:
                scale_val = np.max(np.abs(vert_chi))
            if scale_val > 1e-8:
                vert_chi = np.clip(vert_chi / scale_val, -1.0, 1.0)

            chirality[:, p_idx] = vert_chi.astype(np.float32)

    return chirality

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
    features = compute_hks_features(VPos, Elements, fps_idx, t, neigvecs=neigvecs, return_vecs=False)
    features = normalize_descriptors(features)
    features = np.log(np.abs(features) + 1e-10)

    # ---------------------------------------------------------------
    # Append normalized XYZ coordinates as the last 3 feature dims.
    # Feature layout: [hks_0, ..., hks_{k-1}, x, y, z] (50 HKS + 3 XYZ = 53)
    # ---------------------------------------------------------------
    pos_min = VPos.min(axis=0)
    pos_max = VPos.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range == 0] = 1.0          # avoid divide-by-zero on flat axis
    pos_norm = (VPos - pos_min) / pos_range  # [N, 3], each axis in [0, 1]
    features = np.concatenate([features, pos_norm], axis=1)  # [N, k + 3]

    mat_path = os.path.join(output_dir, "matrix_" + base_name + ".txt")
    np.savetxt(mat_path, features)
    print(f"[{obj_path}] Feature dim after append: {features.shape[1]} ({features.shape[1]-3} HKS + 3 XYZ)")
    return VPos, Elements, features, fps_idx
