"""
utils/dino_utils.py

Utilities to:
  1. Render a 3D mesh to a 2D image (using pyrender + trimesh).
  2. Extract per-patch DINOv2 features from that image.
  3. Project 3D vertices back onto the 2D image and assign each vertex
     the DINOv2 feature of the patch it falls in.

Requirements:
    pip install pyrender trimesh torch torchvision

Note: pyrender requires a display. For headless (SSH) servers, set:
    export PYOPENGL_PLATFORM=osmesa
    pip install pyopengl osmesa   (or use egl: PYOPENGL_PLATFORM=egl)
"""

import numpy as np
import torch
from torchvision import transforms

# -----------------------------------------------------------------------
# DINOv2 Model (loaded once as a module-level singleton)
# -----------------------------------------------------------------------
_dino_model = None
_PATCH_SIZE = 14         # DINOv2 ViT-B/14 patch size
_IMAGE_SIZE = 518        # must be divisible by 14; 518 = 37×14
_DINO_DIM = 768          # vitb14 feature dim; change to 1024 for vitl14

_dino_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_dino_model(variant='dinov2_vitb14_reg', device=None):
    """
    Loads and caches the DINOv2 model.

    Args:
        variant: One of 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14',
                 'dinov2_vitb14_reg', 'dinov2_vitl14_reg'  (reg = with registers)
        device: torch device. Defaults to CUDA if available.

    Returns:
        model: DINOv2 model in eval mode.
        device: The device the model is on.
    """
    global _dino_model
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if _dino_model is None:
        print(f"[DINOv2] Loading model: {variant} on {device}...")
        _dino_model = torch.hub.load('facebookresearch/dinov2', variant)
        _dino_model = _dino_model.to(device).eval()
        print(f"[DINOv2] Model loaded.")
    return _dino_model, device


# -----------------------------------------------------------------------
# Step 1: Render mesh to 2D image using pyrender
# -----------------------------------------------------------------------
def render_mesh_to_image(VPos, ITris, image_size=_IMAGE_SIZE,
                         use_normals_as_color=True):
    """
    Renders a 3D mesh to a 2D RGB image using pyrender.

    For untextured meshes (like FAUST/SCAPE), we color the mesh by its
    surface normals mapped to RGB — this gives DINOv2 much more signal
    than a plain gray surface.

    Args:
        VPos:   np.ndarray [N, 3] vertex positions.
        ITris:  np.ndarray [M, 3] triangle face indices.
        image_size: int, rendered image size (square).
        use_normals_as_color: bool. If True, color vertices by surface
                              normals (normal * 0.5 + 0.5 → RGB).

    Returns:
        color_img: np.ndarray [H, W, 3] uint8 rendered image.
        depth_img: np.ndarray [H, W] float32 depth buffer.
        camera_pose: np.ndarray [4, 4] camera-to-world matrix.
        fov_y: float, vertical field of view in radians.
    """
    import trimesh
    import pyrender

    mesh_tri = trimesh.Trimesh(vertices=VPos, faces=ITris, process=False)

    if use_normals_as_color:
        # Map vertex normals to RGB: normal ∈ [-1,1]³ → [0,1]³
        mesh_tri.vertex_normals  # triggers computation
        normals_rgb = (mesh_tri.vertex_normals * 0.5 + 0.5)
        normals_rgb = np.clip(normals_rgb, 0, 1)
        vertex_colors = (normals_rgb * 255).astype(np.uint8)
        mesh_tri.visual.vertex_colors = vertex_colors

    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri, smooth=False)

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh_pr)

    # Place camera to frame the mesh from the front.
    # Center mesh at origin, move camera back along +Z.
    centroid = VPos.mean(axis=0)
    extent   = np.max(np.linalg.norm(VPos - centroid, axis=1))
    fov_y    = np.pi / 3.0   # 60°
    dist     = extent / np.tan(fov_y / 2) * 1.5  # 1.5x for padding

    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[:3, 3] = centroid + np.array([0.0, 0.0, dist])

    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)

    # Directional light from camera
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    color_img, depth_img = renderer.render(scene)
    renderer.delete()

    return color_img, depth_img, camera_pose, fov_y


# -----------------------------------------------------------------------
# Step 2: Extract DINOv2 patch features from image
# -----------------------------------------------------------------------
def extract_dino_patch_features(color_img, variant='dinov2_vitb14_reg',
                                image_size=_IMAGE_SIZE, device=None):
    """
    Runs DINOv2 on a rendered image and returns spatial patch features.

    Args:
        color_img: np.ndarray [H, W, 3] uint8 RGB image.
        variant: DINOv2 model variant name.
        image_size: int. Image is resized to this before feeding DINOv2.
        device: torch device.

    Returns:
        patch_feats: np.ndarray [n_ph, n_pw, D] where:
                     n_ph = n_pw = image_size // PATCH_SIZE (e.g. 37 for 518)
                     D = model feature dim (e.g. 768 for vitb14)
    """
    from PIL import Image as PILImage

    model, device = get_dino_model(variant, device)

    img_pil = PILImage.fromarray(color_img).resize(
        (image_size, image_size), PILImage.BILINEAR)
    x = _dino_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        out = model.forward_features(x)

    # patch tokens: [1, n_patches, D]
    patch_tokens = out['x_norm_patchtokens']
    n = image_size // _PATCH_SIZE   # number of patches per side (e.g. 37)
    D = patch_tokens.shape[-1]

    patch_feats = patch_tokens.squeeze(0).reshape(n, n, D)  # [n, n, D]
    return patch_feats.cpu().numpy()


# -----------------------------------------------------------------------
# Step 3: Project 3D vertices → 2D pixel → patch index
# -----------------------------------------------------------------------
def project_vertices_to_patches(VPos, camera_pose, fov_y,
                                image_size=_IMAGE_SIZE,
                                patch_size=_PATCH_SIZE):
    """
    Projects each 3D vertex onto the rendered image plane and determines
    which DINOv2 patch it falls into.

    Args:
        VPos:        np.ndarray [N, 3] vertex positions.
        camera_pose: np.ndarray [4, 4] camera-to-world transform (from render step).
        fov_y:       float, vertical FOV in radians.
        image_size:  int, rendered image resolution.
        patch_size:  int, DINOv2 patch size in pixels.

    Returns:
        patch_row:  np.ndarray [N] int, patch row for each vertex.
        patch_col:  np.ndarray [N] int, patch col for each vertex.
        visible:    np.ndarray [N] bool, True if vertex projects inside image.
    """
    N = VPos.shape[0]

    # world-to-camera = inverse of camera_pose
    world_to_cam = np.linalg.inv(camera_pose)

    # Transform vertices to camera space
    VPos_h = np.hstack([VPos, np.ones((N, 1))])        # [N, 4]
    V_cam  = (world_to_cam @ VPos_h.T).T[:, :3]        # [N, 3]  (x_c, y_c, z_c)

    # Perspective projection using the same fov_y as pyrender
    f = image_size / (2.0 * np.tan(fov_y / 2.0))
    cx = cy = image_size / 2.0

    in_front = V_cam[:, 2] > 0   # only vertices with positive depth

    px = np.where(in_front, f * V_cam[:, 0] / (V_cam[:, 2] + 1e-8) + cx, -1)
    py = np.where(in_front, f * V_cam[:, 1] / (V_cam[:, 2] + 1e-8) + cy, -1)

    # Note: pyrender uses a right-handed camera (+Y up, -Z forward).
    # Flip Y so that image Y=0 is top.
    py = image_size - py

    # Determine which patch (row, col) each pixel belongs to
    n_patches = image_size // patch_size
    patch_row = np.clip((py // patch_size).astype(int), 0, n_patches - 1)
    patch_col = np.clip((px // patch_size).astype(int), 0, n_patches - 1)

    # Visible: in front of camera AND within image bounds
    visible = (in_front &
               (px >= 0) & (px < image_size) &
               (py >= 0) & (py < image_size))

    return patch_row, patch_col, visible


# -----------------------------------------------------------------------
# Step 4: Assign DINOv2 patch feature to each vertex
# -----------------------------------------------------------------------
def assign_dino_features_to_vertices(patch_feats, patch_row, patch_col,
                                     visible, N):
    """
    Assigns each 3D vertex the DINOv2 feature of its projected patch.
    Invisible vertices (back-face, out-of-frame) get a zero vector.

    Args:
        patch_feats: np.ndarray [n_ph, n_pw, D] DINOv2 patch features.
        patch_row:   np.ndarray [N] int.
        patch_col:   np.ndarray [N] int.
        visible:     np.ndarray [N] bool.
        N:           int, number of vertices.

    Returns:
        vertex_dino: np.ndarray [N, D] per-vertex DINOv2 features.
    """
    D = patch_feats.shape[2]
    vertex_dino = np.zeros((N, D), dtype=np.float32)
    if visible.any():
        vertex_dino[visible] = patch_feats[patch_row[visible],
                                           patch_col[visible]]
    return vertex_dino


# -----------------------------------------------------------------------
# Convenience: all-in-one for a mesh
# -----------------------------------------------------------------------
def get_vertex_dino_features(VPos, ITris, variant='dinov2_vitb14_reg',
                              image_size=_IMAGE_SIZE, device=None,
                              use_normals_as_color=True):
    """
    Full pipeline: render → DINOv2 → project → assign.

    Args:
        VPos:   np.ndarray [N, 3]
        ITris:  np.ndarray [M, 3]
        variant, image_size, device: passed to DINOv2 extractor.
        use_normals_as_color: color mesh by surface normals (recommended).

    Returns:
        vertex_dino: np.ndarray [N, D] per-vertex DINOv2 features (L2 normalized).
        visible:     np.ndarray [N] bool, which vertices were visible to camera.
        color_img:   np.ndarray [H, W, 3] uint8 rendered image (for inspection).
    """
    N = VPos.shape[0]

    # 1. Render
    color_img, depth_img, camera_pose, fov_y = render_mesh_to_image(
        VPos, ITris, image_size=image_size,
        use_normals_as_color=use_normals_as_color)

    # 2. DINOv2 patch features
    patch_feats = extract_dino_patch_features(color_img, variant=variant,
                                              image_size=image_size,
                                              device=device)

    # 3. Project vertices → patches
    patch_row, patch_col, visible = project_vertices_to_patches(
        VPos, camera_pose, fov_y, image_size=image_size,
        patch_size=_PATCH_SIZE)

    # 4. Assign features
    vertex_dino = assign_dino_features_to_vertices(
        patch_feats, patch_row, patch_col, visible, N)

    # L2-normalize per vertex (zero-vec stays zero)
    norms = np.linalg.norm(vertex_dino, axis=1, keepdims=True)
    vertex_dino = vertex_dino / (norms + 1e-8)

    return vertex_dino, visible, color_img
