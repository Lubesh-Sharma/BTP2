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
# Camera Pose Utilities
# -----------------------------------------------------------------------
def compute_orbit_camera_pose(centroid, dist, azimuth_rad, elevation_rad=0.0):
    """
    Computes a 4x4 camera-to-world matrix orbiting around centroid.
    
    In OpenGL / pyrender camera convention:
      - Camera looks along the -Z axis in local camera coordinates.
      - +Y is camera up, +X is camera right.
      - Orbiting around world Y-axis (upright body):
          azimuth = 0: camera at +Z (front view)
          azimuth = pi/2: camera at +X (right side view)
          azimuth = pi: camera at -Z (back view)
          azimuth = 3pi/2: camera at -X (left side view)

    Args:
        centroid: np.ndarray [3] mesh center of mass.
        dist: float, distance from centroid to camera.
        azimuth_rad: float, azimuth angle around world Y-axis in radians.
        elevation_rad: float, elevation angle above horizontal plane in radians.

    Returns:
        pose: np.ndarray [4, 4] camera-to-world transform.
    """
    cos_el = np.cos(elevation_rad)
    sin_el = np.sin(elevation_rad)
    sin_az = np.sin(azimuth_rad)
    cos_az = np.cos(azimuth_rad)

    # Position in world coordinates
    cam_pos = centroid + dist * np.array([
        sin_az * cos_el,
        sin_el,
        cos_az * cos_el
    ], dtype=np.float64)

    # Camera looking towards centroid: forward = -cam_z, so cam_z points away
    cam_z = (cam_pos - centroid)
    cam_z = cam_z / (np.linalg.norm(cam_z) + 1e-8)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_x = np.cross(world_up, cam_z)
    if np.linalg.norm(cam_x) < 1e-6:
        world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        cam_x = np.cross(world_up, cam_z)
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = cam_x
    pose[:3, 1] = cam_y
    pose[:3, 2] = cam_z
    pose[:3, 3] = cam_pos
    return pose


# -----------------------------------------------------------------------
# Step 1: Render mesh to 2D images (multi-view supported)
# -----------------------------------------------------------------------
def render_mesh_multiview(VPos, ITris, n_views=4, image_size=_IMAGE_SIZE,
                          use_normals_as_color=True, elevation_deg=0.0):
    """
    Renders a 3D mesh from multiple viewpoints orbiting around the vertical axis.

    Args:
        VPos:   np.ndarray [N, 3] vertex positions.
        ITris:  np.ndarray [M, 3] triangle face indices.
        n_views: int, number of orbiting views (default: 4: 0°, 90°, 180°, 270°).
        image_size: int, rendered image resolution (square).
        use_normals_as_color: bool. If True, color vertices by surface normals.
        elevation_deg: float, camera elevation angle in degrees.

    Returns:
        views_data: list of dicts with:
            'color_img': np.ndarray [H, W, 3] uint8
            'depth_img': np.ndarray [H, W] float32
            'camera_pose': np.ndarray [4, 4]
            'fov_y': float
            'azimuth_rad': float
    """
    import trimesh
    import pyrender

    mesh_tri = trimesh.Trimesh(vertices=VPos, faces=ITris, process=False)

    if use_normals_as_color:
        mesh_tri.vertex_normals  # triggers normal computation
        normals_rgb = (mesh_tri.vertex_normals * 0.5 + 0.5)
        normals_rgb = np.clip(normals_rgb, 0, 1)
        mesh_tri.visual.vertex_colors = (normals_rgb * 255).astype(np.uint8)

    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri, smooth=False)

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh_pr)

    centroid = VPos.mean(axis=0)
    extent   = np.max(np.linalg.norm(VPos - centroid, axis=1))
    fov_y    = np.pi / 3.0   # 60°
    dist     = extent / np.tan(fov_y / 2) * 1.5  # 1.5x padding

    elevation_rad = np.radians(elevation_deg)
    azimuths = [2.0 * np.pi * i / n_views for i in range(n_views)]

    initial_pose = compute_orbit_camera_pose(centroid, dist, azimuths[0], elevation_rad)
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=1.0)
    cam_node = scene.add(camera, pose=initial_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=initial_pose)

    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    views_data = []
    try:
        for az in azimuths:
            pose = compute_orbit_camera_pose(centroid, dist, az, elevation_rad)
            scene.set_pose(cam_node, pose=pose)
            scene.set_pose(light_node, pose=pose)
            color_img, depth_img = renderer.render(scene)
            views_data.append({
                'color_img': color_img,
                'depth_img': depth_img,
                'camera_pose': pose,
                'fov_y': fov_y,
                'azimuth_rad': az,
            })
    finally:
        renderer.delete()

    return views_data


def render_mesh_to_image(VPos, ITris, image_size=_IMAGE_SIZE,
                         use_normals_as_color=True):
    """
    Renders a 3D mesh to a single 2D RGB image from front (+Z) view.
    Preserved for backward compatibility.
    """
    views = render_mesh_multiview(
        VPos, ITris, n_views=1, image_size=image_size,
        use_normals_as_color=use_normals_as_color, elevation_deg=0.0)
    v0 = views[0]
    return v0['color_img'], v0['depth_img'], v0['camera_pose'], v0['fov_y']


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
# Step 3: Project 3D vertices → 2D pixel → patch index with depth test
# -----------------------------------------------------------------------
def project_vertices_to_patches(VPos, camera_pose, fov_y,
                                image_size=_IMAGE_SIZE,
                                patch_size=_PATCH_SIZE,
                                depth_img=None,
                                depth_tol=0.03):
    """
    Projects each 3D vertex onto the rendered image plane and determines
    which DINOv2 patch it falls into, with optional z-buffer depth occlusion testing.

    OpenGL / pyrender convention:
      - Camera looks in the  -Z  direction in camera space.
      - Vertices IN FRONT of camera have  z_cam < 0.
      - Projection: px = f * x_cam / (-z_cam) + cx
                    py = -f * y_cam / (-z_cam) + cy   (Y flipped for image coords)

    Args:
        VPos:        np.ndarray [N, 3] vertex positions.
        camera_pose: np.ndarray [4, 4] camera-to-world transform (from render step).
        fov_y:       float, vertical FOV in radians.
        image_size:  int, rendered image resolution.
        patch_size:  int, DINOv2 patch size in pixels.
        depth_img:   Optional np.ndarray [H, W] float32 depth map from renderer.
                     If supplied, vertices occluded by foreground geometry are rejected.
        depth_tol:   float, depth tolerance fraction relative to mesh diameter.

    Returns:
        patch_row:  np.ndarray [N] int, patch row for each vertex.
        patch_col:  np.ndarray [N] int, patch col for each vertex.
        visible:    np.ndarray [N] bool, True if vertex is visible to this camera.
    """
    N = VPos.shape[0]

    # world-to-camera = inverse of camera_pose (camera-to-world)
    world_to_cam = np.linalg.inv(camera_pose)

    # Transform vertices to camera space
    VPos_h = np.hstack([VPos, np.ones((N, 1))])        # [N, 4]
    V_cam  = (world_to_cam @ VPos_h.T).T[:, :3]        # [N, 3]

    # In OpenGL camera space, camera looks in -Z direction.
    # Vertices in front of camera have z_cam < 0.
    in_front = V_cam[:, 2] < 0
    depth = -V_cam[:, 2]                               # positive depth value

    # Perspective projection
    f  = image_size / (2.0 * np.tan(fov_y / 2.0))
    cx = cy = image_size / 2.0

    # px: right is +X in camera space → +col in image
    # py: up is +Y in camera space → -row in image (image Y=0 is top)
    px = np.where(in_front,  f * V_cam[:, 0] / (depth + 1e-8) + cx, -1.0)
    py = np.where(in_front, -f * V_cam[:, 1] / (depth + 1e-8) + cy, -1.0)

    # Determine patch (row, col) from pixel (py, px)
    n_patches = image_size // patch_size
    patch_row = np.clip((py // patch_size).astype(int), 0, n_patches - 1)
    patch_col = np.clip((px // patch_size).astype(int), 0, n_patches - 1)

    # Visible in frustum: in front AND within image bounds
    in_bounds = (in_front &
                 (px >= 0) & (px < image_size) &
                 (py >= 0) & (py < image_size))

    if depth_img is not None:
        # Check z-buffer occlusion against the rendered depth buffer
        py_int = np.clip(np.round(py).astype(int), 0, image_size - 1)
        px_int = np.clip(np.round(px).astype(int), 0, image_size - 1)
        buf_depth = depth_img[py_int, px_int]

        centroid = VPos.mean(axis=0)
        extent = np.max(np.linalg.norm(VPos - centroid, axis=1))
        tol = depth_tol * (extent * 2.0) if extent > 0 else 0.05

        non_occluded = np.zeros(N, dtype=bool)
        valid_hits = in_bounds & (buf_depth > 0)
        non_occluded[valid_hits] = depth[valid_hits] <= (buf_depth[valid_hits] + tol)

        # For boundary pixels where buf_depth == 0 due to rasterization discretization, check 3x3 window
        boundary = in_bounds & (buf_depth == 0)
        if boundary.any():
            for idx in np.where(boundary)[0]:
                r0 = max(0, py_int[idx] - 1)
                r1 = min(image_size, py_int[idx] + 2)
                c0 = max(0, px_int[idx] - 1)
                c1 = min(image_size, px_int[idx] + 2)
                window = depth_img[r0:r1, c0:c1]
                pos_vals = window[window > 0]
                if len(pos_vals) > 0 and depth[idx] <= (pos_vals.min() + tol):
                    non_occluded[idx] = True

        visible = in_bounds & non_occluded
    else:
        visible = in_bounds

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
# Convenience: Multi-view pipeline for a mesh
# -----------------------------------------------------------------------
def get_vertex_dino_features(VPos, ITris, variant='dinov2_vitb14_reg',
                              image_size=_IMAGE_SIZE, device=None,
                              use_normals_as_color=True, n_views=4,
                              elevation_deg=0.0):
    """
    Full multi-view pipeline:
      1. Renders mesh from n_views viewpoints orbiting around centroid.
      2. For each view, extracts DINOv2 patch features.
      3. Projects vertices to patches with z-buffer depth occlusion testing.
      4. Accumulates and averages features across all visible viewpoints.
      5. L2-normalizes the final vertex features.

    Args:
        VPos:   np.ndarray [N, 3] vertex positions.
        ITris:  np.ndarray [M, 3] triangle face indices.
        variant, image_size, device: passed to DINOv2 extractor.
        use_normals_as_color: color mesh by surface normals (recommended).
        n_views: int, number of camera views orbiting horizontally (default: 4: 0°, 90°, 180°, 270°).
                 Set n_views=1 for single front-view.
        elevation_deg: float, camera elevation angle in degrees (default: 0.0).

    Returns:
        vertex_dino: np.ndarray [N, D] per-vertex DINOv2 features (L2 normalized).
        visible:     np.ndarray [N] bool, which vertices were visible in at least one view.
        color_img:   np.ndarray [H, W, 3] uint8 front view rendered image (for inspection/saving).
    """
    N = VPos.shape[0]

    # 1. Render multi-view images and depth maps
    views_data = render_mesh_multiview(
        VPos, ITris, n_views=n_views, image_size=image_size,
        use_normals_as_color=use_normals_as_color, elevation_deg=elevation_deg)

    # 2. Extract DINOv2 patch features and accumulate across views
    accumulated_features = None
    view_counts = np.zeros(N, dtype=np.int32)

    for view in views_data:
        color_img   = view['color_img']
        depth_img   = view['depth_img']
        camera_pose = view['camera_pose']
        fov_y       = view['fov_y']

        patch_feats = extract_dino_patch_features(
            color_img, variant=variant, image_size=image_size, device=device)

        if accumulated_features is None:
            D = patch_feats.shape[-1]
            accumulated_features = np.zeros((N, D), dtype=np.float32)

        # Depth-aware projection: vertices occluded in this camera are marked invisible
        patch_row, patch_col, visible = project_vertices_to_patches(
            VPos, camera_pose, fov_y, image_size=image_size,
            patch_size=_PATCH_SIZE, depth_img=depth_img)

        v_feats = assign_dino_features_to_vertices(
            patch_feats, patch_row, patch_col, visible, N)

        if visible.any():
            accumulated_features[visible] += v_feats[visible]
            view_counts[visible] += 1

    # 3. Average features over visible views
    has_views = view_counts > 0
    if has_views.any():
        accumulated_features[has_views] /= view_counts[has_views, None]

    # 4. Impute any remaining occluded vertices via nearest visible neighbor
    if not np.all(has_views) and np.any(has_views):
        from scipy.spatial import cKDTree
        vis_indices = np.where(has_views)[0]
        tree = cKDTree(VPos[vis_indices])
        unvis_indices = np.where(~has_views)[0]
        _, nearest_idx = tree.query(VPos[unvis_indices], k=1)
        accumulated_features[unvis_indices] = accumulated_features[vis_indices[nearest_idx]]

    # 5. L2-normalize per vertex
    norms = np.linalg.norm(accumulated_features, axis=1, keepdims=True)
    vertex_dino = accumulated_features / (norms + 1e-8)

    front_img = views_data[0]['color_img']
    total_visible = has_views

    return vertex_dino, total_visible, front_img
