"""DINOv2 features for raw point clouds - no mesh, no renderer.

Follows DV-Matcher (Uni3FC_DINO_proj, models/model.py; rqhuang88/DV-Matcher):
  1. Orthographic projection of the points onto an image grid; every point is
     splatted as a (2s+1) x (2s+1) pixel patch (s=2 -> 5x5, as in DV-Matcher).
  2. The depth map is coloured with a matplotlib colormap (DV-Matcher: 'PiYG')
     to obtain an RGB image for the 2D foundation model.
  3. DINOv2 features per view: 'bilinear' (default, no install) samples DINOv2 patch
     tokens bilinearly at each point's pixel - use img_size 448 (32x32 patches) for
     detail; 'featup' uses FeatUp's learned upsampler like DV-Matcher (requires
     pip install of FeatUp).
  4. Image-to-point back-projection: every point reads the feature at its own pixel.

Options beyond DV-Matcher (all configurable):
  views: 'dvmatcher' - the 3 axis-aligned views of DV-Matcher, features concatenated.
                       Orientation dependent: only valid when shapes share an orientation.
         'orbit'     - n_azimuth x elevations views orbiting the vertical (y) axis,
                       features averaged -> approximately yaw invariant, cacheable per shape.
  projection: 'xray'    - DV-Matcher: depth values summed per pixel, every point takes
                          the feature of its pixel (hidden points get the front surface).
              'zbuffer' - nearest point per pixel; a point only reads features in views
                          where it is visible, and gets a visibility confidence.
"""
import os
import math

import numpy as np
import torch
import torch.nn.functional as F

_MODELS = {}


def hub_load(repo, entry, **kwargs):
    """torch.hub.load without the GitHub API call that fails with 'HTTP 403: rate limit exceeded'.

    1. If the repo is already in the hub cache (~/.cache/torch/hub/<owner>_<name>_main),
       load it locally (no network at all).
    2. Otherwise download it with skip_validation=True (no API request; only the
       code archive is fetched). Exporting GITHUB_TOKEN also lifts the API limit.
    """
    owner, name = repo.split('/')
    local = os.path.join(torch.hub.get_dir(), f"{owner}_{name}_main")
    if os.path.isfile(os.path.join(local, 'hubconf.py')):
        return torch.hub.load(local, entry, source='local', **kwargs)
    return torch.hub.load(repo, entry, trust_repo=True, skip_validation=True, **kwargs)
_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# 2D backbones
# ---------------------------------------------------------------------------
class Backbone2D:
    """`__call__(images [B,3,H,W] ImageNet-normalised) -> features [B, C, H', W']`."""

    def __init__(self, upsampler='bilinear', variant='dinov2_vits14', device='cuda'):
        self.kind = upsampler
        self.device = device
        key = (upsampler, variant, str(device))
        if key not in _MODELS:
            if upsampler == 'featup':
                # FeatUp (as in DV-Matcher) is a pip package with a compiled CUDA op;
                # torch.hub alone cannot run it.
                try:
                    import featup  # noqa: F401
                except ImportError as e:
                    raise RuntimeError(
                        "dino.upsampler: featup needs the FeatUp package "
                        "(pip install git+https://github.com/mhamilton723/FeatUp). "
                        "Without installing anything use dino.upsampler: bilinear.") from e
                print("[2D] Loading FeatUp DINOv2 upsampler...")
                model = hub_load("mhamilton723/FeatUp", 'dinov2', use_norm=True)
            elif upsampler == 'bilinear':
                print(f"[2D] Loading {variant} (facebookresearch/dinov2)...")
                model = hub_load('facebookresearch/dinov2', variant)
            else:
                raise ValueError("dino.upsampler must be 'featup' or 'bilinear'")
            _MODELS[key] = model.to(device).eval()
        self.model = _MODELS[key]

    @torch.no_grad()
    def __call__(self, images):
        if self.kind == 'featup':
            return self.model(images)                                  # [B, 384, H, W]
        B, _, H, W = images.shape
        out = self.model.forward_features(images)['x_norm_patchtokens']  # [B, h*w, C]
        h, w = H // 14, W // 14
        return out.transpose(1, 2).reshape(B, -1, h, w)               # sampled bilinearly later


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def view_matrices(views='orbit', n_azimuth=8, elevations_deg=(-30.0, 0.0, 30.0)):
    """List of 3x3 matrices with rows (image u axis, image v axis, depth axis)."""
    if views == 'dvmatcher':
        # DV-Matcher: rotate -90 deg about z, then project on (x,y), (z,x), (y,z).
        Rz = _rot_z(-math.pi / 2)
        e = np.eye(3)
        perms = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
        return [np.stack([e[p[0]], e[p[1]], e[p[2]]]) @ Rz for p in perms]
    mats = []
    for r, el in enumerate(elevations_deg):
        el = math.radians(el)
        for k in range(n_azimuth):
            az = 2 * math.pi * k / n_azimuth + (r % 2) * math.pi / n_azimuth
            d = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])  # to camera
            right = np.cross([0.0, 1.0, 0.0], d)
            right /= np.linalg.norm(right) + 1e-12
            up = np.cross(d, right)
            mats.append(np.stack([right, -up, -d]))   # u right, v down (image rows), depth away from camera
    return mats


# ---------------------------------------------------------------------------
# projection (point splatting)
# ---------------------------------------------------------------------------
def _colormap(values01, name):
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(name) if hasattr(matplotlib, 'colormaps') else \
        __import__('matplotlib.cm', fromlist=['cm']).get_cmap(name)
    return torch.as_tensor(cmap(values01.cpu().numpy())[..., :3], dtype=torch.float32)


def project_points(points, M, img_size=224, splat=2, projection='zbuffer', colormap='PiYG',
                   depth_tol=0.02):
    """Splat one orthographic view.

    points: [N,3] torch (centred, unit radius). M: [3,3] view matrix.
    Returns image [3,H,W] (ImageNet-normalised), pixel (row, col) [N,2] long, visible [N] bool.
    """
    dev = points.device
    S = img_size
    q = points @ torch.as_tensor(M, dtype=points.dtype, device=dev).T     # [N, 3] (u, v, depth)
    pad = splat + 2
    scale = (S - 2 * pad) / 2.0                                             # unit sphere fills the image
    row = torch.clamp(((q[:, 1] + 1) * scale + pad).long(), 0, S - 1)
    col = torch.clamp(((q[:, 0] + 1) * scale + pad).long(), 0, S - 1)
    depth = q[:, 2]

    off = torch.arange(-splat, splat + 1, device=dev)
    dr, dc = torch.meshgrid(off, off, indexing='ij')
    rr = (row[:, None] + dr.reshape(1, -1)).clamp(0, S - 1)
    cc = (col[:, None] + dc.reshape(1, -1)).clamp(0, S - 1)
    pix = (rr * S + cc).reshape(-1)                                         # [N * (2s+1)^2]
    dep = depth[:, None].expand_as(rr).reshape(-1)

    fg = torch.zeros(S * S, dtype=torch.bool, device=dev)
    fg[pix] = True
    if projection == 'xray':
        # DV-Matcher: sum of depth values falling into each pixel, squashed by a sigmoid.
        acc = torch.zeros(S * S, device=dev).index_add_(0, pix, dep)
        img = torch.sigmoid(acc)
        visible = torch.ones(len(points), dtype=torch.bool, device=dev)
    elif projection == 'zbuffer':
        zbuf = torch.full((S * S,), float('inf'), device=dev).scatter_reduce(0, pix, dep, reduce='amin')
        img = torch.where(fg, -zbuf, torch.zeros_like(zbuf))                 # nearer -> larger
        visible = depth <= zbuf[row * S + col] + depth_tol * 2.0             # unit sphere: diameter 2
    else:
        raise ValueError("projection must be 'xray' or 'zbuffer'")

    v = img[fg]
    img01 = torch.zeros(S * S, device=dev)
    img01[fg] = (v - v.min()) / (v.max() - v.min()).clamp_min(1e-8)
    rgb = _colormap(img01.reshape(S, S), colormap).to(dev).permute(2, 0, 1)   # [3, S, S] in [0, 1]
    rgb[:, ~fg.reshape(S, S)] = 1.0                                           # white background
    rgb = (rgb.unsqueeze(0) - _IMNET_MEAN.to(dev)) / _IMNET_STD.to(dev)
    return rgb[0], torch.stack([row, col], 1), visible


def sample_features(feat, pix_rc, img_size):
    """Bilinear read of feature map [C,h,w] at pixel centres (row, col) -> [N, C]."""
    y = (pix_rc[:, 0].float() + 0.5) / img_size * 2 - 1
    x = (pix_rc[:, 1].float() + 0.5) / img_size * 2 - 1
    grid = torch.stack([x, y], -1).view(1, -1, 1, 2)
    out = F.grid_sample(feat.unsqueeze(0), grid, mode='bilinear', align_corners=False)
    return out[0, :, :, 0].T


# ---------------------------------------------------------------------------
# full lifting
# ---------------------------------------------------------------------------
@torch.no_grad()
def point_cloud_dino_features(points_np, cfg, device=None):
    """points [N,3] -> (features [N, D] unit norm, confidence [N] in [0,1])."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    views = cfg.get('views', 'orbit')
    projection = cfg.get('projection', 'zbuffer')
    aggregation = cfg.get('aggregation', 'concat' if views == 'dvmatcher' else 'mean')
    S = cfg.get('img_size', 224)
    backbone = Backbone2D(cfg.get('upsampler', 'bilinear'), cfg.get('variant', 'dinov2_vits14'), device)

    p = torch.as_tensor(np.asarray(points_np), dtype=torch.float32, device=device)
    p = p - p.mean(0, keepdim=True)
    p = p / p.norm(dim=-1).max().clamp_min(1e-8)
    mats = view_matrices(views, cfg.get('n_azimuth', 8), cfg.get('elevations_deg', [-30, 0, 30]))

    projected = [project_points(p, M, S, cfg.get('splat', 2), projection, cfg.get('colormap', 'PiYG'),
                                cfg.get('depth_tol', 0.02)) for M in mats]
    per_view, vis = [], []
    bs = cfg.get('batch_views', 4)
    for s in range(0, len(projected), bs):
        chunk = projected[s:s + bs]
        fmap = backbone(torch.stack([c[0] for c in chunk]))                  # [b, C, h, w]
        for k, (_, rc, visible) in enumerate(chunk):
            per_view.append(F.normalize(sample_features(fmap[k], rc, S), dim=-1))
            vis.append(visible)
    vis = torch.stack(vis).float()                                           # [V, N]

    if aggregation == 'concat':
        feat = torch.cat(per_view, dim=-1)
        conf = vis.mean(0)
    else:
        stack = torch.stack(per_view)                                        # [V, N, C]
        w = vis.unsqueeze(-1)
        feat = (stack * w).sum(0) / w.sum(0).clamp_min(1e-6)
        count = vis.sum(0)
        unseen = count == 0
        if unseen.any() and (~unseen).any():                                 # impute from nearest seen point
            d = torch.cdist(p[unseen], p[~unseen])
            feat[unseen] = feat[~unseen][d.argmin(1)]
        ref = torch.quantile(count[~unseen], 0.9) if (~unseen).any() else torch.tensor(1.0)
        conf = (count / ref.clamp_min(1)).clamp(0, 1)
    feat = F.normalize(feat, dim=-1)

    k = cfg.get('smooth_k', 8)
    if k and k > 1:                                                          # kNN smoothing (Diff3F ball query)
        from scipy.spatial import cKDTree
        _, nn = cKDTree(p.cpu().numpy()).query(p.cpu().numpy(), k=k)
        nn = torch.as_tensor(nn, device=device)
        w = (conf[nn] + 1e-3).unsqueeze(-1)
        feat = F.normalize((feat[nn] * w).sum(1) / w.sum(1), dim=-1)
    return feat.cpu().numpy().astype(np.float32), conf.cpu().numpy().astype(np.float32)


def point_dino_for_shape(name, points, cfg, cache_dir):
    """Cached wrapper. Cache key includes every setting that changes the features."""
    os.makedirs(cache_dir, exist_ok=True)
    keys = ['upsampler', 'variant', 'views', 'projection', 'n_azimuth', 'elevations_deg', 'img_size',
            'splat', 'colormap', 'smooth_k', 'aggregation']
    tag = '_'.join(str(cfg.get(k, '')).replace(' ', '').replace('[', '').replace(']', '').replace(',', '-')
                   for k in keys)
    import hashlib
    path = os.path.join(cache_dir, f"{name}__{hashlib.md5(tag.encode()).hexdigest()[:10]}.npz")
    if os.path.exists(path):
        c = np.load(path)
        return c['feat'].astype(np.float32), c['conf'].astype(np.float32)
    feat, conf = point_cloud_dino_features(points, cfg)
    np.savez_compressed(path, feat=feat.astype(np.float16), conf=conf, tag=tag)
    return feat, conf
