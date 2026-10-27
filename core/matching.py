"""Heat-Anchored Deformation Matching (HADM).

Point-to-point correspondence from encoder + DINO features, without
functional maps and without Sinkhorn. "Features propose, geometry decides":

 1. Feature fusion   encoder embedding (pose-aware, OEM-aligned frame) +
                     DINOv2 (semantic, alignment-free), DINO gated by visibility
                     confidence; unseen vertices use the encoder's predicted DINO.
 2. Anchors          mutual nearest neighbours, spread over the surface, pruned by
                     spectral matching on biharmonic (intrinsic) distance
                     consistency: anchors that disagree with the majority (wrong
                     limb, partial flips) are rejected.
 3. Heat anchors     every vertex gets a descriptor of its intrinsic proximity
                     to the anchors; matching anchors -> matching descriptors.
                     Re-select anchors with these descriptors (coarse -> fine).
 4. Deformation      embedded deformation graph (geodesic skinning, ARAP
                     regularisation) bends the source onto the target, driven by
                     anchors and a bidirectional feature-gated nearest-point term.
 5. Dense map        each source vertex -> best target vertex in joint
                     (deformed position + feature) space.

All operations are chunked so no N1 x N2 matrix is ever materialised.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


DEFAULTS = {
    'w_encoder': 1.0,          # weight of encoder cosine similarity
    'w_dino': 1.0,             # weight of DINO cosine similarity
    'dino_pca_dim': 64,        # joint PCA of the pair's DINO features (0 = off)
    'use_predicted_dino': True,  # fill low-confidence vertices with encoder-predicted DINO
    'w_heat': 1.5,             # weight of heat-anchor descriptors
    'heat_sigmas': [0.15, 0.35],  # in units of the mean biharmonic distance
    'anchor_rounds': [48, 96, 160],  # anchors per round (round 0: features only)
    'anchor_pool': 2000,       # best-margin mutual NN kept before spreading
    'anchor_candidates_factor': 3,  # candidates = factor x anchors (FPS in psi space)
    'consistency_sigma': 0.12,  # tolerance on biharmonic distance disagreement
    'accept_threshold': 0.5,   # min mean consistency with already accepted anchors
    'rigid_init': True,        # Kabsch (+ scale) from anchors before deforming
    'deform': {
        'enabled': True,
        'stages': [
            {'nodes': 64, 'arap': 20.0, 'w_pos': 2.0, 'iters': 120},
            {'nodes': 160, 'arap': 5.0, 'w_pos': 6.0, 'iters': 120},
            {'nodes': 320, 'arap': 1.5, 'w_pos': 15.0, 'iters': 120},
        ],
        'skin_k': 4,
        'node_neighbors': 6,
        'lr': 0.01,
        'reassign_every': 10,
        'data_samples': 4000,
        'w_anchor': 1.0,
        'w_data': 1.0,
        'robust_sigma': 0.08,
    },
    'final_w_pos': None,       # None = last stage w_pos
    'chunk': 4096,
}


def merge_cfg(user):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def normalize_points(p):
    """Centre and scale to unit radius. Returns points, centre, scale."""
    c = p.mean(0, keepdim=True)
    s = (p - c).norm(dim=-1).max().clamp_min(1e-8)
    return (p - c) / s, c, s


def fps_indices(x, m, start=0):
    """Farthest point sampling on rows of x (any dimension)."""
    n = x.shape[0]
    m = min(m, n)
    idx = torch.empty(m, dtype=torch.long, device=x.device)
    dist = torch.full((n,), float('inf'), device=x.device)
    cur = start
    for i in range(m):
        idx[i] = cur
        dist = torch.minimum(dist, (x - x[cur]).pow(2).sum(-1))
        cur = int(dist.argmax())
    return idx


def topk_joint(A, B, k=1, pa=None, pb=None, w_pos=0.0, chunk=4096):
    """Top-k of  A_i . B_j - w_pos ||pa_i - pb_j||^2  over j, chunked over i."""
    vals, idxs = [], []
    for s in range(0, A.shape[0], chunk):
        S = A[s:s + chunk] @ B.T
        if w_pos > 0:
            S = S - w_pos * torch.cdist(pa[s:s + chunk], pb).pow(2)
        v, i = S.topk(k, dim=1)
        vals.append(v)
        idxs.append(i)
    return torch.cat(vals), torch.cat(idxs)


def rodrigues(rvec):
    theta = rvec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = rvec / theta
    K = torch.zeros(rvec.shape[0], 3, 3, device=rvec.device, dtype=rvec.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = theta.sin().unsqueeze(-1), theta.cos().unsqueeze(-1)
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def weighted_similarity_fit(x, y, w=None, allow_scale=True, scale_range=(0.8, 1.25)):
    """Proper rotation R, scale s, translation t minimising sum w ||s R x + t - y||^2."""
    if w is None:
        w = torch.ones(x.shape[0], device=x.device)
    w = w / w.sum()
    mx, my = (w[:, None] * x).sum(0), (w[:, None] * y).sum(0)
    xc, yc = x - mx, y - my
    H = (w[:, None] * xc).T @ yc
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d)], device=x.device))
    R = Vt.T @ D @ U.T
    s = torch.tensor(1.0, device=x.device)
    if allow_scale:
        var = (w * xc.pow(2).sum(-1)).sum().clamp_min(1e-12)
        s = ((S * torch.diag(D)).sum() / var).clamp(*scale_range)
    t = my - s * (R @ mx)
    return R, s, t


def graph_from_edges(pos_np, edges):
    """Symmetric sparse graph with Euclidean edge lengths (kNN graph or mesh edges)."""
    edges = np.asarray(edges)
    if edges.ndim == 2 and edges.shape[1] == 3:
        edges = np.concatenate([edges[:, [0, 1]], edges[:, [1, 2]], edges[:, [2, 0]]])
    if edges.size == 0:
        from scipy.spatial import cKDTree
        _, nn = cKDTree(pos_np).query(pos_np, k=9)
        edges = np.stack([np.repeat(np.arange(len(pos_np)), 8), nn[:, 1:].reshape(-1)], 1)
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    w = np.linalg.norm(pos_np[i] - pos_np[j], axis=1) + 1e-9
    n = len(pos_np)
    G = sparse.coo_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                          shape=(n, n)).tocsr()
    G.sum_duplicates()
    return G


# ---------------------------------------------------------------------------
# matcher
# ---------------------------------------------------------------------------
class HeatAnchoredMatcher:
    """See module docstring. `cfg` overrides DEFAULTS (nested dicts merge)."""

    def __init__(self, cfg=None, device='cpu', verbose=False):
        self.cfg = merge_cfg(cfg)
        self.device = device
        self.verbose = verbose

    def log(self, *a):
        if self.verbose:
            print('   [match]', *a)

    # ---------------- 1. features ----------------
    def fuse_features(self, src, tgt):
        """Concatenate weighted unit blocks so A_i . B_j = sum_w w * cos(.)."""
        c = self.cfg
        blocks1, blocks2 = [], []
        if c['w_encoder'] > 0:
            blocks1.append(math.sqrt(c['w_encoder']) * F.normalize(src['z'], dim=-1))
            blocks2.append(math.sqrt(c['w_encoder']) * F.normalize(tgt['z'], dim=-1))
        if c['w_dino'] > 0 and src.get('dino') is not None and tgt.get('dino') is not None:
            d1, d2 = self._dino_block(src), self._dino_block(tgt)
            if c['dino_pca_dim'] and c['dino_pca_dim'] < d1.shape[1]:
                X = torch.cat([d1, d2])
                mu = X.mean(0, keepdim=True)
                _, _, V = torch.pca_lowrank(X - mu, q=c['dino_pca_dim'], center=False)
                d1, d2 = (d1 - mu) @ V, (d2 - mu) @ V
            # sqrt(conf) on both sides -> pair weight sqrt(c1 c2) on the DINO cosine
            g1 = src['dino_conf'].clamp(0, 1).sqrt()[:, None]
            g2 = tgt['dino_conf'].clamp(0, 1).sqrt()[:, None]
            if c['use_predicted_dino'] and src.get('dino_pred') is not None:
                g1 = torch.ones_like(g1)   # gaps already filled by the encoder prediction
                g2 = torch.ones_like(g2)
            blocks1.append(math.sqrt(c['w_dino']) * g1 * F.normalize(d1, dim=-1))
            blocks2.append(math.sqrt(c['w_dino']) * g2 * F.normalize(d2, dim=-1))
        if not blocks1:
            raise ValueError("No feature block enabled (w_encoder / w_dino).")
        return torch.cat(blocks1, 1), torch.cat(blocks2, 1)

    def _dino_block(self, shape):
        d = F.normalize(shape['dino'], dim=-1)
        if self.cfg['use_predicted_dino'] and shape.get('dino_pred') is not None:
            conf = shape['dino_conf'].clamp(0, 1)[:, None]
            d = F.normalize(conf * d + (1 - conf) * F.normalize(shape['dino_pred'], dim=-1), dim=-1)
        return d

    # ---------------- 2. anchors ----------------
    def select_anchors(self, A, B, psi1, psi2, n_anchors, pa=None, pb=None, w_pos=0.0):
        c = self.cfg
        chunk = c['chunk']
        v12, i12 = topk_joint(A, B, 2, pa, pb, w_pos, chunk)
        _, i21 = topk_joint(B, A, 1, pb, pa, w_pos, chunk)
        n1 = A.shape[0]
        ar = torch.arange(n1, device=A.device)
        margin = v12[:, 0] - v12[:, 1]
        mutual = i21[i12[:, 0], 0] == ar
        cand = torch.nonzero(mutual).squeeze(-1)
        min_c = max(4 * n_anchors, 64)
        if cand.numel() < min_c:
            cand = torch.argsort(-margin)[:min_c]
        cand = cand[torch.argsort(-margin[cand])][:c['anchor_pool']]
        n_cand = min(cand.numel(), c['anchor_candidates_factor'] * n_anchors)
        cand = cand[fps_indices(psi1[cand], n_cand)]
        a, b = cand, i12[cand, 0]

        D1 = torch.cdist(psi1[a], psi1[a])
        D2 = torch.cdist(psi2[b], psi2[b])
        W = torch.exp(-(D1 - D2).pow(2) / (2 * c['consistency_sigma'] ** 2))
        W.fill_diagonal_(0)
        x = torch.ones(len(a), device=A.device) / math.sqrt(len(a))
        for _ in range(60):
            x = W @ x
            x = x / x.norm().clamp_min(1e-12)
        order = torch.argsort(-x).tolist()
        accepted = []
        for idx in order:
            if accepted and W[idx, accepted].mean() < c['accept_threshold']:
                continue
            accepted.append(idx)
            if len(accepted) >= n_anchors:
                break
        acc = torch.tensor(accepted, device=A.device, dtype=torch.long)
        Wa = W[acc][:, acc]
        consistency = Wa.sum() / max(1, len(a) * (len(a) - 1))
        info = {'n_candidates': int(len(a)), 'n_anchors': int(len(acc)),
                'consistency': float(consistency),
                'mean_similarity': float(v12[:, 0].mean())}
        return a[acc], b[acc], info

    # ---------------- 3. heat-anchor descriptors ----------------
    def heat_descriptors(self, psi, anchors):
        blocks = []
        for s in self.cfg['heat_sigmas']:
            d2 = torch.cdist(psi, psi[anchors]).pow(2)
            h = torch.exp(-d2 / (2 * s * s))
            blocks.append(F.normalize(h, dim=-1))
        return torch.cat(blocks, 1) / math.sqrt(len(blocks))

    def anchor_propagation(self, F1, F2, psi1, psi2):
        rounds = self.cfg['anchor_rounds']
        a, b, info = self.select_anchors(F1, F2, psi1, psi2, rounds[0])
        self.log(f"round 0: {info}")
        G1, G2 = F1, F2
        for r, n in enumerate(rounds[1:], 1):
            wh = math.sqrt(self.cfg['w_heat'])
            G1 = torch.cat([F1, wh * self.heat_descriptors(psi1, a)], 1)
            G2 = torch.cat([F2, wh * self.heat_descriptors(psi2, b)], 1)
            a, b, info = self.select_anchors(G1, G2, psi1, psi2, n)
            self.log(f"round {r}: {info}")
        wh = math.sqrt(self.cfg['w_heat'])
        G1 = torch.cat([F1, wh * self.heat_descriptors(psi1, a)], 1)
        G2 = torch.cat([F2, wh * self.heat_descriptors(psi2, b)], 1)
        return G1, G2, a, b, info

    # ---------------- 4. deformation ----------------
    def _skinning(self, graph, n_nodes, rest_np, k_skin, k_nodes):
        nodes = fps_indices(torch.as_tensor(rest_np), n_nodes).numpy()
        D = dijkstra(graph, directed=False, indices=nodes)            # [n_nodes, N]
        if not np.isfinite(D).all():                                  # disconnected parts
            eu = np.linalg.norm(rest_np[nodes][:, None] - rest_np[None], axis=-1)
            D = np.where(np.isfinite(D), D, eu * 10)
        order = np.argsort(D, axis=0)[:k_skin + 1]                    # [k+1, N]
        dk = np.take_along_axis(D, order, 0)
        w = np.clip(1 - dk[:k_skin] / (dk[k_skin:k_skin + 1] + 1e-9), 0, None) ** 2
        w = w / np.clip(w.sum(0, keepdims=True), 1e-9, None)
        Dn = D[:, nodes]
        nn = np.argsort(Dn, axis=1)[:, 1:k_nodes + 1]
        return nodes, order[:k_skin].T, w.T, nn

    def deform(self, src_pos, tgt_pos, G1, G2, a, b, graph, edges_cache=None):
        dc = self.cfg['deform']
        dev = src_pos.device
        cur = src_pos.clone()
        last_wpos = 0.0
        for st, stage in enumerate(dc['stages']):
            rest = cur.detach()
            nodes, skin_idx, skin_w, node_nn = self._skinning(
                graph, stage['nodes'], rest.cpu().numpy(), dc['skin_k'], dc['node_neighbors'])
            nodes = torch.as_tensor(nodes, device=dev)
            skin_idx = torch.as_tensor(skin_idx, device=dev)
            skin_w = torch.as_tensor(skin_w, device=dev, dtype=torch.float32)
            node_nn = torch.as_tensor(node_nn, device=dev)
            g = rest[nodes]
            # tiny non-zero init: the axis-angle norm has no gradient exactly at 0
            rvec = (1e-5 * torch.randn(len(nodes), 3, device=dev)).requires_grad_(True)
            tr = torch.zeros(len(nodes), 3, device=dev, requires_grad=True)
            opt = torch.optim.Adam([rvec, tr], lr=dc['lr'])
            diff = rest[:, None, :] - g[skin_idx]                       # [N, k, 3]
            ek = torch.arange(len(nodes), device=dev)[:, None].expand_as(node_nn).reshape(-1)
            el = node_nn.reshape(-1)
            n1 = rest.shape[0]
            S = torch.randperm(n1, device=dev)[:dc['data_samples']]
            T = torch.randperm(tgt_pos.shape[0], device=dev)[:dc['data_samples']]
            w_pos = stage['w_pos']

            def apply():
                R = rodrigues(rvec)
                Rk = R[skin_idx]                                        # [N, k, 3, 3]
                moved = (Rk @ diff.unsqueeze(-1)).squeeze(-1) + g[skin_idx] + tr[skin_idx]
                return (skin_w.unsqueeze(-1) * moved).sum(1), R

            for it in range(stage['iters']):
                if it % dc['reassign_every'] == 0:
                    with torch.no_grad():
                        v_now, _ = apply()
                        _, js = topk_joint(G1[S], G2, 1, v_now[S], tgt_pos, w_pos, self.cfg['chunk'])
                        js = js[:, 0]
                        _, ist = topk_joint(G2[T], G1, 1, tgt_pos[T], v_now, w_pos, self.cfg['chunk'])
                        ist = ist[:, 0]
                v, R = apply()
                rs = (v[S] - tgt_pos[js]).pow(2).sum(-1)
                rt = (v[ist] - tgt_pos[T]).pow(2).sum(-1)
                sig2 = dc['robust_sigma'] ** 2
                e_data = (rs * (sig2 / (sig2 + rs.detach()))).mean() + (rt * (sig2 / (sig2 + rt.detach()))).mean()
                e_anchor = (v[a] - tgt_pos[b]).pow(2).sum(-1).mean()
                pred = (R[ek] @ (g[el] - g[ek]).unsqueeze(-1)).squeeze(-1) + g[ek] + tr[ek]
                e_arap = (pred - (g[el] + tr[el])).pow(2).sum(-1).mean()
                loss = dc['w_data'] * e_data + dc['w_anchor'] * e_anchor + stage['arap'] * e_arap
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                cur, _ = apply()
            last_wpos = w_pos
            self.log(f"deform stage {st}: nodes={len(nodes)} data={e_data.item():.5f} "
                     f"anchor={e_anchor.item():.5f} arap={e_arap.item():.5f}")
        return cur.detach(), last_wpos

    # ---------------- full pipeline ----------------
    @torch.no_grad()
    def score_alignment(self, src, tgt):
        """Cheap score of a target alignment hypothesis (higher = better).

        Fraction of intrinsically consistent anchor mass + mean best fused similarity.
        Used to pick among OEM rotation hypotheses.
        """
        F1, F2 = self.fuse_features(src, tgt)
        _, _, info = self.select_anchors(F1, F2, src['psi'], tgt['psi'], self.cfg['anchor_rounds'][0])
        total_w = self.cfg['w_encoder'] + (self.cfg['w_dino'] if src.get('dino') is not None else 0)
        return info['consistency'] + info['mean_similarity'] / max(total_w, 1e-6), info

    def match(self, src, tgt):
        """src/tgt dicts (torch, same device):
            pos [N,3] (target already OEM-aligned), z [N,C], psi [N,K], edges (np) ,
            optional dino [N,D], dino_conf [N], dino_pred [N,D].
        Returns p2p (LongTensor [N1], source vertex -> target vertex) and info."""
        c = self.cfg
        p1, _, _ = normalize_points(src['pos'])
        p2, _, _ = normalize_points(tgt['pos'])
        with torch.no_grad():
            F1, F2 = self.fuse_features(src, tgt)
            G1, G2, a, b, info = self.anchor_propagation(F1, F2, src['psi'], tgt['psi'])
            if c['rigid_init'] and len(a) >= 4:
                R, s, t = weighted_similarity_fit(p1[a], p2[b])
                p1 = (s * (R @ p1.T)).T + t
        w_pos = 0.0
        if c['deform']['enabled']:
            graph = graph_from_edges(src['pos'].cpu().numpy(), src.get('edges', np.zeros((0, 2))))
            with torch.enable_grad():
                p1, w_pos = self.deform(p1, p2, G1, G2, a, b, graph)
        if c['final_w_pos'] is not None:
            w_pos = c['final_w_pos']
        with torch.no_grad():
            _, idx = topk_joint(G1, G2, 1, p1, p2, w_pos, c['chunk'])
        info.update({'final_w_pos': w_pos, 'anchors': int(len(a))})
        return idx[:, 0], info

    @torch.no_grad()
    def match_greedy(self, src, tgt):
        """Baseline: argmax of fused feature similarity (previous pipeline)."""
        F1, F2 = self.fuse_features(src, tgt)
        _, idx = topk_joint(F1, F2, 1, chunk=self.cfg['chunk'])
        return idx[:, 0], {}
