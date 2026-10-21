"""MV-RoMa align_corners=False coordinates to COLMAP pixel-center coordinates."""
import numpy as np


def normalized_to_colmap(xy, size):
    """COLMAP places the top-left pixel center at (0.5, 0.5). size is (W,H)."""
    return (np.asarray(xy) + 1.) * np.asarray(size) / 2.


def sample_pair(flow, certainty, reverse_flow, reverse_certainty, source_size,
                target_size, stride=4, confidence=.8, cycle_px=2.):
    """Sample a uniform source grid; filter confidence, bounds, and round-trip error.

    Cycle threshold is measured in the forward model's output pixels, independent
    of source file resolution. Reverse fields are bilinearly sampled; no snapping
    is applied to the returned observations.
    """
    if stride < 1 or not 0 <= confidence <= 1 or cycle_px <= 0:
        raise ValueError('Invalid sampling parameters')
    _, h, w = flow.shape
    _, rh, rw = reverse_flow.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    src = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    dst = flow[:, yy, xx].reshape(2, -1).T.astype(float)
    scores = certainty[yy, xx].ravel()
    valid = np.isfinite(dst).all(1) & np.isfinite(scores) & (scores >= confidence)
    valid &= (dst >= -1).all(1) & (dst <= 1).all(1)
    src, dst, scores = src[valid], dst[valid], scores[valid]
    # PyTorch align_corners=False grid to zero-based array coordinates.
    rxy = normalized_to_colmap(dst, (rw, rh)) - .5
    interior = (rxy >= 0).all(1) & (rxy <= [rw-1, rh-1]).all(1)
    src, dst, scores, rxy = src[interior], dst[interior], scores[interior], rxy[interior]
    lo = np.floor(rxy).astype(int)
    hi = np.minimum(lo+1, [rw-1, rh-1])
    delta = rxy-lo
    back = np.zeros_like(dst)
    back_score = np.zeros(len(dst))
    for dx, dy in [(0,0),(1,0),(0,1),(1,1)]:
        x = hi[:,0] if dx else lo[:,0]
        y = hi[:,1] if dy else lo[:,1]
        weight = (delta[:,0] if dx else 1-delta[:,0]) * (delta[:,1] if dy else 1-delta[:,1])
        back += reverse_flow[:, y, x].T * weight[:,None]
        back_score += reverse_certainty[y, x] * weight
    error = np.linalg.norm(normalized_to_colmap(back, (w,h)) - (src+.5), axis=1)
    valid = np.isfinite(error) & (error <= cycle_px) & (back_score >= confidence)
    return ((src[valid]+.5) * np.asarray(source_size)/[w,h],
            normalized_to_colmap(dst[valid], target_size), scores[valid])


class KeypointIndex:
    """Consolidate subpixel observations in small spatial cells per image.

    Bounded approximation for this custom SfM adapter, not the paper's released
    track sampler. Store averaged coordinates rather than moving points to grid
    centers. COLMAP subsequently verifies edges and constructs valid tracks.
    """
    def __init__(self, cell_size=1.):
        if cell_size <= 0:
            raise ValueError('cell_size must be positive')
        self.cell_size = cell_size
        self.lookup = {}
        self.sums = []
        self.counts = []

    def add(self, points):
        points = np.asarray(points)
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite keypoint')
        ids = []
        for xy in points:
            key = tuple(np.floor(xy/self.cell_size).astype(int))
            idx = self.lookup.get(key)
            if idx is None:
                idx = len(self.sums)
                self.lookup[key] = idx
                self.sums.append(np.array(xy,dtype=float))
                self.counts.append(1)
            else:
                self.sums[idx] += xy
                self.counts[idx] += 1
            ids.append(idx)
        return np.asarray(ids,dtype=np.uint32)

    def points(self):
        return (np.asarray(self.sums).reshape(-1,2) /
                np.asarray(self.counts)[:,None]).astype(np.float32)


def checkpoint_for_model(weights, expected_keys):
    """Allow only obsolete checkpoint branches observed in the official release."""
    import re
    missing=set(expected_keys)-set(weights)
    extras=set(weights)-set(expected_keys)
    allowed=re.compile(r'^(conv_backbones\.[1248]\..+|tracktention_modules\.\d+\.(sampler|splatter)\.attn\.rope\.periods)$')
    unknown=[k for k in extras if not allowed.fullmatch(k)]
    if missing or unknown:
        raise ValueError(f'Incompatible checkpoint: missing={sorted(missing)}, unexpected={sorted(unknown)}')
    return {k:weights[k] for k in expected_keys},sorted(extras)


def validate_cache(root, signature):
    """Never bless unprovenanced prediction files as belonging to a new run."""
    import json
    root.mkdir(parents=True,exist_ok=True)
    config=root/'config.json'
    if config.exists():
        if json.loads(config.read_text()) != signature:
            raise ValueError('Cache provenance differs; use a new cache directory')
    else:
        if any(root.iterdir()):
            raise ValueError('Nonempty cache has no provenance; use a new cache directory')
        config.write_text(json.dumps(signature,indent=2)+'\n')
