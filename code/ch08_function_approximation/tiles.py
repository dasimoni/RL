"""Tile coding from scratch (Chapter 08, Section 6.5).

A tile coder maps a continuous state s in a box [lows, highs] to the indices of
`n_tilings` active binary features -- one tile per tiling. All tilings are the
same uniform grid, each displaced by a fraction of a tile width. We use the
*asymmetric* displacement recommended by Miller & Glanz (1996) and in Sutton &
Barto Sec. 9.5.4: tiling j (j = 0..n-1) is shifted by j * (1, 3, 5, ...) / n
tile widths along dimensions (1, 2, 3, ...). Uniform displacements (j, j, ...)/n
put all tilings along the diagonal and create diagonal artefacts.

Hashing (optional): when the grid is huge (many dimensions), we pseudo-randomly
fold the (tiling, tile-coordinates[, action]) index into a table of `hash_size`
entries. Distant tiles can then *collide* and share a weight; with a table much
larger than the number of tiles actually visited, collisions are rare and
harmless. Two hashing schemes are provided:

* ``"mix"``  -- a stateless integer hash (fast, collisions are random);
* ``"iht"``  -- an index hash table like Sutton's tiles3: each new tile gets the
  next free index; only once the table is full do we fall back to the mix hash.

Running this file performs self-tests and prints a small demo.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


def _mix32(x: np.ndarray) -> np.ndarray:
    """A standard 32-bit integer avalanche hash (works element-wise on uint64)."""
    x = x.astype(np.uint64) & np.uint64(0xFFFFFFFF)
    x = ((x >> np.uint64(16)) ^ x) * np.uint64(0x45D9F3B) & np.uint64(0xFFFFFFFF)
    x = ((x >> np.uint64(16)) ^ x) * np.uint64(0x45D9F3B) & np.uint64(0xFFFFFFFF)
    return (x >> np.uint64(16)) ^ x


class TileCoder:
    def __init__(self, lows, highs, n_tilings: int = 8, tiles_per_dim=8,
                 offsets: str = "asymmetric", hash_size: int | None = None,
                 hashing: str = "mix"):
        self.lows = np.asarray(lows, dtype=float)
        self.highs = np.asarray(highs, dtype=float)
        self.k = len(self.lows)
        self.n_tilings = n_tilings
        tpd = np.broadcast_to(np.asarray(tiles_per_dim), (self.k,)).astype(int)
        # scale so that one tile width = 1 unit
        self.scale = tpd / (self.highs - self.lows)
        if offsets == "asymmetric":
            disp = np.arange(1, 2 * self.k, 2)                 # (1, 3, 5, ...)
        elif offsets == "uniform":
            disp = np.ones(self.k)
        else:
            raise ValueError(offsets)
        # offset of tiling j in tile units, wrapped into [0, 1)
        self.offsets = (np.arange(n_tilings)[:, None] * disp[None, :] / n_tilings) % 1.0
        # a shifted tiling needs one extra tile per dimension to cover the box
        self.dims = tpd + 1
        self.dmax = self.dims - 1
        self.strides = np.cumprod(np.concatenate([[1], self.dims[:-1]])).astype(np.int64)
        self.tiles_per_tiling = int(np.prod(self.dims))
        self.base = np.arange(n_tilings, dtype=np.int64) * self.tiles_per_tiling
        self.n_grid_features = n_tilings * self.tiles_per_tiling
        self.hash_size = hash_size
        self.hashing = hashing
        self.n_features = self.n_grid_features if hash_size is None else hash_size
        self._iht: dict[int, int] = {}
        self.collisions = 0                                    # IHT overflows

    def _grid_indices(self, s) -> np.ndarray:
        u = (np.asarray(s, dtype=float) - self.lows) * self.scale       # in tile units
        coords = np.floor(u + self.offsets).astype(np.int64)            # (n_tilings, k)
        np.clip(coords, 0, self.dmax, out=coords)                       # guard the box edges
        return self.base + coords @ self.strides                        # flat index per tiling

    def active(self, s, action: int | None = None) -> np.ndarray:
        """Indices of the n_tilings active features for state s (and optionally an action
        folded into the hash, as in Sutton's tiles3)."""
        idx = self._grid_indices(s)
        if self.hash_size is None:
            return idx
        if action is not None:
            idx = idx + action * self.n_grid_features
        if self.hashing == "mix":
            return (_mix32(idx) % np.uint64(self.hash_size)).astype(np.int64)
        out = np.empty_like(idx)
        for i, key in enumerate(idx.tolist()):
            j = self._iht.get(key)
            if j is None:
                if len(self._iht) < self.hash_size:
                    j = self._iht[key] = len(self._iht)
                else:                                   # table full: fall back to collisions
                    self.collisions += 1
                    j = int(_mix32(np.array([key]))[0] % np.uint64(self.hash_size))
            out[i] = j
        return out


# --------------------------------------------------------------------------- #
def _self_test(seed=0):
    print(f"seed={seed} (random test states for the self-test)")
    tc = TileCoder([0.0, 0.0], [1.0, 1.0], n_tilings=8, tiles_per_dim=4)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        s = rng.random(2)
        a = tc.active(s)
        assert a.shape == (8,) and len(set(a.tolist())) == 8          # one tile per tiling
        assert np.all(a // tc.tiles_per_tiling == np.arange(8))       # tile j lies in tiling j
        assert a.max() < tc.n_features
    # corners of the box are valid
    tc.active([0.0, 0.0]); tc.active([1.0, 1.0])
    # generalisation: nearby states share many tiles, distant states share none
    near = len(set(tc.active([0.50, 0.50])) & set(tc.active([0.52, 0.50])))
    far = len(set(tc.active([0.10, 0.10])) & set(tc.active([0.90, 0.90])))
    assert near >= 6 and far == 0, (near, far)
    # hashing keeps indices in range; the IHT gives distinct indices until full
    th = TileCoder([0, 0], [1, 1], 8, 4, hash_size=64, hashing="iht")
    for _ in range(1000):
        assert th.active(rng.random(2)).max() < 64
    print(f"self-test passed: near pair shares {near}/8 tiles, far pair shares {far}/8;"
          f" IHT with 64 slots for {tc.n_grid_features} grid tiles overflowed {th.collisions} times")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.parse_args()
    _self_test()
    tc = TileCoder([-1.2, -0.07], [0.6, 0.07], n_tilings=8, tiles_per_dim=8)
    print(f"Mountain Car coder: {tc.n_tilings} tilings x {tc.tiles_per_tiling} tiles"
          f" = {tc.n_grid_features} features per action")
    print("active tiles at (-0.5, 0.0):", tc.active([-0.5, 0.0]))
