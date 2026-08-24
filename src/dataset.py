"""
Loading and normalisation for the Schuit et al. (2023) labelled TROPOMI scenes,
and for the CO scenes you extract yourself.

Zenodo record 13903869 (CC-BY 4.0) contains three NetCDF files:

    CNN_pos_trainingdata.nc    828 scenes labelled "plume_structures"
    CNN_neg_trainingdata.nc  2,242 scenes labelled "no_plume"
    SVC_trainingdata.nc        843 scenes labelled plume / artefact / empty

Each holds 13 spatial channels shaped [N, 32, 32] plus metadata variables
(manual_label, orbit_number, unique_identifier). The CNN in the paper was
trained on the `xch4` channel alone; the other twelve channels feed the
support-vector classifier in stage two.

THE NORMALISATION IS THE CRUX OF THIS PROJECT. The paper does not spell out
its exact scheme, so `normalise_scene` below implements a robust per-scene
standardisation: subtract the scene median, divide by a robust spread. That
choice is what makes a CH4-trained network applicable to a CO field at all --
after normalisation both are dimensionless anomaly maps. Check this against
the paper before you present anything, and say which scheme you used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def normalise_scene(scene: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Robust per-scene standardisation -> dimensionless anomaly map.

    Uses median and the interquartile range rather than mean and standard
    deviation, so that a strong plume occupying part of the scene does not
    inflate the scale it is being measured against.

    NaNs (bad retrievals, cloud, missing pixels) are filled with 0 *after*
    normalisation, i.e. treated as "no anomaly". This is a modelling decision
    worth stating: it biases the network towards missing plumes that are
    partly cloud-masked, rather than towards false positives at cloud edges.
    """
    scene = np.asarray(scene, dtype=np.float32)
    finite = np.isfinite(scene)
    if finite.sum() < 16:                          # essentially empty scene
        return np.zeros_like(scene, dtype=np.float32)

    values = scene[finite]
    median = np.median(values)
    q75, q25 = np.percentile(values, [75, 25])
    scale = max(float(q75 - q25), eps)

    out = (scene - median) / scale
    out[~finite] = 0.0
    return out.astype(np.float32)


def normalise_batch(scenes: np.ndarray) -> np.ndarray:
    """Apply `normalise_scene` over the leading axis of an [N, 32, 32] array."""
    return np.stack([normalise_scene(s) for s in scenes])


# ---------------------------------------------------------------------------
# augmentation (the paper augments 8x: 4 rotations x 2 flips)
# ---------------------------------------------------------------------------


def augment_eightfold(scenes: np.ndarray, labels: np.ndarray):
    """Expand [N, 32, 32] -> [8N, 32, 32] by the dihedral group of the square.

    Schuit et al. augment their 3,070 scenes to 19,648 this way. Plume
    morphology has no preferred orientation on a satellite grid, so this is
    a physically justified augmentation rather than a generic trick.
    """
    out_scenes, out_labels = [], []
    for k in range(4):
        rotated = np.rot90(scenes, k=k, axes=(1, 2))
        out_scenes.append(rotated)
        out_scenes.append(np.flip(rotated, axis=2))
        out_labels.append(labels)
        out_labels.append(labels)
    return (
        np.ascontiguousarray(np.concatenate(out_scenes)),
        np.concatenate(out_labels),
    )


# ---------------------------------------------------------------------------
# NetCDF loading
# ---------------------------------------------------------------------------


def load_channel(path: str | Path, channel: str = "xch4") -> np.ndarray:
    """Read one [N, 32, 32] channel out of a Zenodo training file."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if channel not in ds:
            raise KeyError(
                f"channel {channel!r} not in {Path(path).name}. "
                f"available: {sorted(ds.data_vars)}"
            )
        return ds[channel].values.astype(np.float32)


def load_cnn_training_set(data_dir: str | Path, channel: str = "xch4"):
    """Load positives + negatives into (scenes, labels) with labels in {0, 1}."""
    data_dir = Path(data_dir)
    pos = load_channel(data_dir / "CNN_pos_trainingdata.nc", channel)
    neg = load_channel(data_dir / "CNN_neg_trainingdata.nc", channel)

    scenes = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    print(f"loaded {len(pos)} positive and {len(neg)} negative scenes")
    return scenes, labels


# ---------------------------------------------------------------------------
# torch Dataset
# ---------------------------------------------------------------------------


class SceneDataset(Dataset):
    """Normalised 32x32 scenes with binary labels."""

    def __init__(self, scenes: np.ndarray, labels: np.ndarray, normalise: bool = True):
        if normalise:
            scenes = normalise_batch(scenes)
        self.scenes = torch.from_numpy(np.ascontiguousarray(scenes)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).float()

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int):
        return self.scenes[idx], self.labels[idx]
