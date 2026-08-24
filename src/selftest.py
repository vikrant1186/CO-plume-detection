"""
End-to-end check on SYNTHETIC data -- run this first, before downloading
anything. It takes about a minute on a laptop CPU.

    python src/selftest.py

It builds fake 32x32 scenes: half plain noise, half noise plus a Gaussian
plume advected downwind, then trains the CNN for a few epochs. If the model
cannot separate those, something is wrong with your environment, not with
the science, and you should fix it before spending time on the real data.

This also gives you a synthetic plume generator you can reuse to sanity-check
the CO transfer step later.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SceneDataset, augment_eightfold
from model import PlumeCNN
from train import evaluate


def synthetic_plume(size: int = 32, rng: np.random.Generator | None = None) -> np.ndarray:
    """A source point plus a downwind Gaussian tail, on a noisy background."""
    rng = rng or np.random.default_rng()
    y, x = np.mgrid[0:size, 0:size]

    # random source location near the centre, random wind direction
    sy, sx = rng.uniform(10, 22, size=2)
    angle = rng.uniform(0, 2 * np.pi)
    ux, uy = np.cos(angle), np.sin(angle)

    scene = np.zeros((size, size), dtype=np.float32)
    length = rng.uniform(6, 16)
    strength = rng.uniform(0.8, 2.5)
    for step in np.linspace(0, length, 40):
        cy, cx = sy + uy * step, sx + ux * step
        width = 1.2 + 0.35 * step               # plume broadens downwind
        scene += (
            strength
            * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width ** 2))
            / (1 + 0.25 * step)                 # and dilutes
        )
    return scene


def make_dataset(n: int = 800, noise: float = 1.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    scenes, labels = [], []
    for i in range(n):
        # correlated background: a smooth gradient plus white noise, so the
        # negatives are not trivially flat
        y, x = np.mgrid[0:32, 0:32]
        background = (
            rng.uniform(-0.5, 0.5) * x / 32
            + rng.uniform(-0.5, 0.5) * y / 32
            + rng.normal(0, noise, (32, 32))
        )
        if i % 2 == 0:
            scenes.append(background + synthetic_plume(rng=rng))
            labels.append(1.0)
        else:
            scenes.append(background)
            labels.append(0.0)
    return np.array(scenes, dtype=np.float32), np.array(labels, dtype=np.float32)


def main() -> None:
    print("building synthetic scenes ...")
    scenes, labels = make_dataset(800)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(scenes))
    n_test = 160
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_scenes, train_labels = augment_eightfold(scenes[train_idx], labels[train_idx])
    train_loader = DataLoader(
        SceneDataset(train_scenes, train_labels), batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        SceneDataset(scenes[test_idx], labels[test_idx]), batch_size=64
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlumeCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        model.train()
        for batch_scenes, batch_labels in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_scenes.to(device)), batch_labels.to(device))
            loss.backward()
            optimiser.step()
        m = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch} | recall {m['recall']:.3f} "
            f"precision {m['precision']:.3f} AP {m['avg_precision']:.3f}"
        )

    final = evaluate(model, test_loader, device)
    ok = final["avg_precision"] > 0.9
    print("\nSELFTEST", "PASSED" if ok else "FAILED", f"(AP {final['avg_precision']:.3f})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
