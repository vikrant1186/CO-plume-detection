"""
Apply the CH4-trained plume CNN to TROPOMI CO tiles, and flag likely fire
contamination using the CO:NO2 ratio and MODIS FRP.

    python src/predict_co.py --model plume_cnn.pt --tiles data/co_tiles.npz

THE EXPERIMENT
--------------
The CNN saw only normalised, dimensionless anomaly maps during training. It
therefore cannot know whether it is looking at methane or carbon monoxide --
it can only key on spatial morphology. So the question is sharp and worth
asking:

    Does a plume-morphology detector trained on CH4 transfer to CO?

Both answers are publishable-in-a-README results, and both are useful to
someone building a CO detection system:

  * It transfers  -> morphology is species-agnostic; COGNITO can warm-start
                     from the existing methane model instead of labelling a
                     CO training set from scratch.
  * It does not   -> you have quantified exactly what needs retraining, and
                     why: CO plumes sit on a larger and more variable
                     background, are broader, and are pervasively confused
                     with biomass burning.

Do not decide in advance which answer you want.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import normalise_batch
from model import PlumeCNN

# DELIBERATELY NOT SET. Van der Velde et al. (2021, ACP 21, 597) show that low
# CO:NO2 column ratios indicate efficient flaming combustion and high ratios
# smouldering combustion (peat, boreal fires). The numerical threshold that
# separates an industrial plume from a fire plume depends on the column units,
# the fuel, the season and the NO2 lifetime -- there is no universal value, and
# shipping a plausible-looking constant you have not calibrated is worse than
# shipping none.
#
# Derive it from your own data: plot the CO:NO2 ratio distribution for scenes
# with MODIS FRP > 0 against scenes with FRP == 0, and put the threshold where
# they separate. Until you do, fire flagging falls back to FRP alone.
FIRE_RATIO_THRESHOLD: float | None = None


def load_model(path: str, device: str) -> PlumeCNN:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PlumeCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def co_no2_ratio(co: np.ndarray, no2: np.ndarray) -> float:
    """Scene-level enhancement ratio, using the upper decile as the plume."""
    with np.errstate(invalid="ignore"):
        co_finite, no2_finite = co[np.isfinite(co)], no2[np.isfinite(no2)]
        if co_finite.size < 32 or no2_finite.size < 32:
            return float("nan")
        co_enh = np.percentile(co_finite, 90) - np.median(co_finite)
        no2_enh = np.percentile(no2_finite, 90) - np.median(no2_finite)
        if no2_enh <= 0:
            return float("nan")
        return float(co_enh / no2_enh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="plume_cnn.pt")
    ap.add_argument("--tiles", default="data/co_tiles.npz")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="co_detections.json")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)

    blob = np.load(args.tiles, allow_pickle=True)
    co, no2, frp, meta = blob["co"], blob["no2"], blob["frp"], blob["meta"]
    print(f"{len(co)} CO tiles")

    scenes = normalise_batch(co)
    with torch.no_grad():
        logits = model(torch.from_numpy(scenes).float().to(device))
        scores = torch.sigmoid(logits).cpu().numpy()

    records = []
    for i, score in enumerate(scores):
        name, kind, lat, lon, date = meta[i]
        ratio = co_no2_ratio(co[i], no2[i])
        fire_frp = float(np.nanmax(frp[i])) if np.isfinite(frp[i]).any() else 0.0
        records.append({
            "facility": str(name),
            "kind": str(kind),
            "lat": float(lat), "lon": float(lon),
            "date": str(date),
            "plume_score": float(score),
            "detected": bool(score > args.threshold),
            "co_no2_ratio": None if np.isnan(ratio) else round(ratio, 1),
            "max_frp": round(fire_frp, 1),
            "fire_suspected": bool(
                fire_frp > 0
                or (
                    FIRE_RATIO_THRESHOLD is not None
                    and not np.isnan(ratio)
                    and ratio > FIRE_RATIO_THRESHOLD
                )
            ),
        })

    detected = [r for r in records if r["detected"]]
    fire_flagged = [r for r in detected if r["fire_suspected"]]
    print(f"detections: {len(detected)}/{len(records)} "
          f"({100 * len(detected) / len(records):.0f}%)")
    print(f"  of which fire-suspected: {len(fire_flagged)}")

    by_facility: dict[str, list[float]] = {}
    for r in records:
        by_facility.setdefault(r["facility"], []).append(r["plume_score"])
    print("\nmean plume score by facility:")
    for name, values in sorted(by_facility.items(),
                               key=lambda kv: -np.mean(kv[1])):
        print(f"  {np.mean(values):.3f}  ({len(values):3d} scenes)  {name}")

    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nwrote {args.out}")

    _plot_top(co, scores, meta, args.figures)
    _plot_fire_calibration(records, args.figures)


def _plot_top(co, scores, meta, figure_dir: str, n: int = 12) -> None:
    """Contact sheet of the highest-scoring scenes -- always eyeball these."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    order = np.argsort(-scores)[:n]

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax, idx in zip(axes.ravel(), order):
        scene = co[idx]
        finite = scene[np.isfinite(scene)]
        vmin, vmax = (np.percentile(finite, [5, 98]) if finite.size
                      else (0, 1))
        ax.imshow(scene, cmap="inferno", vmin=vmin, vmax=vmax)
        name, _, _, _, date = meta[idx]
        ax.set_title(f"{name}\n{date}  score {scores[idx]:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")

    fig.suptitle("Highest-scoring TROPOMI CO scenes (CH4-trained CNN)", fontsize=12)
    fig.tight_layout()
    out = Path(figure_dir) / "top_detections.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def _plot_fire_calibration(records: list[dict], figure_dir: str) -> None:
    """CO:NO2 ratio for fire scenes vs non-fire scenes.

    This is how you set FIRE_RATIO_THRESHOLD honestly: split by MODIS FRP,
    which is independent of the columns, and look at where the two
    distributions separate. If they overlap completely, the ratio is not a
    usable discriminator on your sample and you should say so.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fire = [r["co_no2_ratio"] for r in records
            if r["max_frp"] > 0 and r["co_no2_ratio"] is not None]
    clean = [r["co_no2_ratio"] for r in records
             if r["max_frp"] == 0 and r["co_no2_ratio"] is not None]

    if len(fire) < 3 or len(clean) < 3:
        print("fire calibration: too few scenes in one class, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.histogram_bin_edges(np.array(fire + clean), bins=30)
    ax.hist(clean, bins=bins, alpha=0.6, label=f"no fire detected (n={len(clean)})")
    ax.hist(fire, bins=bins, alpha=0.6, label=f"MODIS FRP > 0 (n={len(fire)})")
    ax.set_xlabel("CO : NO$_2$ column enhancement ratio")
    ax.set_ylabel("scenes")
    ax.set_title("Combustion-efficiency proxy, split by independent fire detection")
    ax.legend()
    fig.tight_layout()

    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    out = Path(figure_dir) / "fire_calibration.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    print(f"  median CO:NO2  fire {np.median(fire):.1f} | no fire {np.median(clean):.1f}")


if __name__ == "__main__":
    main()
