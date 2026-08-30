"""
Stack every scene at a site into one composite, then score it. Nothing is filled.

    python src/predict_stacked.py --model plume_cnn.pt --tiles data/co_tiles.npz

TWO RULES
---------
1. ENTRY. A scene only enters the stack if at least --min-scene-valid of its
   1024 pixels were retrieved (default 2/3 = 66.6%). A day on which half the
   tile was cloud is thrown away whole -- not partly used, not weighted down,
   discarded. What is left is a record of good days only.

2. STACKING. Each pixel of the composite is then the average of the surviving
   days that saw it. A pixel missing on Tuesday is simply taken from the days
   it exists; missing days contribute nothing at all -- not a zero, not a
   guess, not a neighbour's value.

If, after stacking every surviving day, a pixel was seen by NO day, it stays
NaN. It is not filled. It is not interpolated. It is left as a hole, and it is
drawn as a hole in the figure.

Rule 1 is a real cost and it is not evenly distributed. On the 2019 tiles it
keeps 249 of 335 scenes (74%), but 85% of the Indian steel scenes and only 48%
of the European ones -- Europe is cloudier, so Europe pays for it. Four sites
come out with just three usable days. `scenes_kept` therefore has to be read
next to every score in this output, and a composite built from three days is
not the same measurement as one built from fifteen however similar the number
next to it looks.

That last part is the change from the earlier version of this script, which
filled leftover holes with the local median. A local median is a synthetic
value: it is plausible, it looks like data, and downstream nothing can tell it
apart from a measurement. Over a tile built from very few scenes it can
manufacture smooth structure that a plume detector will call a plume -- which
is exactly what happened at the ocean control, scoring 0.99 with no source
present.

WHY NOT FILLING COSTS ALMOST NOTHING
------------------------------------
Cloud sits somewhere different on every overpass, so the holes move between
days and averaging closes nearly all of them. Filling was never doing much
work; it was just quietly available to do damage where the data was thinnest,
and it did -- the ocean control scored 0.99 on invented structure. Removing it
costs a fraction of a percent of the pixels.

Screening whole scenes on entry (rule 1) makes this better still: the days that
would have contributed the most holes are the ones now excluded.

THE ONE SUBSTITUTION THAT CANNOT BE AVOIDED, STATED PLAINLY
------------------------------------------------------------
The CNN takes a dense 32x32 array, and a convolution touching a NaN returns
NaN for the whole scene. Something must occupy the few remaining holes at the
moment of scoring.

The composite is normalised on the pixels that exist, and the holes are then
set to 0 -- meaning "no anomaly relative to this scene's own background",
which is what Schuit et al.'s published pipeline does. Zero adds no structure,
no gradient and no texture. It declares absence; it does not invent presence.

    local-median fill  ->  invents structure that looks like data   (removed)
    zero after norm    ->  declares "nothing known here"            (kept)

Everywhere else -- the stored composites, the figure, the statistics -- the
NaNs are preserved.

WHAT IS REPORTED
----------------
  * how many scenes each site started with and how many survived the screen
  * the whole-record composite score for each site, built from survivors only
  * the category plume score: the same statistics averaged over Indian steel,
    European steel, Indian cities and background controls
  * a figure of every stacked composite with its score, holes drawn as holes
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from model import PlumeCNN

EUROPEAN = {
    "ArcelorMittal Gent", "ArcelorMittal Dunkirk", "Tata Steel IJmuiden",
    "ThyssenKrupp Duisburg", "ArcelorMittal Taranto", "ArcelorMittal Gijon",
    "US Steel Kosice",
}
CATEGORIES = ["India steel", "EU steel", "India cities", "background"]


def group_of(name: str, kind: str) -> str:
    """'steel' alone conflates two regions that behave completely differently,
    so it is split by continent for the category scores."""
    if kind == "steel":
        return "EU steel" if name in EUROPEAN else "India steel"
    return "India cities" if kind == "urban" else "background"


# --------------------------------------------------------------------------
# stacking
# --------------------------------------------------------------------------
def screen(scenes: np.ndarray, min_scene_valid: float) -> np.ndarray:
    """Boolean mask of scenes clean enough to enter the stack.

    A scene is judged as a whole: if fewer than `min_scene_valid` of its 1024
    pixels were retrieved, the entire scene is discarded, not partly used.

    The reasoning is that a half-cloudy scene is not half a measurement of the
    plume. Whatever is left of it is a biased sample of the tile -- whichever
    corner happened to be clear -- and averaging it in drags the composite
    towards that corner. Better to lose the day than to average a fragment.
    """
    return np.array([np.isfinite(s).mean() >= min_scene_valid for s in scenes])


def stack(scenes: np.ndarray, min_obs: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Average each pixel over the days that saw it. Unseen pixels stay NaN.

    Returns (composite, n_obs). n_obs is how many days contributed to each
    pixel -- keep it, because a pixel resting on one observation and a pixel
    resting on fifteen are not the same measurement even though they look
    identical in the image.

    min_obs=1 is the honest default: a pixel is kept if anything saw it, and
    dropped only if nothing did. Raising it discards thin pixels, which is a
    quality decision rather than a fill, and it creates NaNs rather than
    hiding them.
    """
    finite = np.isfinite(scenes)
    n_obs = finite.sum(axis=0)
    keep = n_obs >= min_obs
    composite = np.full(scenes.shape[1:], np.nan, dtype=np.float32)
    if keep.any():
        summed = np.where(finite, np.nan_to_num(scenes), 0.0).sum(axis=0)
        composite[keep] = summed[keep] / n_obs[keep]
    return composite, n_obs.astype(np.int16)


def normalise_keep_nan(field: np.ndarray) -> np.ndarray:
    """Median-subtract and IQR-scale using only the pixels that exist.

    NaNs stay NaN. This is what gets stored and plotted.
    """
    valid = np.isfinite(field)
    if valid.sum() < 32:
        return np.full_like(field, np.nan, dtype=np.float32)
    v = field[valid]
    med = np.median(v)
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    if iqr <= 0:
        iqr = float(np.std(v)) or 1.0
    return ((field - med) / iqr).astype(np.float32)


def to_model_input(field: np.ndarray) -> np.ndarray:
    """The one place a hole must be given a number. See the module docstring.

    Normalise on the real pixels, then write 0 into the holes -- 'no anomaly'.
    No interpolation. If you change this, you are changing what the score means.
    """
    z = normalise_keep_nan(field)
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def load_model(path: str, device: str) -> PlumeCNN:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = PlumeCNN().to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def score(model, arrays: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(arrays, dtype=np.float32)).to(device)
        return torch.sigmoid(model(x)).cpu().numpy().ravel()


def auc(pos, neg) -> float:
    """P(a random positive outranks a random negative). 0.5 = no skill."""
    a, b = np.asarray(pos, float), np.asarray(neg, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


def centre_excess(field: np.ndarray, half: int = 3) -> float:
    """Normalised value in the middle of the tile, over the tile's own spread.

    Independent of the network entirely -- it is arithmetic on the composite.
    A source at the marked facility should raise it; a control should not.
    """
    z = normalise_keep_nan(field)
    c = z[16 - half:16 + half, 16 - half:16 + half]
    return float(np.nanmean(c)) if np.isfinite(c).any() else float("nan")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="plume_cnn.pt")
    ap.add_argument("--tiles", default="data/co_tiles.npz")
    ap.add_argument("--out", default="stacked_nofill.json")
    ap.add_argument("--figure", default="figures/analysis/stacked_nofill.png")
    ap.add_argument("--npz", default="stacked_composites.npz",
                    help="the composites themselves, NaNs intact")
    ap.add_argument("--min-scene-valid", type=float, default=2 / 3, metavar="F",
                    help="a scene must have at least this fraction of its 1024 "
                         "pixels retrieved or the whole scene is discarded "
                         "before stacking. Default 0.666.")
    ap.add_argument("--min-obs", type=int, default=1,
                    help="days a pixel needs before it is kept. 1 = keep "
                         "anything that was seen at all (the honest default). "
                         "Raising it drops thin pixels to NaN; it never fills.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)

    blob = np.load(args.tiles, allow_pickle=True)
    co, meta = blob["co"], blob["meta"]
    print(f"{len(co)} scenes   scene screen >= {100*args.min_scene_valid:.1f}% valid   "
          f"min-obs={args.min_obs}   nothing is filled\n")

    by_site: dict[str, list[int]] = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by_site[str(m[0])].append(i)

    # single scenes, scored the same way, so the two columns differ only by
    # the averaging and not by a change of gap rule. Only screened-in scenes
    # are counted, so the one-scene column describes the same days as the stack.
    per_scene = score(model, np.stack([to_model_input(c) for c in co]), device)

    records, composites = [], {}
    for name, idx in by_site.items():
        allscenes = co[idx]
        grp = group_of(name, str(meta[idx[0]][1]))

        # ---- rule 1: screen whole scenes on entry ------------------------
        ok = screen(allscenes, args.min_scene_valid)
        kept_idx = [i for i, k in zip(idx, ok) if k]
        scenes = allscenes[ok]
        if len(scenes) == 0:                       # nothing usable at this site
            records.append({
                "facility": name, "kind": str(meta[idx[0]][1]), "group": grp,
                "lat": float(meta[idx[0]][2]), "lon": float(meta[idx[0]][3]),
                "scenes_before": len(idx), "scenes_kept": 0,
                "scenes_dropped": len(idx), "usable": False,
                "stacked_score": None,
            })
            continue

        comp, n_obs = stack(scenes, args.min_obs)
        composites[name] = comp
        holes = int(np.isnan(comp).sum())

        rec = {
            "facility": name, "kind": str(meta[idx[0]][1]), "group": grp,
            "lat": float(meta[idx[0]][2]), "lon": float(meta[idx[0]][3]),
            "usable": True,
            "scenes_before": len(idx),
            "scenes_kept": int(ok.sum()),
            "scenes_dropped": int((~ok).sum()),
            "valid_per_scene_all": round(float(np.isfinite(allscenes).mean()), 3),
            "valid_per_scene_kept": round(float(np.isfinite(scenes).mean()), 3),
            "holes_left": holes,
            "holes_pct": round(100 * holes / comp.size, 2),
            "median_obs_per_pixel": int(np.median(n_obs)),
            "thin_pixels_pct": round(100 * float((n_obs < 3).mean()), 1),
            "per_scene_mean": round(float(per_scene[kept_idx].mean()), 4),
            "per_scene_max": round(float(per_scene[kept_idx].max()), 4),
            "stacked_score": round(float(score(model, to_model_input(comp)[None], device)[0]), 4),
            "centre_excess": round(centre_excess(comp), 3),
        }

        records.append(rec)

    records.sort(key=lambda r: -(r["stacked_score"] if r["stacked_score"] is not None else -1))
    Path(args.out).write_text(json.dumps(records, indent=2))
    np.savez_compressed(args.npz, **composites)
    usable = [r for r in records if r["usable"]]

    # ---- what the screen kept, per site -----------------------------------
    print("SCENES KEPT BY THE SCREEN")
    print(f"  {'site':24s} {'category':13s} {'before':>7s} {'kept':>5s} {'dropped':>8s} "
          f"{'valid/scene, kept':>18s}")
    for r in sorted(records, key=lambda r: (r["group"], -r["scenes_kept"])):
        vp = (f"{100*r['valid_per_scene_kept']:17.0f}%" if r["usable"] else
              f"{'--':>18s}")
        print(f"  {r['facility'][:24]:24s} {r['group']:13s} {r['scenes_before']:7d} "
              f"{r['scenes_kept']:5d} {r['scenes_dropped']:8d} {vp}")
    tb = sum(r["scenes_before"] for r in records)
    tk = sum(r["scenes_kept"] for r in records)
    print(f"  {'TOTAL':24s} {'':13s} {tb:7d} {tk:5d} {tb-tk:8d}"
          f"   ({100*tk/tb:.0f}% of scenes kept)")
    gk = collections.defaultdict(lambda: [0, 0])
    for r in records:
        gk[r["group"]][0] += r["scenes_before"]
        gk[r["group"]][1] += r["scenes_kept"]
    print("  by category: " + " | ".join(
        f"{g} {gk[g][1]}/{gk[g][0]} ({100*gk[g][1]/gk[g][0]:.0f}%)"
        for g in CATEGORIES if g in gk))
    thin = [r for r in usable if r["scenes_kept"] < 5]
    if thin:
        print(f"\n  {len(thin)} site(s) left with fewer than 5 usable days — "
              "read their scores with that in mind:")
        for r in sorted(thin, key=lambda r: r["scenes_kept"]):
            print(f"    {r['facility']:24s} {r['scenes_kept']} days, "
                  f"{r['thin_pixels_pct']:.0f}% of pixels on fewer than 3 of them")

    # ---- per-site scores --------------------------------------------------
    print(f"\n{'site':24s} {'category':13s} {'kept':>5s} {'valid/scene':>11s} "
          f"{'holes':>7s} {'one scene':>10s} {'stacked':>8s} {'centre':>7s}")
    for r in records:
        if not r["usable"]:
            print(f"{r['facility'][:24]:24s} {r['group']:13s} {0:5d}"
                  f"{'  no scene passed the screen':>50s}")
            continue
        print(f"{r['facility'][:24]:24s} {r['group']:13s} {r['scenes_kept']:5d} "
              f"{100*r['valid_per_scene_kept']:10.0f}% {r['holes_left']:5d} px "
              f"{r['per_scene_mean']:10.3f} {r['stacked_score']:8.3f} "
              f"{r['centre_excess']:+7.2f}")

    tot = sum(r["holes_left"] for r in usable)
    with_holes = [r for r in usable if r["holes_left"]]
    print(f"\n{tot} pixels of {len(usable)*1024} left unfilled "
          f"({100*tot/(len(usable)*1024):.2f}%)"
          + (", at: " + ", ".join(f"{r['facility']} ({r['holes_left']})"
                                  for r in with_holes) if with_holes else " — none"))
    print("Every other pixel in every composite is a real average of real observations.")

    # ---- category plume score ---------------------------------------------
    print("\nCATEGORY PLUME SCORE")
    gs = collections.defaultdict(lambda: {"scene": [], "whole": [], "ce": []})
    for r in usable:
        g = gs[r["group"]]
        keep = [i for i in by_site[r["facility"]]
                if np.isfinite(co[i]).mean() >= args.min_scene_valid]
        g["scene"].extend(per_scene[keep])
        g["whole"].append(r["stacked_score"])
        g["ce"].append(r["centre_excess"])

    print(f"  {'category':14s} {'sites':>6s} {'one overpass':>13s} {'whole record':>13s} "
          f"{'detected':>9s} {'centre excess':>14s}")
    for g in CATEGORIES:
        if g not in gs:
            continue
        d = gs[g]
        w = np.array(d["whole"])
        print(f"  {g:14s} {len(w):6d} {np.mean(d['scene']):13.3f} {w.mean():13.3f} "
              f"{100*(w > 0.5).mean():8.0f}% {np.nanmean(d['ce']):+14.2f}")

    # AUC on the individual scenes that passed the screen. This is the only
    # well-powered comparison here: it rests on hundreds of scenes rather than
    # on 3 background sites, so it is the number to quote.
    print("\n  AUC on single screened scenes")
    for pos, neg in [("India steel", "background"), ("India steel", "EU steel"),
                     ("India cities", "background"), ("EU steel", "background")]:
        if gs[pos]["scene"] and gs[neg]["scene"]:
            print(f"    {pos + ' vs ' + neg:32s} {auc(gs[pos]['scene'], gs[neg]['scene']):.3f}"
                  f"   ({len(gs[pos]['scene'])} vs {len(gs[neg]['scene'])} scenes)")
    print("\n  AUC on the stacked site scores  -- few sites, read with care")
    for pos, neg in [("India steel", "background"), ("India steel", "EU steel")]:
        if gs[pos]["whole"] and gs[neg]["whole"]:
            print(f"    {pos + ' vs ' + neg:32s} {auc(gs[pos]['whole'], gs[neg]['whole']):.3f}"
                  f"   ({len(gs[pos]['whole'])} vs {len(gs[neg]['whole'])} sites)")

    _plot(records, composites, args.figure, args.min_scene_valid)
    print(f"\nwrote {args.out}")
    print(f"wrote {args.npz}  (composites with NaNs intact)")


# --------------------------------------------------------------------------
def _plot(records, composites, path, min_scene_valid):
    """Every stacked composite with its plume score, on one common colour scale.

    Any pixel still NaN is drawn in flat grey so a hole reads as a hole and
    never as a measurement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, BG = "#0b0b0b", "#8a8983", "#fcfcfb"
    COL = {"India steel": "#eb6834", "EU steel": "#2a4d9e",
           "India cities": "#12a074", "background": "#2a9ad6"}
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("#b9b7ae")                     # NaN: outside the colour scale

    records = [r for r in records if r["usable"]]
    n = len(records)
    ncol = 7
    nrow = int(np.ceil(n / ncol))
    # header and legend need a fixed number of inches, not a fixed fraction,
    # or they collide once the grid gets short
    head_in, foot_in = 1.30, 0.62
    fig_h = 2.62 * nrow + head_in + foot_in
    fig = plt.figure(figsize=(2.28 * ncol, fig_h), facecolor=BG)
    gs = fig.add_gridspec(nrow, ncol, hspace=.46, wspace=.07,
                          left=.012, right=.988,
                          top=1 - head_in / fig_h, bottom=foot_in / fig_h)

    for i, r in enumerate(records):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        z = normalise_keep_nan(composites[r["facility"]])
        ax.imshow(np.ma.masked_invalid(z), cmap=cm, vmin=-1.6, vmax=2.6,
                  interpolation="nearest")
        col = COL.get(r["group"], INK)
        ax.set_title(f"{r['facility'][:23]}\nscore {r['stacked_score']:.3f}   "
                     f"n={r['scenes_kept']} of {r['scenes_before']}",
                     fontsize=9.0, color=col, pad=5, linespacing=1.35)
        ax.plot(15.5, 15.5, "+", color="white", ms=8, mew=1.3)
        if r["holes_left"]:
            ax.text(.5, -.075, f"{r['holes_left']} px unfilled", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.6, color=MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.6)

    fig.text(.5, 1 - 0.34 / fig_h, "Every site, its good days stacked — nothing filled in",
             fontsize=16, color=INK, ha="center", va="top", fontweight="bold")
    fig.text(.5, 1 - 0.66 / fig_h,
             f"Only scenes with at least {100*min_scene_valid:.0f}% of their pixels retrieved were used; "
             f"n shows how many of each site's days survived that screen.\nEach pixel is then the average "
             "of the surviving days that saw it. Sorted by plume score; + marks the facility; "
             "common colour scale.",
             fontsize=9.6, color=MUTED, ha="center", va="top", linespacing=1.5)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COL[c]) for c in CATEGORIES]
    fig.legend(handles, ["Indian steel", "European steel", "Indian cities", "Background"],
               loc="lower center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, 0.12 / fig_h))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=BG)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
