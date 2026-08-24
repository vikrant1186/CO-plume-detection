"""
Look at your tiles. One PNG per site, filtered however you like.

    # every tile the detector nearly rejected, one file per site
    python src/browse_tiles.py --max-score 0.3 --out figures/rejected

    # everything, for one site
    python src/browse_tiles.py --site "Bhilai Steel Plant" --out figures/bhilai

    # the confident detections only
    python src/browse_tiles.py --min-score 0.9 --out figures/confident

    # no scores yet? it still works, just without the filtering
    python src/browse_tiles.py --tiles data/co_tiles.npz --out figures/all

Each panel is annotated with date, score, the fraction of the tile that was
missing data, and a * if MODIS saw a fire. Each figure also lists the days on
which that site produced NO tile at all, because those absences are data too
-- see below.

WHY THE MISSING DAYS MATTER
---------------------------
fetch_co_tiles.py drops a site-day when the tile is empty or when less than
--min-valid of it survived the qa filter. Those drops are silent, and they
are not random:

  Central Pacific   15/18 days lost   dark ocean, the 2.3 um retrieval needs
                                      surface reflectance it does not get
  Mumbai, Dolvi,    10-12/18 lost     west-coast India: monsoon cloud
  Hazira
  Bokaro, Kosice     1/18 lost        inland, dry, nearly complete coverage

  June  50% lost                      monsoon
  Aug   54% lost                      monsoon
  Sep   12% lost                      clear season

So what survives is the clear-sky subset, and clear sky is not evenly
distributed over your sites or your calendar. Anything you conclude about
"Indian steel plants" is really about inland eastern plants in the dry
season, which is also where CO backgrounds are highest -- the same direction
as the regional confound. Say this in the README; it is the kind of selection
effect a reviewer finds in thirty seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date as date_cls, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset import normalise_scene

INK, INK2, MUTED, SURF = "#0b0b0b", "#3f3e3b", "#8a8983", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2,
})


def load(tiles: str, scores: str | None):
    d = np.load(tiles, allow_pickle=True)
    co, meta = d["co"], d["meta"]
    frp = d["frp"] if "frp" in d else np.full_like(co, np.nan)
    score = {}
    if scores and Path(scores).exists():
        for r in json.load(open(scores)):
            score[(r["facility"], r["date"])] = r["plume_score"]
    elif scores:
        print(f"note: {scores} not found — showing tiles without scores")
    return co, meta, frp, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/co_tiles.npz")
    ap.add_argument("--scores", default="co_detections.json")
    ap.add_argument("--site", default=None, help="one site name; default is every site")
    ap.add_argument("--max-score", type=float, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--sort", choices=["score", "date"], default="score")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--out", default="figures/browse")
    ap.add_argument("--vmin", type=float, default=-2.2)
    ap.add_argument("--vmax", type=float, default=3.2)
    args = ap.parse_args()

    co, meta, frp, score = load(args.tiles, args.scores)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    all_days = sorted({str(m[4]) for m in meta})
    by_site: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by_site.setdefault(str(m[0]), []).append(i)

    sites = [args.site] if args.site else sorted(by_site)
    if args.site and args.site not in by_site:
        sys.exit(f"no tiles for {args.site!r}. Available:\n  "
                 + "\n  ".join(sorted(by_site)))

    kind_col = {"steel": ORANGE, "urban": AQUA, "background": BLUE}
    made = 0
    for site in sites:
        idx = by_site[site]
        rows = []
        for i in idx:
            dt = str(meta[i][4])
            sc = score.get((site, dt))
            if sc is not None:
                if args.max_score is not None and sc > args.max_score:
                    continue
                if args.min_score is not None and sc < args.min_score:
                    continue
            rows.append((i, dt, sc))
        if not rows:
            continue
        rows.sort(key=lambda r: (r[2] if r[2] is not None else 0) if args.sort == "score" else r[1])

        got = {str(meta[i][4]) for i in idx}
        missing = [d for d in all_days if d not in got]

        n = len(rows)
        cols = min(args.cols, n)
        nrows = (n + cols - 1) // cols
        fig_h = 1.05 + nrows * 2.15
        fig = plt.figure(figsize=(2.5 * cols + 0.5, fig_h))

        kind = str(meta[idx[0]][1])
        edge = kind_col.get(kind, MUTED)
        filt = []
        if args.max_score is not None: filt.append(f"score < {args.max_score}")
        if args.min_score is not None: filt.append(f"score > {args.min_score}")
        sub = f"{n} of {len(idx)} tiles" + (f"   ({', '.join(filt)})" if filt else "")
        fig.suptitle(site, fontsize=15, fontweight="bold", color=INK, y=1 - 0.18 / fig_h)
        fig.text(0.5, 1 - 0.48 / fig_h, sub, ha="center", fontsize=9.6, color=INK2)

        top = 1 - 0.80 / fig_h
        cell_h = (top - 0.055) / nrows
        for k, (i, dt, sc) in enumerate(rows):
            r, c = divmod(k, cols)
            ax = fig.add_axes([0.02 + c * (0.96 / cols), top - (r + 1) * cell_h + 0.035,
                               0.96 / cols - 0.02, cell_h - 0.10])
            ax.imshow(normalise_scene(co[i]), cmap="inferno",
                      vmin=args.vmin, vmax=args.vmax)
            ax.plot(15.5, 15.5, marker="+", ms=8, mew=1.2, color="white", alpha=.8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(edge); sp.set_linewidth(1.4)
            holes = 100 * (1 - np.isfinite(co[i]).mean())
            fire = np.isfinite(frp[i]).any() and np.nanmax(frp[i]) > 0
            lab = f"{dt}{'  *fire' if fire else ''}"
            if sc is not None:
                lab += f"\nscore {sc:.3f}   {holes:.0f}% missing"
            else:
                lab += f"\n{holes:.0f}% missing"
            ax.set_title(lab, fontsize=7.6, color=ORANGE if fire else INK2, pad=3)

        note = f"+ marks the facility.   Colour scale fixed at {args.vmin} to {args.vmax}."
        if missing:
            note += f"\nNo tile on {len(missing)} of {len(all_days)} sampled days: " \
                    + ", ".join(missing)
        fig.text(0.5, 0.012, note, ha="center", fontsize=7.8, color=MUTED, linespacing=1.6)

        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in site)
        path = out / f"{safe}.png"
        fig.savefig(path, dpi=140, facecolor=SURF, bbox_inches="tight")
        plt.close(fig)
        print(f"  {site:28s} {n:3d} tiles"
              + (f", {len(missing)} days with none" if missing else "")
              + f"  -> {path}")
        made += 1

    if not made:
        print("nothing matched those filters")
    else:
        print(f"\n{made} file(s) in {out}/")


if __name__ == "__main__":
    main()