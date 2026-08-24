"""
Pull 32x32 TROPOMI CO tiles (plus co-located NO2 and MODIS fire radiative
power) around a list of target facilities, using Google Earth Engine.

    earthengine authenticate            # once
    python src/fetch_co_tiles.py --targets targets/facilities.csv \
        --start 2019-01-01 --end 2019-12-31 --out data/co_tiles.npz

WHY EARTH ENGINE, AND WHAT IT COSTS YOU
---------------------------------------
GEE is the fastest route to CO fields in a two-day sprint: no downloads, no
HARP regridding, ~20 lines of code. The price is that COPERNICUS/S5P/OFFL/L3_CO
is a *regridded* product, not the native L2 that SRON actually works with. It
has already had a qa_value filter applied and you do not get the averaging
kernel, the a-priori profile, or the retrieval diagnostics.

READ THE RESOLUTION NUMBER CAREFULLY. The Earth Engine catalogue lists
"1113.2 meters" for this collection. That is the GRID SPACING of the
harpconvert bin_spatial output -- 0.01 degrees at the equator -- not the
resolution. The real information content is the native TROPOMI CO ground
pixel: 7.0 x 7.0 km (across x along track) at nadir at launch, and
7.0 x 5.5 km since 6 August 2019. So the L3 grid oversamples the true
footprint by roughly six times in each direction. Nothing is gained by
sampling this product finer than about 7 km, which is why PIXEL_KM below is
set to 7 and not to 1.1.

Earth Engine also applies qa_value > 0.5 BEFORE gridding, so that decision
has been made for you and cannot be revisited here. See fetch_co_l2.py for
the native-geometry alternative, where it is yours to make.

Say this out loud in your README. SRON built the operational L2 CO retrieval --
they will spot the difference immediately, and pre-empting it reads as
competence. The honest framing is: "prototyped on the harmonised L3 product
for speed; the same pipeline runs on L2 with a HARP regridding step in front."

WHY NO2 AND FIRE COUNTS ARE HERE TOO
------------------------------------
Van der Velde et al. (2021, ACP 21, 597) use the TROPOMI CO:NO2 column ratio
as an observed proxy for combustion efficiency -- low for smouldering
biomass burning, high for efficient flaming. That gives you a physically
motivated way to separate an industrial CO plume from a fire plume, which is
the main contaminant in any global CO catalogue built from 2018 onwards.
MODIS FRP is the independent check on whether a fire was actually there.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date as date_cls, timedelta
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

TILE = 32
PIXEL_KM = 7.0                     # native TROPOMI CO across-track pixel at nadir
KM_PER_DEG = 111.32


SETUP_HELP = """
Earth Engine one-time setup (about ten minutes, free for research use)
---------------------------------------------------------------------

STEP 1  Sign in to Google Cloud and create or pick a project
        https://console.cloud.google.com/earth-engine
        Use a personal Google account if your institution's account is not
        allowed to create Cloud projects. Note the PROJECT ID -- it is the
        short string like 'ee-vikranttomar', not the display name.

STEP 2  Register the project for NONCOMMERCIAL use on that same page
        and answer the eligibility questionnaire. Since September 2025 this
        questionnaire is mandatory; without it the API returns permission
        errors even after you authenticate. No billing card is needed for a
        noncommercial project.

STEP 3  Authenticate this computer. A browser window opens; approve it.
            earthengine authenticate
        If the 'earthengine' command is not found, use:
            python -c "import ee; ee.Authenticate()"

STEP 4  Tell the tools which project to use. Do this once and you never
        need the --project flag again:
            earthengine set_project YOUR_PROJECT_ID

        The alternatives, if you prefer not to store it:
            export EARTHENGINE_PROJECT=YOUR_PROJECT_ID
            python src/fetch_co_tiles.py --project YOUR_PROJECT_ID

STEP 5  Check it worked:
            python -c "import ee; ee.Initialize(); print(ee.Number(1).getInfo())"
        A printed 1 means everything is in place.
"""


def _stored_project() -> str | None:
    """The project saved by `earthengine set_project`, if any.

    It lives as a "project" key in ~/.config/earthengine/credentials.
    """
    try:
        from ee import oauth

        path = Path(oauth.get_credentials_path())
        if path.exists():
            return json.loads(path.read_text()).get("project") or None
    except Exception:
        pass
    return None


def _credentials_exist() -> bool:
    try:
        from ee import oauth

        return Path(oauth.get_credentials_path()).exists()
    except Exception:
        return False


def _init_ee(project: str | None):
    try:
        import ee
    except ImportError:
        sys.exit("earthengine-api not installed:  pip install earthengine-api")

    if not _credentials_exist():
        sys.exit("Not authenticated yet -- you have not run step 3.\n" + SETUP_HELP)

    # --project  >  EARTHENGINE_PROJECT  >  earthengine set_project
    source = "--project"
    if not project:
        project, source = os.environ.get("EARTHENGINE_PROJECT"), "EARTHENGINE_PROJECT"
    if not project:
        project, source = _stored_project(), "earthengine set_project"

    if not project:
        sys.exit(
            "Authenticated, but no Cloud project is set -- you have not run step 4.\n"
            + SETUP_HELP
        )

    try:
        ee.Initialize(project=project)
        # Initialize can succeed lazily, so force one real round trip.
        ee.Number(1).getInfo()
    except Exception as exc:
        text = str(exc)
        low = text.lower()

        if any(s in low for s in ("proxy", "max retries", "connection", "timed out",
                                  "temporary failure in name resolution")):
            hint = ("-> This is a network problem, not an Earth Engine one. Check your\n"
                    "   internet connection, VPN, or institutional proxy, then retry.\n")
        elif any(s in low for s in ("has not been used in project", "not registered",
                                    "permission_denied", "permission denied",
                                    "caller does not have permission",
                                    "earth engine api")):
            hint = ("-> The project exists but is not set up for Earth Engine. Either the\n"
                    "   Earth Engine API is not enabled on it, or the noncommercial\n"
                    "   questionnaire (step 2) has not been filled in. Both are on the\n"
                    "   console page in step 1.\n")
        elif any(s in low for s in ("invalid_grant", "credentials", "unauthenticated",
                                    "reauth", "token")):
            hint = ("-> Your saved login has expired or was revoked. Run:\n"
                    "       earthengine authenticate --force\n")
        elif "not found" in low or "does not exist" in low:
            hint = (f"-> No project called '{project}'. Use the short PROJECT ID from the\n"
                    "   Cloud console, not the display name.\n")
        else:
            hint = "-> Full error text is above; the setup steps below cover the usual causes.\n"

        sys.exit(
            f"Earth Engine rejected the connection.\n\n"
            f"  project : {project}   (from {source})\n"
            f"  error   : {text[:400]}\n\n"
            + hint + SETUP_HELP
        )

    print(f"Earth Engine ready (project '{project}', from {source})")
    return ee


def _pixel_size_deg(lat: float) -> tuple[float, float]:
    """Degrees per pixel in (lon, lat) for a PIXEL_KM square ground cell."""
    dlat = PIXEL_KM / KM_PER_DEG
    dlon = dlat / max(np.cos(np.radians(lat)), 0.2)
    return dlon, dlat


def _tile_region(ee, lat: float, lon: float):
    """A TILE x TILE box at roughly PIXEL_KM resolution, centred on the site."""
    dlon, dlat = _pixel_size_deg(lat)
    half_deg_lon, half_deg_lat = (TILE / 2) * dlon, (TILE / 2) * dlat
    return ee.Geometry.Rectangle(
        [lon - half_deg_lon, lat - half_deg_lat, lon + half_deg_lon, lat + half_deg_lat],
        proj="EPSG:4326",
        geodesic=False,
    )


def _sample(ee, image, region, band: str, lat: float, lon: float) -> np.ndarray | None:
    """Return a [<=TILE, <=TILE] array for one band, or None if empty.

    WHY crsTransform AND NOT scale -- THIS WAS A REAL BUG
    -----------------------------------------------------
    The obvious call is `.reproject(crs="EPSG:4326", scale=PIXEL_KM * 1000)`,
    and it is wrong away from the equator. Earth Engine converts a metre
    `scale` inside a geographic CRS using metres-per-degree AT THE EQUATOR, so
    every pixel comes out 0.0629 deg square regardless of latitude. Meanwhile
    _tile_region widens the longitude half-width by 1/cos(lat) to keep the box
    square in kilometres. Nothing then undoes that widening, so the returned
    array is TILE rows tall but 32/cos(lat) columns wide -- 34 columns at
    Bhilai (21 N), 51 at Duisburg (51.5 N).

    _pad_to_tile used to keep arr[:32, :32], i.e. the WESTERN 32 columns, which
    left the facility at column ~25 instead of ~16. Every European site was
    scored, stacked and centre-excess'd about 65 km west of the actual plant,
    which is most of the reason European steel looked quieter than the Sahara.

    An explicit crsTransform pins the grid instead: [dlon, 0, x0, 0, -dlat, y0]
    gives pixels that really are PIXEL_KM square on the ground, an array that
    really is TILE x TILE, and an origin that really is centred on the site.
    """
    dlon, dlat = _pixel_size_deg(lat)
    x0, y0 = lon - (TILE / 2) * dlon, lat + (TILE / 2) * dlat
    try:
        arr = (
            image.select(band)
            .reproject(crs="EPSG:4326",
                       crsTransform=[dlon, 0, x0, 0, -dlat, y0])
            .sampleRectangle(region=region, defaultValue=-9999)
            .get(band)
            .getInfo()
        )
    except Exception:
        # Let the caller decide how loud to be. A per-tile print here turned
        # one bad date into thousands of lines of scrollback.
        raise
    if arr is None:
        return None
    out = np.array(arr, dtype=np.float32)
    out[out == -9999] = np.nan
    return out


def _sample_days(start: str, end: str, n: int) -> list[str]:
    """n dates spread evenly over [start, end], inclusive.

    Sampling DAYS rather than orbits is the important design choice here.
    Earth Engine gives every S5P orbit a near-global bounding box -- a
    pole-to-pole track wraps all longitudes near the poles -- so filterBounds
    matches roughly every orbit of the year for any site. Striding that list
    picks the same handful of orbit IDs for every site, most of which do not
    actually cover it. That is how a run ends up with 45 tiles landing on 7
    days, Europe sampled only in summer and India only in spring: region
    confounded with season, which ruins the very comparison the experiment
    rests on.

    Choosing the days up front fixes both at once -- identical temporal
    sampling everywhere, and a daily mosaic that actually covers the site.
    """
    d0 = date_cls.fromisoformat(start)
    d1 = date_cls.fromisoformat(end)
    span = (d1 - d0).days
    if n <= 1 or span <= 0:
        return [start]
    step = span / (n - 1)
    return [(d0 + timedelta(days=round(i * step))).isoformat() for i in range(n)]



def _date_from_index(image_id: str) -> str | None:
    """'20190101T003745_20190101T021915'  ->  '2019-01-01'.

    Earth Engine's ee.Date() needs ISO 'YYYY-MM-DD'. The first version of this
    file passed the raw 'YYYYMMDD' stamp straight through, which ee.Date()
    rejects -- that is why every NO2 and MODIS lookup raised EEException while
    the CO tiles (which never touch the date) came back fine.
    """
    digits = re.match(r"(\d{8})", image_id)
    if not digits:
        return None
    ymd = digits.group(1)
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _daily_image(ee, collection_id: str, band: str, date: str, region, how: str):
    """One day of a collection over the region, reduced, never empty.

    An empty ImageCollection reduces to an image with no bands, and .select()
    on that throws. ee.Algorithms.If picks a fully-flagged blank instead --
    server-side, so it costs no extra round trip.
    """
    coll = (
        ee.ImageCollection(collection_id)
        .select(band)
        .filterDate(date, ee.Date(date).advance(1, "day"))
        .filterBounds(region)
    )
    reduced = coll.max() if how == "max" else coll.mean()
    blank = ee.Image.constant(-9999).rename(band).toFloat()
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), reduced, blank))


def _pad_to_tile(arr: np.ndarray) -> np.ndarray:
    """Centre-crop or NaN-pad an arbitrary array to exactly TILE x TILE.

    With the crsTransform in _sample this should already be TILE x TILE, so
    this is a guard rather than a workhorse. It crops from the CENTRE, not the
    top-left: the facility is at the middle of whatever Earth Engine returns,
    and the old top-left crop is what silently moved every high-latitude site
    off its own plant.
    """
    out = np.full((TILE, TILE), np.nan, dtype=np.float32)
    h, w = min(arr.shape[0], TILE), min(arr.shape[1], TILE)
    r0, c0 = (arr.shape[0] - h) // 2, (arr.shape[1] - w) // 2       # source
    r1, c1 = (TILE - h) // 2, (TILE - w) // 2                       # destination
    out[r1:r1 + h, c1:c1 + w] = arr[r0:r0 + h, c0:c0 + w]
    return out


def read_targets(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if not r["name"].startswith("#")]
    for r in rows:
        r["lat"], r["lon"] = float(r["lat"]), float(r["lon"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="targets/facilities.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--out", default="data/co_tiles.npz")
    ap.add_argument("--project", default=None, help="Earth Engine cloud project id")
    ap.add_argument("--days", type=int, default=18,
                    help="how many days to sample, spread evenly across the date "
                         "range. Every site is sampled on the SAME days, so region "
                         "is never confounded with season.")
    ap.add_argument("--min-valid", type=float, default=0.5,
                    help="reject a tile with less than this fraction of real "
                         "pixels. Large NaN holes are filled with zeros and make "
                         "artificial straight edges that look like structure.")
    ap.add_argument("--no-extras", action="store_true",
                    help="skip NO2 and MODIS fire power. Three times faster, but "
                         "you lose the fire discriminator in predict_co.py")
    args = ap.parse_args()

    ee = _init_ee(args.project)
    targets = read_targets(args.targets)
    print(f"{len(targets)} targets, {args.start} to {args.end}")

    co_tiles, no2_tiles, frp_tiles, meta = [], [], [], []
    extras_missing = {"no2": 0, "frp": 0}   # queried fine, no data there
    extras_failed = {"no2": 0, "frp": 0}    # the call itself errored
    first_error: dict[str, str] = {}
    co_failed = 0

    days = _sample_days(args.start, args.end, args.days)
    print(f"sampling {len(days)} days: {days[0]} .. {days[-1]}")
    print("(every site on the same days, so region is not confounded with season)\n")

    for site in targets:
        region = _tile_region(ee, site["lat"], site["lon"])
        kept = 0

        for date in days:
            try:
                co = _sample(
                    ee,
                    _daily_image(ee, "COPERNICUS/S5P/OFFL/L3_CO",
                                 "CO_column_number_density", date, region, "mean"),
                    region, "CO_column_number_density", site["lat"], site["lon"])
            except Exception as exc:
                co_failed += 1
                first_error.setdefault("co", f"{type(exc).__name__}: {exc}")
                continue
            if co is None or np.isfinite(co).sum() < args.min_valid * TILE * TILE:
                continue                            # empty, or mostly hole


            # Same-day NO2 and fire radiative power, for the fire discriminator
            # of van der Velde et al. (2021). Skipped entirely with --no-extras,
            # which cuts the run to a third of the round trips.
            no2 = frp = None
            if not args.no_extras:
                try:
                    no2 = _sample(
                        ee,
                        _daily_image(ee, "COPERNICUS/S5P/OFFL/L3_NO2",
                                     "tropospheric_NO2_column_number_density",
                                     date, region, "mean"),
                        region, "tropospheric_NO2_column_number_density",
                        site["lat"], site["lon"])
                    if no2 is None:
                        extras_missing["no2"] += 1
                except Exception as exc:
                    extras_failed["no2"] += 1
                    first_error.setdefault("no2", f"{type(exc).__name__}: {exc}")
                try:
                    frp = _sample(
                        ee,
                        _daily_image(ee, "MODIS/061/MOD14A1", "MaxFRP",
                                     date, region, "max"),
                        region, "MaxFRP", site["lat"], site["lon"])
                    if frp is None:
                        extras_missing["frp"] += 1
                except Exception as exc:
                    extras_failed["frp"] += 1
                    first_error.setdefault("frp", f"{type(exc).__name__}: {exc}")

            co_tiles.append(_pad_to_tile(co))
            no2_tiles.append(_pad_to_tile(no2) if no2 is not None
                             else np.full((TILE, TILE), np.nan, np.float32))
            frp_tiles.append(_pad_to_tile(frp) if frp is not None
                             else np.full((TILE, TILE), np.nan, np.float32))
            meta.append((site["name"], site.get("kind", ""), site["lat"],
                         site["lon"], date))
            kept += 1

        print(f"  {site['name']:28s} {kept:3d}/{len(days)} days")

    if not co_tiles:
        sys.exit("no tiles retrieved -- check authentication, dates and target list")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        co=np.stack(co_tiles),
        no2=np.stack(no2_tiles),
        frp=np.stack(frp_tiles),
        meta=np.array(meta, dtype=object),
    )
    print(f"\nsaved {len(co_tiles)} tiles -> {args.out}")

    # ---- one honest summary instead of a line per failure -----------------
    n = len(co_tiles)
    if co_failed:
        print(f"  {co_failed} CO tile(s) failed to sample")
    if args.no_extras:
        print("  NO2 and MODIS fire power skipped (--no-extras): the fire "
              "discriminator in predict_co.py will fall back to nothing")
    else:
        for key, label in (("no2", "NO2"), ("frp", "MODIS fire power")):
            got = n - extras_missing[key] - extras_failed[key]
            print(f"  {label}: {got}/{n} tiles"
                  + (f", {extras_missing[key]} with no data that day" if extras_missing[key] else "")
                  + (f", {extras_failed[key]} errored" if extras_failed[key] else ""))
    for key, msg in first_error.items():
        print(f"  first {key} error was: {msg[:200]}")

    if not args.no_extras and extras_failed["no2"] >= n:
        print("\n  Every NO2 lookup failed, so predict_co.py cannot compute the "
              "CO:NO2 ratio.\n  The detection experiment still works — the fire flag "
              "just falls back to MODIS alone.")


if __name__ == "__main__":
    main()