"""
Download the Schuit et al. (2023) labelled TROPOMI training scenes from Zenodo.

    python src/get_training_data.py --out data
    python src/get_training_data.py --out data --skip-svc    # 328 MB instead of 418

Zenodo record 13903869, CC-BY 4.0, ~418 MB total:

    CNN_pos_trainingdata.nc   88.4 MB    828 scenes, "plume_structures"
    CNN_neg_trainingdata.nc  239.2 MB  2,242 scenes, "no_plume"
    SVC_trainingdata.nc       89.9 MB    843 scenes, plume / artefact / empty

Attribution is a licence condition, not a courtesy. Cite the record and the
paper in your README and in anything you show SRON.

WHAT CHANGED FROM THE FIRST VERSION
-----------------------------------
- Live progress with transfer speed and time remaining, printed as fresh
  lines when output is not a terminal (the old '\\r' bar was invisible in
  notebooks and log files, so a working download looked like a hang).
- Downloads to a '.part' file and renames only on success, so an interrupted
  download is never mistaken for a complete one.
- Resumes an interrupted download instead of starting again from zero.
- Retries a few times before giving up, and tells you what it is doing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RECORD = "13903869"
BASE = f"https://zenodo.org/records/{RECORD}/files"
FILES = {
    "CNN_pos_trainingdata.nc": 88.4,
    "CNN_neg_trainingdata.nc": 239.2,
    "SVC_trainingdata.nc": 89.9,
}
CHUNK = 1 << 16          # 64 KiB
RETRIES = 4

CITATION = """
Schuit, B. J., Maasakkers, J. D., Bijl, P., Mahapatra, G., van den Berg, A.-W.,
Pandey, S., Lorente, A., Borsdorff, T., Houweling, S., Varon, D. J., McKeever,
J., Jervis, D., Girard, M., Irakulis-Loitxate, I., Gorrono, J., Guanter, L.,
Cusworth, D. H., and Aben, I.: Automated detection and monitoring of methane
super-emitters using satellite data, Atmos. Chem. Phys., 23, 9071-9098, 2023.
https://doi.org/10.5194/acp-23-9071-2023

Training data: https://doi.org/10.5281/zenodo.13903869  (CC-BY 4.0)
"""

IS_TTY = sys.stdout.isatty()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


class DownloadProgress:
    """Terminal: one redrawn line. Otherwise: a fresh line every 5%."""

    def __init__(self, total: int, already: int = 0, width: int = 30, step_pct: int = 5):
        self.total = max(1, int(total))
        self.start_bytes = already
        self.width = width
        self.step_pct = step_pct
        self.start = time.time()
        self.last_draw = 0.0
        self.next_line_pct = step_pct

    def update(self, done: int) -> None:
        frac = min(1.0, done / self.total)
        elapsed = time.time() - self.start
        moved = done - self.start_bytes
        rate = moved / elapsed if elapsed > 0.5 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("nan")
        eta_txt = fmt_time(eta) if eta == eta else "--"
        speed = f"{rate / 1e6:5.2f} MB/s" if rate else "  -- MB/s"

        if IS_TTY:
            if time.time() - self.last_draw < 0.15 and done < self.total:
                return
            self.last_draw = time.time()
            filled = int(self.width * frac)
            bar = "#" * filled + "." * (self.width - filled)
            cols = shutil.get_terminal_size((100, 24)).columns
            line = (f"\r    [{bar}] {100 * frac:5.1f}%  "
                    f"{done / 1e6:6.1f}/{self.total / 1e6:.1f} MB  {speed}  ETA {eta_txt}")
            sys.stdout.write(line[: cols - 1])
            sys.stdout.flush()
        else:
            pct = 100 * frac
            if pct >= self.next_line_pct or done >= self.total:
                while self.next_line_pct <= pct:
                    self.next_line_pct += self.step_pct
                log(f"    {pct:5.1f}%  {done / 1e6:6.1f}/{self.total / 1e6:.1f} MB  "
                    f"{speed}  ETA {eta_txt}")

    def close(self) -> None:
        if IS_TTY:
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size((100, 24)).columns - 1) + "\r")
            sys.stdout.flush()


def remote_size(url: str) -> int:
    """Content length via a HEAD request; 0 if the server will not say."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return 0


def download(url: str, target: Path, expected_mb: float) -> None:
    """Download with resume, progress and retries. Atomic on success."""
    part = target.with_suffix(target.suffix + ".part")
    total = remote_size(url) or int(expected_mb * 1e6)

    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= total > 0:
            break
        if have:
            log(f"    resuming from {have / 1e6:.1f} MB "
                f"(attempt {attempt} of {RETRIES})")

        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # server ignored the Range header -> start over
                if have and resp.status != 206:
                    have = 0
                    part.unlink(missing_ok=True)
                if not total:
                    total = have + int(resp.headers.get("Content-Length", 0))

                bar = DownloadProgress(total, already=have)
                mode = "ab" if have else "wb"
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        bar.update(have)
                bar.close()

            if total and part.stat().st_size < total:
                raise IOError(f"truncated: {part.stat().st_size} of {total} bytes")

            part.replace(target)
            log(f"    saved {target.name}  ({target.stat().st_size / 1e6:.1f} MB)")
            return

        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            log(f"    interrupted: {exc}")
            if attempt == RETRIES:
                raise
            wait = 3 * attempt
            log(f"    retrying in {wait}s ...")
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--skip-svc", action="store_true",
                    help="skip SVC_trainingdata.nc (only needed for stage two)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {n: s for n, s in FILES.items()
              if not (args.skip_svc and n.startswith("SVC"))}
    log(f"Downloading {len(wanted)} file(s), "
        f"{sum(wanted.values()):.0f} MB total, into {out_dir}/")
    log("This takes a while. Progress is printed continuously — if it stops")
    log("moving for more than about a minute, the connection has stalled.")
    log("")

    t0 = time.time()
    for i, (name, size_mb) in enumerate(wanted.items(), 1):
        target = out_dir / name
        if target.exists():
            log(f"[{i}/{len(wanted)}] {name}: already here "
                f"({target.stat().st_size / 1e6:.1f} MB), skipping")
            continue
        log(f"[{i}/{len(wanted)}] {name}  ({size_mb:.1f} MB)")
        try:
            download(f"{BASE}/{name}", target, size_mb)
        except Exception as exc:
            sys.exit(
                f"\ndownload failed after {RETRIES} attempts: {exc}\n"
                f"The partial file is kept as {target.name}.part — run this again\n"
                f"and it will resume, or fetch it by hand from\n"
                f"  https://zenodo.org/records/{RECORD}\n"
                f"into {out_dir}/"
            )

    log("")
    log(f"All files present in {out_dir}/  (total {fmt_time(time.time() - t0)})")
    log("")
    log("Cite this data:")
    log(CITATION)
    (out_dir / "CITATION.txt").write_text(CITATION.strip() + "\n")


if __name__ == "__main__":
    main()