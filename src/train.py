"""
Train the plume CNN on the Schuit et al. (2023) labelled TROPOMI CH4 scenes.

    python src/train.py --data-dir data --epochs 30

    python src/train.py --data-dir data --quick     # 3-minute sanity run first

Benchmark to beat: the published model reports recall 95.6% and precision
94.2% on held-out test data. You will probably land somewhat below that,
because the architecture here is a reconstruction and the normalisation
scheme may differ. Getting close is enough -- the point of this step is to
show the pipeline works end to end before you touch CO.

Note on the split: scenes are augmented 8x, so the split MUST happen before
augmentation. Otherwise a rotated copy of a training scene lands in the test
set and the score is meaningless. This is the single easiest way to fool
yourself here.

PROGRESS OUTPUT
---------------
Nothing in this script runs silently for more than a second or two. Every
long step prints a live bar with a running estimate of time remaining, and
each training epoch reports progress batch by batch rather than only at the
end. If output ever stops moving for more than ~30 seconds, something really
is stuck -- that is the point.

On a laptop CPU expect roughly 30-90 seconds per epoch, so a full 30-epoch
run is 20-45 minutes. The model is saved after every epoch that improves, so
you can stop with Ctrl-C at any time and keep the best result so far.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import SceneDataset, augment_eightfold, load_cnn_training_set, normalise_scene
from model import PlumeCNN, count_parameters


# ---------------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------------

IS_TTY = sys.stdout.isatty()


def log(msg: str = "") -> None:
    """Print and flush immediately, so nothing is stuck in a buffer."""
    print(msg, flush=True)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


class Progress:
    """A live progress bar that degrades gracefully.

    In a terminal it redraws one line with '\\r'. When output is piped to a
    file, or running somewhere that does not handle '\\r' (some notebooks,
    some IDE consoles), it prints a fresh line every `step_pct` instead --
    so there is always visible evidence of life.
    """

    def __init__(self, total: int, label: str, width: int = 30, step_pct: int = 10):
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self.step_pct = step_pct
        self.start = time.time()
        self.last_draw = 0.0
        self.next_line_pct = step_pct
        self.closed = False

    def update(self, done: int, suffix: str = "") -> None:
        if self.closed:
            return
        frac = min(1.0, done / self.total)
        elapsed = time.time() - self.start
        eta = (elapsed / frac - elapsed) if frac > 0.02 else float("nan")
        eta_txt = fmt_time(eta) if eta == eta else "--"
        rate = done / elapsed if elapsed > 0 else 0.0

        if IS_TTY:
            # redraw at most ~8 times a second
            if time.time() - self.last_draw < 0.12 and done < self.total:
                return
            self.last_draw = time.time()
            filled = int(self.width * frac)
            bar = "#" * filled + "." * (self.width - filled)
            cols = shutil.get_terminal_size((100, 24)).columns
            line = (f"\r  {self.label} [{bar}] {100 * frac:5.1f}%  "
                    f"{done}/{self.total}  {rate:.0f}/s  ETA {eta_txt}  {suffix}")
            sys.stdout.write(line[: cols - 1].ljust(min(cols - 1, len(line))))
            sys.stdout.flush()
        else:
            pct = 100 * frac
            if pct >= self.next_line_pct or done >= self.total:
                while self.next_line_pct <= pct:
                    self.next_line_pct += self.step_pct
                log(f"  {self.label} {pct:5.1f}%  {done}/{self.total}  "
                    f"{rate:.0f}/s  ETA {eta_txt}  {suffix}")

    def close(self, suffix: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        elapsed = time.time() - self.start
        if IS_TTY:
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size((100, 24)).columns - 1) + "\r")
            sys.stdout.flush()
        log(f"  {self.label}: done in {fmt_time(elapsed)}  {suffix}")


def normalise_with_progress(scenes: np.ndarray, label: str) -> np.ndarray:
    """dataset.normalise_batch, but it tells you how far along it is.

    This loop is pure Python over ~20,000 scenes and takes a little while.
    In the original it happened silently inside SceneDataset, which is
    exactly the kind of dead air that looks like a crash.
    """
    out = np.empty_like(scenes, dtype=np.float32)
    bar = Progress(len(scenes), label)
    for i, scene in enumerate(scenes):
        out[i] = normalise_scene(scene)
        if i % 250 == 0:
            bar.update(i)
    bar.close()
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    probs, truth = [], []
    with torch.no_grad():
        for scenes, labels in loader:
            logits = model(scenes.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            truth.append(labels.numpy())
    probs = np.concatenate(probs)
    truth = np.concatenate(truth)
    preds = (probs > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(truth, preds, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(truth, preds, zero_division=0)),
        "precision": float(precision_score(truth, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probs)) if len(set(truth)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(truth, probs)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--channel", default="xch4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="plume_cnn.pt")
    ap.add_argument("--quick", action="store_true",
                    help="600 scenes, 3 epochs, no augmentation -- proves the "
                         "pipeline runs in about three minutes")
    ap.add_argument("--max-scenes", type=int, default=0,
                    help="cap the number of scenes before augmentation (0 = all)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for torch (0 = leave as is)")
    args = ap.parse_args()

    if args.quick:
        args.max_scenes = args.max_scenes or 600
        args.epochs = min(args.epochs, 3)
        args.no_augment = True

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_start = time.time()
    log("=" * 68)
    log("  Training the plume detector on labelled TROPOMI methane scenes")
    log("=" * 68)
    log(f"  device        : {device}"
        + (f"  ({torch.get_num_threads()} CPU threads)" if device == "cpu" else ""))
    log(f"  epochs        : {args.epochs}")
    log(f"  batch size    : {args.batch_size}")
    log(f"  augmentation  : {'off' if args.no_augment else 'on (8x)'}")
    if args.quick:
        log("  MODE          : --quick, this is a sanity run, not a real result")
    log("")

    # ---- load -------------------------------------------------------------
    log("[1/5] Reading the NetCDF files ...")
    t0 = time.time()
    scenes, labels = load_cnn_training_set(args.data_dir, args.channel)
    log(f"      {len(scenes)} scenes, {int(labels.sum())} of them plumes"
        f"   ({fmt_time(time.time() - t0)})")

    if args.max_scenes and args.max_scenes < len(scenes):
        keep = rng.permutation(len(scenes))[: args.max_scenes]
        scenes, labels = scenes[keep], labels[keep]
        log(f"      capped to {len(scenes)} scenes (--max-scenes)")
    log("")

    # ---- split BEFORE augmentation ----------------------------------------
    log("[2/5] Splitting into train and test (before augmentation, on purpose) ...")
    idx = rng.permutation(len(scenes))
    n_test = max(1, int(len(scenes) * args.test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    if args.no_augment:
        train_scenes, train_labels = scenes[train_idx], labels[train_idx]
    else:
        train_scenes, train_labels = augment_eightfold(scenes[train_idx], labels[train_idx])
    test_scenes, test_labels = scenes[test_idx], labels[test_idx]
    log(f"      train {len(train_scenes)} scenes   test {len(test_scenes)} scenes")
    log("")

    # ---- normalise (this is the slow silent bit in the original) ----------
    log("[3/5] Removing the units from every scene ...")
    train_scenes = normalise_with_progress(train_scenes, "train")
    test_scenes = normalise_with_progress(test_scenes, "test ")
    log("")

    train_loader = DataLoader(
        SceneDataset(train_scenes, train_labels, normalise=False),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        SceneDataset(test_scenes, test_labels, normalise=False),
        batch_size=args.batch_size,
    )
    n_batches = len(train_loader)

    # ---- model ------------------------------------------------------------
    log("[4/5] Building the model ...")
    model = PlumeCNN().to(device)
    log(f"      {count_parameters(model):,} trainable coefficients")

    n_pos, n_neg = float((labels == 1).sum()), float((labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    log(f"      {n_batches} batches per epoch")
    log("")

    # ---- train ------------------------------------------------------------
    log("[5/5] Training. Progress updates continuously — if it stops moving")
    log("      for more than about 30 seconds, then it really is stuck.")
    log("")

    best = -1.0
    history = []
    epoch_times = []
    interrupted = False

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t_epoch = time.time()
            running, seen = 0.0, 0

            eta_run = ""
            if epoch_times:
                mean_epoch = sum(epoch_times) / len(epoch_times)
                eta_run = f"  |  whole run ETA {fmt_time(mean_epoch * (args.epochs - epoch + 1))}"
            log(f"  Epoch {epoch}/{args.epochs}{eta_run}")

            bar = Progress(len(train_loader.dataset), "    batches", step_pct=20)
            for batch_scenes, batch_labels in train_loader:
                optimiser.zero_grad()
                loss = criterion(model(batch_scenes.to(device)), batch_labels.to(device))
                loss.backward()
                optimiser.step()

                running += loss.item() * len(batch_scenes)
                seen += len(batch_scenes)
                bar.update(seen, f"loss {running / seen:.4f}")
            bar.close(f"loss {running / max(seen, 1):.4f}")
            scheduler.step()

            metrics = evaluate(model, test_loader, device)
            metrics["epoch"] = epoch
            metrics["train_loss"] = running / max(seen, 1)
            history.append(metrics)

            flag = ""
            if metrics["avg_precision"] > best:
                best = metrics["avg_precision"]
                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "epoch": epoch, "metrics": metrics}, args.out)
                flag = f"  <- best so far, saved to {args.out}"

            epoch_times.append(time.time() - t_epoch)
            log(f"    recall {metrics['recall']:.3f}   "
                f"precision {metrics['precision']:.3f}   "
                f"AP {metrics['avg_precision']:.3f}   "
                f"({fmt_time(epoch_times[-1])}){flag}")
            log("")

    except KeyboardInterrupt:
        interrupted = True
        log("")
        log("  Stopped by you (Ctrl-C). The best model so far is already saved.")

    # ---- summary ----------------------------------------------------------
    log("=" * 68)
    if history:
        final = max(history, key=lambda m: m["avg_precision"])
        log(f"  Best epoch: {final['epoch']} of {len(history)} run"
            + ("  (interrupted)" if interrupted else ""))
        log(f"    recall     {final['recall']:.3f}      (published: 0.956)")
        log(f"    precision  {final['precision']:.3f}      (published: 0.942)")
        log(f"    AP         {final['avg_precision']:.3f}")
        log(f"    confusion  tp {final['tp']}  fp {final['fp']}  "
            f"tn {final['tn']}  fn {final['fn']}")
        Path("metrics_ch4.json").write_text(
            json.dumps({"best": final, "history": history}, indent=2))
        log("")
        log(f"  Model   -> {args.out}")
        log("  Metrics -> metrics_ch4.json")
        if args.quick:
            log("")
            log("  Reminder: --quick used a small subset with no augmentation.")
            log("  Re-run without --quick for a result you can report.")
    else:
        log("  No epochs completed.")
    log(f"  Total time: {fmt_time(time.time() - t_start)}")
    log("=" * 68)


if __name__ == "__main__":
    main()