#!/usr/bin/env python3
"""
Real-data validation: Organ ROI comparisons for PyTIA outputs.

Compares ROI-level TACs and TIA/R2 across organs/labels.
Writes CSV summaries and optionally TAC CSV and simple plots.

Example:
  python scripts/validate_realdata_rois.py \
      --images tp1.nii.gz tp2.nii.gz tp3.nii.gz \
      --times 3600 14400 86400 \
      --labels organs_labels.nii.gz \
      --pytia_out ./out \
      --prefix patient01 \
      --write_tacs
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass
class RoiStats:
    label: int
    n_vox: int
    mean_tia: float
    median_tia: float
    mean_r2: float
    frac_ok: float


def load_4d(images: list[str]) -> tuple[np.ndarray, nib.Nifti1Image]:
    imgs = [nib.load(p) for p in images]
    ref = imgs[0]
    data4 = np.stack([np.asanyarray(im.dataobj).astype(np.float32) for im in imgs], axis=-1)
    return data4, ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="Timepoint activity maps (Bq/ml)")
    ap.add_argument("--times", nargs="+", required=True, type=float, help="Times in seconds (same order as images)")
    ap.add_argument("--labels", required=True, help="Label map NIfTI (int)")
    ap.add_argument("--pytia_out", required=True, help="PyTIA output directory")
    ap.add_argument("--prefix", default=None, help="Output prefix used by PyTIA (optional)")
    ap.add_argument("--out_csv", default="roi_summary.csv", help="CSV output name (in pytia_out)")
    ap.add_argument("--write_tacs", action="store_true", help="Write long-form TAC CSV")
    args = ap.parse_args()

    out_dir = Path(args.pytia_out)
    prefix = args.prefix

    def oname(base: str) -> Path:
        return out_dir / (base if prefix is None else f"{prefix}_{base}")

    tia = np.asanyarray(nib.load(str(oname("tia.nii.gz"))).dataobj).astype(np.float32)
    r2 = np.asanyarray(nib.load(str(oname("r2.nii.gz"))).dataobj).astype(np.float32)
    status = np.asanyarray(nib.load(str(oname("status_id.nii.gz"))).dataobj).astype(np.uint8)

    labels = np.asanyarray(nib.load(args.labels).dataobj).astype(np.int32)
    if labels.shape != tia.shape:
        raise ValueError("Label map shape must match PyTIA outputs.")

    data4, _ = load_4d(args.images)
    if data4.shape[:3] != tia.shape:
        raise ValueError("Activity images shape must match PyTIA outputs.")

    times = np.asarray(args.times, dtype=np.float64)
    if times.shape[0] != data4.shape[-1]:
        raise ValueError("Number of times must match number of images.")

    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]

    if args.write_tacs:
        tac_path = out_dir / "roi_tacs.csv"
        with tac_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "time_s", "mean_activity_bq_per_ml"])
            for lab in uniq:
                m = labels == lab
                if not np.any(m):
                    continue
                tac = np.mean(data4[m, :], axis=0)
                for t, a in zip(times.tolist(), tac.tolist()):
                    w.writerow([int(lab), float(t), float(a)])

    rows: list[RoiStats] = []
    for lab in uniq:
        m = labels == lab
        n = int(np.sum(m))
        if n == 0:
            continue

        tia_roi = tia[m]
        r2_roi = r2[m]
        status_roi = status[m]

        ok = status_roi == 1  # STATUS_OK
        frac_ok = float(np.mean(ok)) if ok.size else 0.0

        tia_f = tia_roi[np.isfinite(tia_roi)]
        r2_f = r2_roi[np.isfinite(r2_roi)]

        rows.append(
            RoiStats(
                label=int(lab),
                n_vox=n,
                mean_tia=float(np.mean(tia_f)) if tia_f.size else float("nan"),
                median_tia=float(np.median(tia_f)) if tia_f.size else float("nan"),
                mean_r2=float(np.mean(r2_f)) if r2_f.size else float("nan"),
                frac_ok=frac_ok,
            )
        )

    out_csv = out_dir / args.out_csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_vox", "mean_tia_bq_s", "median_tia_bq_s", "mean_r2", "frac_ok"])
        for r in rows:
            w.writerow([r.label, r.n_vox, r.mean_tia, r.median_tia, r.mean_r2, r.frac_ok])

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig = plt.figure(figsize=(10, 4))
        labs = [r.label for r in rows]
        mtia = [r.mean_tia for r in rows]
        plt.bar([str(x) for x in labs], mtia)
        plt.ylabel("Mean TIA (Bq·s/voxel)")
        plt.xlabel("Label")
        plt.title("ROI mean TIA")
        fig.tight_layout()
        fig.savefig(out_dir / "roi_mean_tia.png", dpi=150)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())