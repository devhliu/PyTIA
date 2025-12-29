from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .classify import CLASS_FALLING, CLASS_HUMP, CLASS_RISING, classify_curves
from .config import Config
from .denoise import masked_gaussian
from .io import ensure_dir, load_images, make_like, stack_4d, voxel_volume_ml
from .masking import load_mask, make_body_mask, mask_to_bool
from .metrics import r2_score
from .models.gamma_linear import fit_gamma_linear_wls, tia_from_gamma_params
from .models.hybrid import tia_trapz_plus_phys_tail
from .models.hybrid_predict import hybrid_piecewise_hat_at_samples
from .models.monoexp import fit_monoexp_tail, tia_monoexp_with_triangle_uptake
from .models.biexp import fit_biexp, tia_from_biexp_params
from .models.uptake_washout import fit_linear_uptake_monoexp_washout_matrix, tia_linear_uptake_monoexp_washout
from .models.rectangular_uptake import fit_rectangular_uptake_monoexp_washout_matrix, tia_rectangular_uptake_monoexp_washout
from .models.three_phase import fit_three_phase_exp, tia_from_three_phase_params
from .noise import clamp_negative_to_zero, compute_noise_floor, valid_mask_from_floor
from .types import Results
from .uncertainty import residual_bootstrap

STATUS_OUTSIDE = 0
STATUS_OK = 1
STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS = 2
STATUS_FIT_FAILED = 3
STATUS_ALL_BELOW_FLOOR = 4
STATUS_NONPHYSICAL = 5


STATUS_LEGEND = {
    int(STATUS_OUTSIDE): "outside mask/background",
    int(STATUS_OK): "ok",
    int(STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS): "not applicable: <2 valid points",
    int(STATUS_FIT_FAILED): "fit failed",
    int(STATUS_ALL_BELOW_FLOOR): "all points below noise floor",
    int(STATUS_NONPHYSICAL): "nonphysical parameters",
}


def _times_to_seconds(times: Sequence[float], unit: str) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    if unit == "seconds":
        return t
    if unit == "hours":
        return t * 3600.0
    raise ValueError(f"Unsupported time unit: {unit}")


def _save_summary(output_dir: Path, prefix: str | None, summary: dict[str, Any]) -> Path:
    name = "pytia_summary.yaml" if prefix is None else f"{prefix}_pytia_summary.yaml"
    path = output_dir / name
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return path


def _out_name(prefix: str | None, base: str) -> str:
    return base if prefix is None else f"{prefix}_{base}"


def _chunk_slices(n: int, chunk: int) -> list[slice]:
    if chunk is None or chunk <= 0 or chunk >= n:
        return [slice(0, n)]
    out: list[slice] = []
    for s in range(0, n, chunk):
        out.append(slice(s, min(s + chunk, n)))
    return out


def run_tia(
    images: Sequence[str | Path | Any],
    times: Sequence[float],
    config: str | Path | dict[str, Any] | None = None,
    mask: str | Path | Any | None = None,
) -> Results:
    """
    Compute time-integrated activity (TIA) from activity images.

    Supports both multi-timepoint (dynamic) and single-timepoint (static) modes:
    
    **Multi-timepoint mode** (T ≥ 2):
        - Fits exponential/gamma models to activity time-curves
        - Computes TIA via curve fitting and numerical integration
        - Supports 3 curve classes: rising, hump (gamma), falling (exponential)
        - Optionally applies physical decay tail extrapolation
        
    **Single-timepoint mode** (T = 1, `single_time.enabled=True`):
        Computes TIA = A(t) / λ_eff using one of three methods:
        
        1. **phys** (physical decay):
           Uses physical decay rate from radionuclide half-life.
           λ = ln(2) / half_life (from `physics.half_life_seconds`)
           
        2. **haenscheid** (Hänscheid method):
           Uses effective half-life of tracer in human body.
           λ = ln(2) / eff_half_life (from `single_time.haenscheid_eff_half_life_seconds`,
           falls back to `physics.half_life_seconds`)
           
        3. **prior_half_life** (prior segmentation-based):
           Supports global or label-map based half-life priors.
           - Global: λ = ln(2) / half_life (from `single_time.half_life_seconds`)
           - Label-based: λ per voxel from `single_time.label_half_lives` mapping,
             keyed by label from `single_time.label_map_path`
    
    Args:
        images: Single image or sequence of images (NIfTI paths or nibabel objects)
        times: Timepoints in specified unit (one per image)
        config: Config dict, YAML path, or None (uses defaults)
        mask: Optional mask image (path or nibabel object)
        
    Returns:
        Results object with TIA, R², sigma_TIA, model_id, status_id images and summary
    """
    import time as _time

    t0_all = _time.perf_counter()

    cfg = Config.load(config).data
    io_cfg = cfg["io"]
    prefix = io_cfg.get("prefix", None)
    out_dir = ensure_dir(io_cfg["output_dir"])
    perf_cfg = cfg.get("performance", {})
    chunk_size = int(perf_cfg.get("chunk_size_vox", 0) or 0)
    enable_prof = bool(perf_cfg.get("enable_profiling", False))

    timing_ms: dict[str, float] = {}

    t0 = _time.perf_counter()
    imgs = load_images(images)
    data4, ref = stack_4d(imgs)

    t_s = _times_to_seconds(times, cfg["time"]["unit"])
    if cfg["time"].get("sort_timepoints", True):
        order = np.argsort(t_s)
        t_s = t_s[order]
        data4 = data4[..., order]
    timing_ms["load_sort_ms"] = 1000.0 * (_time.perf_counter() - t0)

    # Denoise + clamp negatives (in density space)
    t0 = _time.perf_counter()
    data4 = clamp_negative_to_zero(data4)
    # Mask
    if mask is not None:
        mimg = mask if not isinstance(mask, (str, Path)) else load_mask(mask)
        mask3 = mask_to_bool(mimg, ref.shape[:3])
    else:
        mmode = cfg["mask"]["mode"]
        if mmode == "none":
            mask3 = np.ones(ref.shape[:3], dtype=bool)
        elif mmode == "provided":
            mimg = load_mask(cfg["mask"]["provided_path"])
            mask3 = mask_to_bool(mimg, ref.shape[:3])
        else:
            mask3 = make_body_mask(data4, min_fraction_of_max=cfg["mask"]["min_fraction_of_max"])

    if cfg["denoise"]["enabled"]:
        data4 = masked_gaussian(data4, mask3, sigma_vox=float(cfg["denoise"]["sigma_vox"]))
        data4 = clamp_negative_to_zero(data4)
    timing_ms["mask_denoise_ms"] = 1000.0 * (_time.perf_counter() - t0)

    # Convert to Bq per voxel
    vml = voxel_volume_ml(ref)
    A4 = (data4 * vml).astype(np.float32)

    # Flatten masked voxels
    idx = np.where(mask3.ravel())[0]
    n_vox = idx.size
    T = A4.shape[-1]

    # Physics
    hl = cfg["physics"]["half_life_seconds"]
    if hl is None:
        lambda_phys = None
    else:
        lambda_phys = float(np.log(2.0) / float(hl))

    # Global output buffers (masked)
    tia = np.full((n_vox,), np.nan, dtype=np.float32)
    r2 = np.full((n_vox,), np.nan, dtype=np.float32)
    sigma_tia = np.full((n_vox,), np.nan, dtype=np.float32)
    model_id = np.zeros((n_vox,), dtype=np.uint8)
    status_id = np.full((n_vox,), STATUS_OK, dtype=np.uint8)
    tpeak = np.full((n_vox,), np.nan, dtype=np.float32)

    # Single-timepoint (STP) handling
    # When single image is provided with STP enabled, compute TIA = A(t) / λ_eff
    # using one of three methods: phys (physical), haenscheid (effective), or prior_half_life (segmentation-prior)
    st_cfg = cfg.get("single_time", {})
    if T == 1 and bool(st_cfg.get("enabled", False)):
        t0 = _time.perf_counter()
        method = (st_cfg.get("method", "phys") or "phys").lower()

        # Extract activity values for masked voxels (shape: n_vox,)
        Aflat = A4.reshape((-1, T))[idx, 0].astype(np.float32)

        # Apply noise-floor filtering (same as multi-timepoint)
        nf_cfg = cfg["noise_floor"]
        if nf_cfg["enabled"]:
            floor = compute_noise_floor(
                Aflat[:, None],
                mode=nf_cfg["mode"],
                absolute=float(nf_cfg["absolute_bq_per_ml"] * vml),
                rel_frac=float(nf_cfg["relative_fraction_of_voxel_max"]),
            )
            valid = valid_mask_from_floor(Aflat[:, None], floor)[:, 0]
        else:
            valid = np.isfinite(Aflat)

        all_below = ~valid
        status_id[all_below] = STATUS_ALL_BELOW_FLOOR

        # Determine effective decay rate (λ_eff) per voxel according to method
        # Will compute TIA = A / λ_eff for each voxel
        lambda_eff = np.full((n_vox,), np.nan, dtype=np.float32)

        if method == "phys":
            # Method 1: Physical decay using radionuclide half-life
            if lambda_phys is None:
                status_id[~all_below] = STATUS_FIT_FAILED
            else:
                lambda_eff[:] = float(lambda_phys)
                model_code = 101

        elif method == "haenscheid" or method == "hanscheid":
            # Method 2: Hänscheid method using effective half-life in human body
            # Falls back to physics.half_life_seconds if explicit effective HL not provided
            eff_hl = st_cfg.get("haenscheid_eff_half_life_seconds") or cfg["physics"].get("half_life_seconds")
            if eff_hl is None:
                status_id[~all_below] = STATUS_FIT_FAILED
            else:
                lambda_eff[:] = float(np.log(2.0) / float(eff_hl))
                model_code = 102

        elif method == "prior_half_life" or method == "prior":
            # Method 3: Prior segmentation-based half-lives
            # Supports both global and label-map based mapping
            if st_cfg.get("label_map_path"):
                # Label-map mode: map voxel label -> half-life value
                lab_img = load_mask(st_cfg["label_map_path"])
                labs = np.asanyarray(lab_img.dataobj).reshape(-1)[idx].astype(np.int32)
                mapping = {int(k): float(v) for k, v in (st_cfg.get("label_half_lives") or {}).items()}
                default_hl = st_cfg.get("half_life_seconds")
                for i, lab in enumerate(labs):
                    hl_val = mapping.get(int(lab), default_hl)
                    if hl_val is None:
                        lambda_eff[i] = np.nan
                    else:
                        lambda_eff[i] = float(np.log(2.0) / float(hl_val))
                model_code = 103
            else:
                # Global mode: same half-life for all voxels
                hl_val = st_cfg.get("half_life_seconds")
                if hl_val is None:
                    status_id[~all_below] = STATUS_FIT_FAILED
                else:
                    lambda_eff[:] = float(np.log(2.0) / float(hl_val))
                    model_code = 103
        else:
            status_id[~all_below] = STATUS_FIT_FAILED

        # Compute TIA = A(t) / λ_eff where activity is valid and λ_eff is positive and finite
        ok = valid & np.isfinite(lambda_eff) & (lambda_eff > 0)
        tia_vals = np.full((n_vox,), np.nan, dtype=np.float32)
        tia_vals[ok] = (Aflat[ok] / lambda_eff[ok]).astype(np.float32)

        tia[:] = tia_vals
        model_id[ok] = np.uint8(model_code if "model_code" in locals() else 0)
        # mark invalid voxels
        bad = ~ok & ~all_below
        status_id[bad] = STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS

        timing_ms["single_time_ms"] = 1000.0 * (_time.perf_counter() - t0)

        # Unflatten to volumes and save (reuse same behaviour as multi-timepoint)
        t0 = _time.perf_counter()
        shape3 = ref.shape[:3]
        tia_vol = np.full((np.prod(shape3),), np.nan, dtype=np.float32)
        r2_vol = np.full_like(tia_vol, np.nan)
        sig_vol = np.full_like(tia_vol, np.nan)
        model_vol = np.zeros_like(tia_vol, dtype=np.uint8)
        status_vol = np.zeros_like(tia_vol, dtype=np.uint8)
        tpeak_vol = np.full_like(tia_vol, np.nan)

        status_vol[:] = STATUS_OUTSIDE
        tia_vol[idx] = tia
        r2_vol[idx] = r2
        sig_vol[idx] = sigma_tia
        model_vol[idx] = model_id
        status_vol[idx] = status_id
        tpeak_vol[idx] = tpeak

        tia_img = make_like(ref, tia_vol.reshape(shape3))
        r2_img = make_like(ref, r2_vol.reshape(shape3))
        sig_img = make_like(ref, sig_vol.reshape(shape3))
        model_img = make_like(ref, model_vol.reshape(shape3).astype(np.uint8))
        status_img = make_like(ref, status_vol.reshape(shape3).astype(np.uint8))
        tpeak_img = None
        timing_ms["assemble_ms"] = 1000.0 * (_time.perf_counter() - t0)

        # Save outputs
        t0 = _time.perf_counter()
        outputs: dict[str, Path] = {}
        import nibabel as nib  # local import

        fn_tia = _out_name(prefix, "tia.nii.gz")
        fn_r2 = _out_name(prefix, "r2.nii.gz")
        fn_sig = _out_name(prefix, "sigma_tia.nii.gz")
        fn_model = _out_name(prefix, "model_id.nii.gz")
        fn_status = _out_name(prefix, "status_id.nii.gz")

        nib.save(tia_img, str(out_dir / fn_tia)); outputs["tia"] = out_dir / fn_tia
        nib.save(r2_img, str(out_dir / fn_r2)); outputs["r2"] = out_dir / fn_r2
        nib.save(sig_img, str(out_dir / fn_sig)); outputs["sigma_tia"] = out_dir / fn_sig
        nib.save(model_img, str(out_dir / fn_model)); outputs["model_id"] = out_dir / fn_model
        nib.save(status_img, str(out_dir / fn_status)); outputs["status_id"] = out_dir / fn_status
        timing_ms["save_ms"] = 1000.0 * (_time.perf_counter() - t0)

        # Summary and return
        status_counts = {int(k): int(v) for k, v in zip(*np.unique(status_vol[idx], return_counts=True))}
        summary = {
            "pytia_version": "0.1.0",
            "times_seconds": [float(x) for x in t_s.tolist()],
            "voxel_volume_ml": float(vml),
            "status_legend": STATUS_LEGEND,
            "status_counts": {STATUS_LEGEND[int(k)]: int(v) for k, v in status_counts.items()},
            "timing_ms": timing_ms,
            "config": cfg,
        }
        if io_cfg.get("write_summary_yaml", True):
            outputs["summary"] = _save_summary(out_dir, prefix, summary)

        timing_ms["total_ms"] = 1000.0 * (_time.perf_counter() - t0_all)
        summary["timing_ms"] = timing_ms

        return Results(
            tia_img=tia_img,
            r2_img=r2_img,
            sigma_tia_img=sig_img,
            model_id_img=model_img,
            status_id_img=status_img,
            tpeak_img=tpeak_img,
            summary=summary,
            output_paths=outputs,
            config=cfg,
            times_s=t_s.astype(np.float64),
        )

    # Regions mode?
    reg_cfg = cfg["regions"]
    if reg_cfg.get("enabled", False):
        # Region ROI aggregate mode runs without chunking (regions usually smaller count);
        # could be chunked later but not critical.
        t0 = _time.perf_counter()
        label_img = load_mask(reg_cfg["label_map_path"])
        labels = np.asanyarray(label_img.dataobj).reshape(-1)[idx].astype(np.int32)
        classes = reg_cfg.get("classes", {})
        if not classes:
            raise ValueError("regions.enabled is true but regions.classes is empty.")

        # Build full A and valid once (region operations need grouped data)
        A = A4.reshape((-1, T))[idx, :]  # (N_vox, T)
        nf_cfg = cfg["noise_floor"]
        if nf_cfg["enabled"]:
            floor = compute_noise_floor(
                A,
                mode=nf_cfg["mode"],
                absolute=float(nf_cfg["absolute_bq_per_ml"] * vml),
                rel_frac=float(nf_cfg["relative_fraction_of_voxel_max"]),
            )
            valid = valid_mask_from_floor(A, floor)
        else:
            valid = np.isfinite(A)

        n_valid = np.sum(valid, axis=1)
        all_below = n_valid == 0
        insufficient = n_valid < 2
        status_id[all_below] = STATUS_ALL_BELOW_FLOOR
        status_id[insufficient] = STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        # optional voxel-level R2 in region mode
        voxel_level_r2 = bool(reg_cfg.get("voxel_level_r2", False))

        for lab in unique_labels:
            vox_mask = labels == lab
            if not np.any(vox_mask):
                continue
            lab_key = str(int(lab))
            if lab_key not in classes:
                status_id[vox_mask] = STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS
                continue

            region_def = classes[lab_key]
            region_class = region_def.get("class", "hybrid").lower()
            default_model = region_def.get("default_model", None)
            allowed = [m.lower() for m in region_def.get("allowed_models", [])]

            # Mean TAC in Bq
            Areg = np.nanmean(np.where(valid[vox_mask], A[vox_mask], np.nan), axis=0).astype(np.float32)
            valid_reg = np.isfinite(Areg) & (Areg > 0)
            if np.sum(valid_reg) < 2:
                status_id[vox_mask] = STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS
                continue

            # Fit region
            if region_class == "hump" and (default_model in (None, "gamma")) and ("gamma" in allowed or not allowed):
                if np.any(t_s <= 0):
                    # cannot do gamma linear if t includes 0
                    tail_mode = cfg["integration"]["tail_mode"]
                    if lambda_phys is None and tail_mode not in ["fitted", "hybrid"]:
                        status_id[vox_mask] = STATUS_FIT_FAILED
                        continue
                    tia_reg_arr, Ahat_reg2, r2_reg, _ = tia_trapz_plus_phys_tail(
                        Areg[None, :], t_s, valid_reg[None, :], lambda_phys=lambda_phys,
                        include_t0=cfg["integration"]["include_t0"],
                        tail_mode=tail_mode,
                        min_tail_points=cfg["integration"]["min_tail_points"],
                        fit_tail_slope=cfg["integration"]["fit_tail_slope"],
                        lambda_phys_constraint=cfg["physics"]["enforce_lambda_ge_phys"]
                    )
                    tia_reg = float(tia_reg_arr[0])
                    region_model_id = 11
                    region_tpeak = np.nan
                    region_r2 = float(r2_reg[0])
                    Ahat_reg = Ahat_reg2.astype(np.float32)
                else:
                    params, tpk, Ahat_reg, r2_reg = fit_gamma_linear_wls(
                        Areg[None, :], t_s, valid_reg[None, :], lambda_phys=lambda_phys
                    )
                    tia_reg = float(tia_from_gamma_params(params)[0])
                    region_model_id = 30
                    region_tpeak = float(tpk[0])
                    region_r2 = float(r2_reg[0])
            elif region_class == "falling" and (default_model in (None, "exp")) and ("exp" in allowed or not allowed):
                peak_index = np.array([int(np.argmax(valid_reg))], dtype=np.int64)
                lam, Ahat_reg, r2_reg = fit_monoexp_tail(
                    Areg[None, :], t_s, valid_reg[None, :], lambda_phys=lambda_phys, peak_index=peak_index
                )
                tia_reg = float(
                    tia_monoexp_with_triangle_uptake(Areg[None, :], t_s, valid_reg[None, :], lam, peak_index)[0]
                )
                region_model_id = 20
                region_tpeak = float(t_s[peak_index[0]])
                region_r2 = float(r2_reg[0])
            else:
                tail_mode = cfg["integration"]["tail_mode"]
                if lambda_phys is None and tail_mode not in ["fitted", "hybrid"]:
                    status_id[vox_mask] = STATUS_FIT_FAILED
                    continue
                tia_reg_arr, Ahat_reg, r2_reg, _ = tia_trapz_plus_phys_tail(
                    Areg[None, :], t_s, valid_reg[None, :], lambda_phys=lambda_phys,
                    include_t0=cfg["integration"]["include_t0"],
                    tail_mode=tail_mode,
                    min_tail_points=cfg["integration"]["min_tail_points"],
                    fit_tail_slope=cfg["integration"]["fit_tail_slope"],
                    lambda_phys_constraint=cfg["physics"]["enforce_lambda_ge_phys"]
                )
                tia_reg = float(tia_reg_arr[0])
                region_model_id = 10 if region_class == "rising" else 11
                region_tpeak = np.nan
                region_r2 = float(r2_reg[0])

            if not np.isfinite(tia_reg):
                status_id[vox_mask] = STATUS_FIT_FAILED
                continue

            # reference time selection for scaling
            ref_choice = reg_cfg.get("scaling", {}).get("reference_time", "peak")
            if isinstance(ref_choice, str) and ref_choice.startswith("index:"):
                jref = int(ref_choice.split(":", 1)[1])
            elif ref_choice == "last":
                jref = int(np.nanmax(np.where(valid_reg)[0]))
            else:
                jref = int(np.nanargmax(Ahat_reg[0]))

            Aref = float(Ahat_reg[0, jref])
            if not np.isfinite(Aref) or Aref <= 0:
                status_id[vox_mask] = STATUS_FIT_FAILED
                continue

            Avox_ref = A[vox_mask, jref].astype(np.float32)
            vref_ok = valid[vox_mask, jref] & np.isfinite(Avox_ref)
            scale = np.full((np.sum(vox_mask),), np.nan, dtype=np.float32)
            scale[vref_ok] = Avox_ref[vref_ok] / Aref

            # Apply scaling: voxel TIA
            tia[vox_mask] = scale * float(tia_reg)
            model_id[vox_mask] = np.uint8(region_model_id)
            tpeak[vox_mask] = float(region_tpeak) if np.isfinite(region_tpeak) else np.nan

            # R2:
            if voxel_level_r2:
                # voxel predicted curves: scale_v * Ahat_reg
                Ahat_vox = (scale[:, None] * Ahat_reg.astype(np.float32)).astype(np.float32)
                r2_vox = r2_score(A[vox_mask], Ahat_vox, valid[vox_mask])
                r2[vox_mask] = r2_vox
            else:
                r2[vox_mask] = float(region_r2)

            # voxels without valid tref become not applicable
            bad = ~vref_ok
            status_id[vox_mask] = np.where(bad, STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS, STATUS_OK).astype(
                np.uint8
            )

            # Bootstrap in region mode: refit region per replicate, fixed class; scale sigma by voxel scale
            if cfg["bootstrap"]["enabled"]:
                rng = np.random.default_rng(int(cfg["bootstrap"]["seed"]) + int(lab) * 17)
                B = int(cfg["bootstrap"]["n"])

                # residual bootstrap needs baseline Ahat_reg for region
                Ahat_reg0 = Ahat_reg.astype(np.float32)
                tia_bs = np.full((B,), np.nan, dtype=np.float32)

                for b in range(B):
                    Astar = residual_bootstrap(
                        Areg[None, :], Ahat_reg0, valid_reg[None, :], rng=rng
                    )[0]
                    valid_star = np.isfinite(Astar) & (Astar > 0)
                    if np.sum(valid_star) < 2:
                        continue

                    try:
                        if region_model_id == 30:
                            params_s, _, _, _ = fit_gamma_linear_wls(
                                Astar[None, :], t_s, valid_star[None, :], lambda_phys=lambda_phys
                            )
                            tia_bs[b] = float(tia_from_gamma_params(params_s)[0])
                        elif region_model_id == 20:
                            peak_index = np.array([int(np.argmax(valid_star))], dtype=np.int64)
                            lam_s, _, _ = fit_monoexp_tail(
                                Astar[None, :], t_s, valid_star[None, :], lambda_phys=lambda_phys, peak_index=peak_index
                            )
                            tia_bs[b] = float(
                                tia_monoexp_with_triangle_uptake(
                                    Astar[None, :], t_s, valid_star[None, :], lam_s, peak_index
                                )[0]
                            )
                        else:
                            tail_mode = cfg["integration"]["tail_mode"]
                            if lambda_phys is None and tail_mode not in ["fitted", "hybrid"]:
                                continue
                            tia_s, _, _, _ = tia_trapz_plus_phys_tail(
                                Astar[None, :], t_s, valid_star[None, :], lambda_phys=lambda_phys,
                                include_t0=cfg["integration"]["include_t0"],
                                tail_mode=tail_mode,
                                min_tail_points=cfg["integration"]["min_tail_points"],
                                fit_tail_slope=cfg["integration"]["fit_tail_slope"],
                                lambda_phys_constraint=cfg["physics"]["enforce_lambda_ge_phys"]
                            )
                            tia_bs[b] = float(tia_s[0])
                    except Exception:
                        continue

                if np.any(np.isfinite(tia_bs)):
                    reg_sigma = float(np.nanstd(tia_bs, ddof=1))
                    sig = scale * reg_sigma
                    sigma_tia[vox_mask] = sig.astype(np.float32)

        timing_ms["regions_ms"] = 1000.0 * (_time.perf_counter() - t0)

    else:
        # Voxel mode with chunking
        t0 = _time.perf_counter()
        nf_cfg = cfg["noise_floor"]
        slices = _chunk_slices(n_vox, chunk_size)

        # We need A4 flattened once; but we slice idx chunk-wise to avoid allocating A for all voxels at once.
        Aflat = A4.reshape((-1, T))

        # For bootstrap, we will store baseline Ahat0 and valid0 across all voxels; in chunking mode,
        # we compute Ahat0 per chunk and store (N_vox,T) only if bootstrap enabled.
        do_boot = bool(cfg["bootstrap"]["enabled"])
        Ahat0_all = np.full((n_vox, T), np.nan, dtype=np.float32) if do_boot else None
        valid_all = np.zeros((n_vox, T), dtype=bool) if do_boot else None
        cls_all = np.zeros((n_vox,), dtype=np.uint8) if do_boot else None

        for sl in slices:
            vox_idx = idx[sl]
            A = Aflat[vox_idx, :].astype(np.float32)  # (Nc,T)

            # noise floor validity
            if nf_cfg["enabled"]:
                floor = compute_noise_floor(
                    A,
                    mode=nf_cfg["mode"],
                    absolute=float(nf_cfg["absolute_bq_per_ml"] * vml),
                    rel_frac=float(nf_cfg["relative_fraction_of_voxel_max"]),
                )
                valid = valid_mask_from_floor(A, floor)
            else:
                valid = np.isfinite(A)

            n_valid = np.sum(valid, axis=1)
            all_below = n_valid == 0
            insufficient = n_valid < 2

            status = np.full((A.shape[0],), STATUS_OK, dtype=np.uint8)
            status[all_below] = STATUS_ALL_BELOW_FLOOR
            status[insufficient] = STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS

            # classify
            cls = classify_curves(A, valid)

            # peak index
            A_for_peak = np.where(valid, A, -np.inf)
            peak_index = np.argmax(A_for_peak, axis=1).astype(np.int64)

            # Fit subsets
            tia_c = np.full((A.shape[0],), np.nan, dtype=np.float32)
            r2_c = np.full((A.shape[0],), np.nan, dtype=np.float32)
            model_c = np.zeros((A.shape[0],), dtype=np.uint8)
            tpeak_c = np.full((A.shape[0],), np.nan, dtype=np.float32)

            hump = (cls == CLASS_HUMP) & (n_valid >= int(cfg["model_selection"]["min_points_for_gamma"])) & (~insufficient) & (~all_below)
            if np.any(hump):
                # gamma linear requires t>0
                if np.any(t_s <= 0):
                    # fallback to hybrid
                    hump2 = hump.copy()
                    hump[:] = False
                else:
                    # Try 3-phase model for hump curves with sufficient points (>= 6)
                    hump_3phase = hump & (n_valid >= 6)
                    hump_biexp = hump & (n_valid >= 4) & (~hump_3phase)
                    hump_gamma = hump & (~hump_biexp) & (~hump_3phase)
                    
                    if np.any(hump_3phase):
                        params, tpk, Ahat, r2_3p = fit_three_phase_exp(
                            A[hump_3phase, :], t_s, valid[hump_3phase, :], lambda_phys=lambda_phys, peak_index=peak_index[hump_3phase]
                        )
                        tia_3p = tia_from_three_phase_params(params)
                        tia_c[hump_3phase] = tia_3p
                        r2_c[hump_3phase] = r2_3p
                        tpeak_c[hump_3phase] = tpk
                        model_c[hump_3phase] = 60
                        if do_boot:
                            Ahat0_all[sl][hump_3phase] = Ahat
                    
                    if np.any(hump_biexp):
                        params, tpk, Ahat, r2_be = fit_biexp(
                            A[hump_biexp, :], t_s, valid[hump_biexp, :], lambda_phys=lambda_phys, peak_index=peak_index[hump_biexp]
                        )
                        tia_be = tia_from_biexp_params(params)
                        tia_c[hump_biexp] = tia_be
                        r2_c[hump_biexp] = r2_be
                        tpeak_c[hump_biexp] = tpk
                        model_c[hump_biexp] = 40
                        if do_boot:
                            Ahat0_all[sl][hump_biexp] = Ahat
                    
                    if np.any(hump_gamma):
                        params, tpk, Ahat, r2_h = fit_gamma_linear_wls(A[hump_gamma, :], t_s, valid[hump_gamma, :], lambda_phys=lambda_phys)
                        tia_h = tia_from_gamma_params(params)
                        tia_c[hump_gamma] = tia_h
                        r2_c[hump_gamma] = r2_h
                        tpeak_c[hump_gamma] = tpk
                        model_c[hump_gamma] = 30
                        if do_boot:
                            Ahat0_all[sl][hump_gamma] = Ahat

            falling = (cls == CLASS_FALLING) & (~insufficient) & (~all_below)
            if np.any(falling):
                # Try uptake-washout models for falling curves with sufficient points (>= 4)
                falling_uw = falling & (n_valid >= 4)
                falling_mono = falling & (~falling_uw)
                
                if np.any(falling_uw):
                    # Try linear uptake + mono-exponential washout
                    params, tpk, Ahat, r2_uw = fit_linear_uptake_monoexp_washout_matrix(
                        A[falling_uw, :], t_s, valid[falling_uw, :], lambda_phys=lambda_phys, peak_index=peak_index[falling_uw]
                    )
                    tia_uw = tia_linear_uptake_monoexp_washout(params, tpk)
                    tia_c[falling_uw] = tia_uw
                    r2_c[falling_uw] = r2_uw
                    tpeak_c[falling_uw] = tpk
                    model_c[falling_uw] = 50
                    if do_boot:
                        Ahat0_all[sl][falling_uw] = Ahat
                
                if np.any(falling_mono):
                    lam, Ahat, r2_f = fit_monoexp_tail(
                        A[falling_mono, :], t_s, valid[falling_mono, :], lambda_phys=lambda_phys, peak_index=peak_index[falling_mono]
                    )
                    tia_f = tia_monoexp_with_triangle_uptake(A[falling_mono, :], t_s, valid[falling_mono, :], lam, peak_index[falling_mono])
                    tia_c[falling_mono] = tia_f
                    r2_c[falling_mono] = r2_f
                    tpeak_c[falling_mono] = np.take_along_axis(
                        t_s[None, :], peak_index[falling_mono][:, None], axis=1
                    )[:, 0].astype(np.float32)
                    model_c[falling_mono] = 20
                    if do_boot:
                        Ahat0_all[sl][falling_mono] = Ahat

            # Hybrid for rising + ambiguous + gamma fallback (if times<=0)
            hybrid_mask = ((cls == CLASS_RISING) | (cls == 4)) & (~insufficient) & (~all_below)
            if "hump2" in locals() and np.any(hump2):
                hybrid_mask = hybrid_mask | hump2

            if np.any(hybrid_mask):
                tail_mode = cfg["integration"]["tail_mode"]
                if lambda_phys is None and tail_mode not in ["fitted", "hybrid"]:
                    status[hybrid_mask] = STATUS_FIT_FAILED
                else:
                    tia_hy, _, _, _ = tia_trapz_plus_phys_tail(
                        A[hybrid_mask], t_s, valid[hybrid_mask], lambda_phys=lambda_phys,
                        include_t0=cfg["integration"]["include_t0"],
                        tail_mode=tail_mode,
                        min_tail_points=cfg["integration"]["min_tail_points"],
                        fit_tail_slope=cfg["integration"]["fit_tail_slope"],
                        lambda_phys_constraint=cfg["physics"]["enforce_lambda_ge_phys"]
                    )
                    # Improved R2: compute Ahat piecewise at samples
                    Ahat_hy = hybrid_piecewise_hat_at_samples(A[hybrid_mask], valid[hybrid_mask], t_s)
                    r2_hy = r2_score(A[hybrid_mask], Ahat_hy, valid[hybrid_mask])

                    tia_c[hybrid_mask] = tia_hy
                    r2_c[hybrid_mask] = r2_hy
                    model_c[hybrid_mask] = np.where(cls[hybrid_mask] == CLASS_RISING, 10, 11).astype(np.uint8)
                    if do_boot:
                        Ahat0_all[sl][hybrid_mask] = Ahat_hy

            # mark remaining NaN as fit failed
            bad_fit = (~insufficient) & (~all_below) & (~np.isfinite(tia_c))
            status[bad_fit] = STATUS_FIT_FAILED

            # write chunk into global buffers
            tia[sl] = tia_c
            r2[sl] = r2_c
            model_id[sl] = model_c
            status_id[sl] = status
            tpeak[sl] = tpeak_c

            if do_boot:
                valid_all[sl] = valid
                cls_all[sl] = cls

        timing_ms["voxel_fit_ms"] = 1000.0 * (_time.perf_counter() - t0)

        # Bootstrap in voxel mode (global, uses stored Ahat0/valid/cls)
        if do_boot:
            t0 = _time.perf_counter()
            B = int(cfg["bootstrap"]["n"])
            rng = np.random.default_rng(int(cfg["bootstrap"]["seed"]))
            reclass = bool(cfg["bootstrap"]["reclassify_each_replicate"])
            # compute A once for bootstrap (inevitable for resampling); if too big, can chunk-bootstrap later.
            A_all = Aflat[idx, :].astype(np.float32)

            tias = np.full((B, n_vox), np.nan, dtype=np.float32)
            for b in range(B):
                Astar = residual_bootstrap(A_all, Ahat0_all, valid_all, rng=rng).astype(np.float32)

                if nf_cfg["enabled"]:
                    floor_b = compute_noise_floor(
                        Astar,
                        mode=nf_cfg["mode"],
                        absolute=float(nf_cfg["absolute_bq_per_ml"] * vml),
                        rel_frac=float(nf_cfg["relative_fraction_of_voxel_max"]),
                    )
                    valid_b = valid_mask_from_floor(Astar, floor_b)
                else:
                    valid_b = np.isfinite(Astar)

                n_valid_b = np.sum(valid_b, axis=1)
                ok_b = n_valid_b >= 2

                cls_b = classify_curves(Astar, valid_b) if reclass else cls_all

                tia_b = np.full((n_vox,), np.nan, dtype=np.float32)

                hump_b = (cls_b == CLASS_HUMP) & ok_b & (n_valid_b >= int(cfg["model_selection"]["min_points_for_gamma"]))
                if np.any(hump_b) and not np.any(t_s <= 0):
                    try:
                        # Try 3-phase for curves with >= 6 points
                        hump_3p_b = hump_b & (n_valid_b >= 6)
                        hump_be_b = hump_b & (n_valid_b >= 4) & (~hump_3p_b)
                        hump_g_b = hump_b & (~hump_be_b) & (~hump_3p_b)
                        
                        if np.any(hump_3p_b):
                            params_3p_b, _, _, _ = fit_three_phase_exp(
                                Astar[hump_3p_b], t_s, valid_b[hump_3p_b], lambda_phys=lambda_phys, peak_index=peak_b[hump_3p_b]
                            )
                            tia_b[hump_3p_b] = tia_from_three_phase_params(params_3p_b)
                        
                        if np.any(hump_be_b):
                            params_b, _, _, _ = fit_biexp(
                                Astar[hump_be_b], t_s, valid_b[hump_be_b], lambda_phys=lambda_phys, peak_index=peak_b[hump_be_b]
                            )
                            tia_b[hump_be_b] = tia_from_biexp_params(params_b)
                        
                        if np.any(hump_g_b):
                            params_b, _, _, _ = fit_gamma_linear_wls(
                                Astar[hump_g_b], t_s, valid_b[hump_g_b], lambda_phys=lambda_phys
                            )
                            tia_b[hump_g_b] = tia_from_gamma_params(params_b)
                    except Exception:
                        pass

                falling_b = (cls_b == CLASS_FALLING) & ok_b
                if np.any(falling_b):
                    A_for_peak_b = np.where(valid_b, Astar, -np.inf)
                    peak_b = np.argmax(A_for_peak_b, axis=1).astype(np.int64)
                    # Try uptake-washout for curves with >= 4 points
                    falling_uw_b = falling_b & (n_valid_b >= 4)
                    falling_mono_b = falling_b & (~falling_uw_b)
                    
                    if np.any(falling_uw_b):
                        params_uw_b, tpk_uw_b, _, _ = fit_linear_uptake_monoexp_washout_matrix(
                            Astar[falling_uw_b], t_s, valid_b[falling_uw_b], lambda_phys=lambda_phys, peak_index=peak_b[falling_uw_b]
                        )
                        tia_b[falling_uw_b] = tia_linear_uptake_monoexp_washout(params_uw_b, tpk_uw_b)
                    
                    if np.any(falling_mono_b):
                        lam_b, _, _ = fit_monoexp_tail(
                            Astar[falling_mono_b], t_s, valid_b[falling_mono_b], lambda_phys=lambda_phys, peak_index=peak_b[falling_mono_b]
                        )
                        tia_b[falling_mono_b] = tia_monoexp_with_triangle_uptake(
                            Astar[falling_mono_b], t_s, valid_b[falling_mono_b], lam_b, peak_b[falling_mono_b]
                        )

                hybrid_b = ((cls_b == CLASS_RISING) | (cls_b == 4)) & ok_b
                tail_mode = cfg["integration"]["tail_mode"]
                if np.any(hybrid_b) and (lambda_phys is not None or tail_mode in ["fitted", "hybrid"]):
                    tia_hyb, _, _, _ = tia_trapz_plus_phys_tail(
                        Astar[hybrid_b], t_s, valid_b[hybrid_b], lambda_phys=lambda_phys,
                        include_t0=cfg["integration"]["include_t0"],
                        tail_mode=tail_mode,
                        min_tail_points=cfg["integration"]["min_tail_points"],
                        fit_tail_slope=cfg["integration"]["fit_tail_slope"],
                        lambda_phys_constraint=cfg["physics"]["enforce_lambda_ge_phys"]
                    )
                    tia_b[hybrid_b] = tia_hyb

                tias[b] = tia_b

            sigma_tia[:] = np.nanstd(tias, axis=0, ddof=1).astype(np.float32)
            timing_ms["bootstrap_ms"] = 1000.0 * (_time.perf_counter() - t0)

    # Unflatten to volumes
    t0 = _time.perf_counter()
    shape3 = ref.shape[:3]
    tia_vol = np.full((np.prod(shape3),), np.nan, dtype=np.float32)
    r2_vol = np.full_like(tia_vol, np.nan)
    sig_vol = np.full_like(tia_vol, np.nan)
    model_vol = np.zeros_like(tia_vol, dtype=np.uint8)
    status_vol = np.zeros_like(tia_vol, dtype=np.uint8)
    tpeak_vol = np.full_like(tia_vol, np.nan)

    status_vol[:] = STATUS_OUTSIDE
    tia_vol[idx] = tia
    r2_vol[idx] = r2
    sig_vol[idx] = sigma_tia
    model_vol[idx] = model_id
    status_vol[idx] = status_id
    tpeak_vol[idx] = tpeak

    tia_img = make_like(ref, tia_vol.reshape(shape3))
    r2_img = make_like(ref, r2_vol.reshape(shape3))
    sig_img = make_like(ref, sig_vol.reshape(shape3))
    model_img = make_like(ref, model_vol.reshape(shape3).astype(np.uint8))
    status_img = make_like(ref, status_vol.reshape(shape3).astype(np.uint8))
    tpeak_img = make_like(ref, tpeak_vol.reshape(shape3)) if np.any(np.isfinite(tpeak_vol)) else None
    timing_ms["assemble_ms"] = 1000.0 * (_time.perf_counter() - t0)

    # Save outputs
    t0 = _time.perf_counter()
    outputs: dict[str, Path] = {}
    import nibabel as nib  # local import

    fn_tia = _out_name(prefix, "tia.nii.gz")
    fn_r2 = _out_name(prefix, "r2.nii.gz")
    fn_sig = _out_name(prefix, "sigma_tia.nii.gz")
    fn_model = _out_name(prefix, "model_id.nii.gz")
    fn_status = _out_name(prefix, "status_id.nii.gz")
    fn_tpeak = _out_name(prefix, "tpeak.nii.gz")

    nib.save(tia_img, str(out_dir / fn_tia)); outputs["tia"] = out_dir / fn_tia
    nib.save(r2_img, str(out_dir / fn_r2)); outputs["r2"] = out_dir / fn_r2
    nib.save(sig_img, str(out_dir / fn_sig)); outputs["sigma_tia"] = out_dir / fn_sig
    nib.save(model_img, str(out_dir / fn_model)); outputs["model_id"] = out_dir / fn_model
    nib.save(status_img, str(out_dir / fn_status)); outputs["status_id"] = out_dir / fn_status
    if tpeak_img is not None:
        nib.save(tpeak_img, str(out_dir / fn_tpeak)); outputs["tpeak"] = out_dir / fn_tpeak
    timing_ms["save_ms"] = 1000.0 * (_time.perf_counter() - t0)

    # Summary
    status_counts = {int(k): int(v) for k, v in zip(*np.unique(status_vol[idx], return_counts=True))}
    summary = {
        "pytia_version": "0.1.0",
        "times_seconds": [float(x) for x in t_s.tolist()],
        "voxel_volume_ml": float(vml),
        "status_legend": STATUS_LEGEND,
        "status_counts": {STATUS_LEGEND[int(k)]: int(v) for k, v in status_counts.items()},
        "timing_ms": timing_ms if (enable_prof or True) else {},  # keep timings; toggle later if desired
        "config": cfg,
    }
    if io_cfg.get("write_summary_yaml", True):
        outputs["summary"] = _save_summary(out_dir, prefix, summary)

    timing_ms["total_ms"] = 1000.0 * (_time.perf_counter() - t0_all)
    summary["timing_ms"] = timing_ms

    return Results(
        tia_img=tia_img,
        r2_img=r2_img,
        sigma_tia_img=sig_img,
        model_id_img=model_img,
        status_id_img=status_img,
        tpeak_img=tpeak_img,
        summary=summary,
        output_paths=outputs,
        config=cfg,
        times_s=t_s.astype(np.float64),
    )