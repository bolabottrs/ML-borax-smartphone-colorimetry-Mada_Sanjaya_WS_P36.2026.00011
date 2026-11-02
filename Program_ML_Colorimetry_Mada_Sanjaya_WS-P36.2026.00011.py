# -*- coding: utf-8 -*-
"""
BORAX + PURPLE CABBAGE ANTHOCYANIN
Premium Q1 top-tier figure and table generator

Paper title:
Multimodal Smartphone Colorimetry for Quantitative Borax Determination Using Purple Cabbage Anthocyanin:
Spectroscopic Validation, Chemometric Interpretation, and Explainable Machine Learning

Purpose
-------
This script generates publication-grade main figures, supplementary figures,
and manuscript/supplementary tables from:
1) Smartphone raw photos: ganjil and genap folders.
2) pH values for 0-20 mM borax.
3) Optional UV-Vis spectra: Data Spectro Ganjil and Data Spectro Genap folders.

Recommended folder structure
----------------------------
project_folder/
├─ borax_Q1_PREMIUM_FIGURES_AND_TABLES_FINAL.py
├─ ganjil/                         # photos: 0,1,3,...,19 mM left-to-right
├─ genap/                          # photos: 0,2,4,...,20 mM left-to-right
├─ Data Spectro Ganjil/            # optional UV-Vis files: 0 mm.xls, 1 mm.xls, etc.
└─ Data Spectro Genap/             # optional UV-Vis files: 0 mm.xls, 2 mm.xls, etc.

Run
---
python borax_Q1_PREMIUM_FIGURES_AND_TABLES_FINAL.py

Manual ROI correction
---------------------
After first run, inspect:
Q1_Borax_TOP_TIER_Final_Enhanced_Output/roi_overlays/
If any ROI needs correction, edit:
Q1_Borax_TOP_TIER_Final_Enhanced_Output/tables_supplementary/TableS_ROI_coordinates_and_QC.csv
Save a copy as manual_roi_override.csv beside this script, then rerun.

Dependencies
------------
pip install numpy pandas matplotlib opencv-python scikit-learn scipy openpyxl xlrd joblib
Optional:
pip install xgboost shap umap-learn
"""

from __future__ import annotations

import os
import re
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Ellipse, Polygon
from matplotlib.gridspec import GridSpec

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import t as student_t

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
import joblib

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Q1_Borax_TOP_TIER_Final_Enhanced_Output"
FIG_MAIN = OUTPUT / "figures_main"
FIG_SUPP = OUTPUT / "figures_supplementary"
TAB_MAIN = OUTPUT / "tables_main"
TAB_SUPP = OUTPUT / "tables_supplementary"
PROC_DIR = OUTPUT / "processed_data"
MODEL_DIR = OUTPUT / "models"
ROI_OVERLAY_DIR = OUTPUT / "roi_overlays"
ROI_CROP_DIR = OUTPUT / "roi_crops"
for d in [OUTPUT, FIG_MAIN, FIG_SUPP, TAB_MAIN, TAB_SUPP, PROC_DIR, MODEL_DIR, ROI_OVERLAY_DIR, ROI_CROP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PHOTO_GANJIL_CANDIDATES = ["ganjil", "Ganjil", "Data Foto Ganjil", "Foto Ganjil", "Data Raw Ganjil"]
PHOTO_GENAP_CANDIDATES = ["genap", "Genap", "Data Foto Genap", "Foto Genap", "Data Raw Genap"]
UVVIS_GANJIL_CANDIDATES = ["Data Spectro Ganjil", "Spectro Ganjil", "UVVis Ganjil"]
UVVIS_GENAP_CANDIDATES = ["Data Spectro Genap", "Spectro Genap", "UVVis Genap"]

GANJIL_CONC = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
GENAP_CONC  = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def ensure_pH_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a clean pH column exists using PH_MAP; tolerant to pH_x/pH_y/ph variants."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "pH" not in out.columns:
        candidates = [c for c in out.columns if c.lower() in {"ph", "p_h"} or c.lower().startswith("ph_") or c.lower().startswith("ph.")]
        if candidates:
            out["pH"] = pd.to_numeric(out[candidates[0]], errors="coerce")
        else:
            out["pH"] = np.nan
    out["pH"] = pd.to_numeric(out["pH"], errors="coerce")
    if "concentration_mM" in out.columns:
        conc_int = pd.to_numeric(out["concentration_mM"], errors="coerce").round().astype("Int64")
        mapped = conc_int.map(PH_MAP).astype(float)
        out["pH"] = out["pH"].fillna(mapped)
    return out

ALL_CONC = list(range(0, 21))
SELECTED_CONC = [0, 5, 10, 15, 20]

PH_MAP = {
    0: 6.24, 1: 6.93, 2: 7.35, 3: 7.45, 4: 7.55,
    5: 7.72, 6: 7.76, 7: 7.83, 8: 7.93, 9: 8.01,
    10: 8.06, 11: 8.13, 12: 8.14, 13: 8.17, 14: 8.24,
    15: 8.33, 16: 8.34, 17: 8.38, 18: 8.43, 19: 8.46, 20: 8.43,
}

# ROI parameters: zone-constrained weighted centroid, never free segmentation.
ROI_RADIUS_RATIO_TO_IMAGE_HEIGHT = 0.032
ROI_RADIUS_MIN_PX = 18
ROI_RADIUS_MAX_PX = 95
ROW_SEARCH_TOP_FRAC = 0.20
ROW_SEARCH_BOTTOM_FRAC = 0.78
LEFT_MARGIN_FRAC = 0.055
RIGHT_MARGIN_FRAC = 0.055
ZONE_Y_HALF_FRAC = 0.105
ZONE_INNER_X_MARGIN_FRAC = 0.12
CENTROID_MAX_SHIFT_ZONE_FRAC = 0.18
LOWER_V_PERCENTILE = 7
UPPER_V_PERCENTILE = 93

SAVE_PDF = True
SAVE_TIFF = True
FIG_DPI = 600
MANUAL_ROI_OVERRIDE_CSV = ROOT / "manual_roi_override.csv"

# =============================================================================
# STYLE AND BASIC UTILITIES
# =============================================================================
def set_premium_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": FIG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(path: Path, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    plt.savefig(str(path) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        plt.savefig(str(path) + ".pdf", bbox_inches="tight")
    if SAVE_TIFF:
        plt.savefig(str(path) + ".tiff", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def add_panel_label(ax, label: str, x=-0.08, y=1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def first_existing_folder(candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = ROOT / name
        if p.exists() and p.is_dir():
            return p
    return None


def list_images(folder: Optional[Path]) -> List[Path]:
    if folder is None:
        return []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.PNG"]
    files: List[Path] = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


def natural_concentration_from_filename(path: Path) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mM|mm|MM)?", path.stem)
    return float(m.group(1)) if m else None


def safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b + 1e-9)


def rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = np.clip(np.round(rgb), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])


def metric(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_mM": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_mM": float(mean_absolute_error(y_true, y_pred)),
    }

# =============================================================================
# ROI DETECTION
# =============================================================================
def circular_mask(shape: Tuple[int, int, int], cx: int, cy: int, r: int) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def robust_roi_score_image(img_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 140).astype(np.float32)
    edges = cv2.GaussianBlur(edges, (0, 0), 3)

    def norm(x):
        lo, hi = np.percentile(x, [2, 98])
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    # Emphasize colored liquid and cup edge, penalize white background.
    score = 0.48 * norm(S) + 0.34 * norm(chroma) + 0.10 * norm(edges) + 0.08 * (1 - norm(V))
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 5)
    return score


def estimate_sample_row_y(score: np.ndarray) -> int:
    h, w = score.shape[:2]
    y1, y2 = int(h * ROW_SEARCH_TOP_FRAC), int(h * ROW_SEARCH_BOTTOM_FRAC)
    band = score[y1:y2, :]
    profile = band.mean(axis=1)
    k = max(9, int(h * 0.025) // 2 * 2 + 1)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32), (1, k), 0).ravel()
    return int(y1 + np.argmax(profile))


def zone_weighted_centroid(score: np.ndarray, x_center: float, zone_width: float, row_y: int, r0: int) -> Tuple[int, int, int, float]:
    h, w = score.shape[:2]
    inner_margin = zone_width * ZONE_INNER_X_MARGIN_FRAC
    x1 = int(max(0, x_center - zone_width / 2 + inner_margin))
    x2 = int(min(w, x_center + zone_width / 2 - inner_margin))
    y1 = int(max(0, row_y - h * ZONE_Y_HALF_FRAC))
    y2 = int(min(h, row_y + h * ZONE_Y_HALF_FRAC))
    patch = score[y1:y2, x1:x2].copy()
    if patch.size == 0:
        return int(round(x_center)), int(row_y), r0, 0.0

    # Use percentile threshold and weighted centroid, not local argmax.
    thr = np.percentile(patch, 72)
    W = np.maximum(patch - thr, 0)
    if W.sum() <= 1e-9:
        W = np.maximum(patch, 0)
    yy, xx = np.indices(W.shape)
    if W.sum() <= 1e-9:
        cx = int(round(x_center)); cy = int(row_y); conf = 0.0
    else:
        cx0 = float((xx * W).sum() / W.sum() + x1)
        cy0 = float((yy * W).sum() / W.sum() + y1)
        max_shift = zone_width * CENTROID_MAX_SHIFT_ZONE_FRAC
        cx = int(round(np.clip(cx0, x_center - max_shift, x_center + max_shift)))
        cy = int(round(np.clip(cy0, row_y - h * 0.060, row_y + h * 0.060)))
        conf = float(np.percentile(patch, 95) - np.percentile(patch, 50))

    # Radius refinement around expected liquid center.
    candidates = np.arange(max(ROI_RADIUS_MIN_PX, int(r0 * 0.75)), min(ROI_RADIUS_MAX_PX, int(r0 * 1.25)) + 1, max(2, int(r0 * 0.08)))
    best_r, best_metric = r0, -np.inf
    for rr in candidates:
        mask = circular_mask((h, w, 3), cx, cy, int(rr))
        vals = score[mask]
        if len(vals) < 50:
            continue
        metric_val = float(np.nanmedian(vals) - 0.15 * np.nanstd(vals))
        if metric_val > best_metric:
            best_metric, best_r = metric_val, int(rr)
    return cx, cy, best_r, conf


def detect_rois_for_image(img_rgb: np.ndarray, concentrations: List[int]) -> List[Tuple[int, int, int, float]]:
    h, w = img_rgb.shape[:2]
    n = len(concentrations)
    score = robust_roi_score_image(img_rgb)
    row_y = estimate_sample_row_y(score)
    r0 = int(np.clip(h * ROI_RADIUS_RATIO_TO_IMAGE_HEIGHT, ROI_RADIUS_MIN_PX, ROI_RADIUS_MAX_PX))
    xs = np.linspace(w * LEFT_MARGIN_FRAC, w * (1 - RIGHT_MARGIN_FRAC), n)
    zone_width = (w * (1 - RIGHT_MARGIN_FRAC) - w * LEFT_MARGIN_FRAC) / max(n - 1, 1)
    rois = [zone_weighted_centroid(score, float(x), zone_width, int(row_y), r0) for x in xs]
    # Enforce left-to-right order while preserving association to known concentrations.
    # If rare x inversion occurs, fallback to expected x for unstable point.
    fixed = []
    for i, (roi, x) in enumerate(zip(rois, xs)):
        cx, cy, rr, conf = roi
        if i > 0 and cx <= fixed[-1][0]:
            cx = int(round(x))
        fixed.append((cx, cy, rr, conf))
    return fixed


def load_manual_roi_override(path: Path) -> Dict[Tuple[str, str, int], Tuple[int, int, int]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"dataset", "image_file", "concentration_mM", "roi_cx", "roi_cy", "roi_radius"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"manual_roi_override.csv must contain columns: {sorted(required)}")
    out = {}
    for _, row in df.iterrows():
        key = (str(row["dataset"]), str(row["image_file"]), int(row["concentration_mM"]))
        out[key] = (int(row["roi_cx"]), int(row["roi_cy"]), int(row["roi_radius"]))
    return out


def trim_pixels_by_value(rgb_pixels: np.ndarray) -> np.ndarray:
    if len(rgb_pixels) < 30:
        return rgb_pixels
    hsv = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    v = hsv[:, 2].astype(float)
    lo, hi = np.percentile(v, [LOWER_V_PERCENTILE, UPPER_V_PERCENTILE])
    keep = (v >= lo) & (v <= hi)
    return rgb_pixels[keep] if keep.sum() >= 30 else rgb_pixels


def cv2_lab_to_standard(lab_cv: np.ndarray) -> np.ndarray:
    L = lab_cv[:, 0].astype(float) * 100.0 / 255.0
    a = lab_cv[:, 1].astype(float) - 128.0
    b = lab_cv[:, 2].astype(float) - 128.0
    return np.vstack([L, a, b]).T


def extract_color_features_from_set(dataset_name: str, folder: Optional[Path], concentrations: List[int], manual: Dict) -> pd.DataFrame:
    files = list_images(folder)
    rows: List[Dict] = []
    if not files:
        print(f"[WARNING] No images found for {dataset_name}. Folder: {folder}")
        return pd.DataFrame()

    roi_template_rows = []
    for rep_idx, fp in enumerate(files, start=1):
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        auto_rois = detect_rois_for_image(img_rgb, concentrations)

        final_rois = []
        for i, conc in enumerate(concentrations):
            key = (dataset_name, fp.name, int(conc))
            if key in manual:
                cx, cy, rr = manual[key]
                conf = 1.0
            else:
                cx, cy, rr, conf = auto_rois[i]
            final_rois.append((cx, cy, rr, conf))
            roi_template_rows.append({
                "dataset": dataset_name,
                "image_file": fp.name,
                "replicate": rep_idx,
                "physical_order_left_to_right": i + 1,
                "concentration_mM": conc,
                "roi_cx": int(cx),
                "roi_cy": int(cy),
                "roi_radius": int(rr),
                "roi_confidence": float(conf),
                "note": "Inspect overlay. If correction is needed, edit roi_cx/roi_cy/roi_radius and save as manual_roi_override.csv beside script."
            })

        overlay = img_rgb.copy()
        for conc, (cx, cy, rr, conf) in zip(concentrations, final_rois):
            cv2.circle(overlay, (int(cx), int(cy)), int(rr), (0, 0, 0), max(2, int(h * 0.002)))
            cv2.putText(overlay, f"{conc}", (int(cx) - 15, int(cy) - int(rr) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.42, h / 1850), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(ROI_OVERLAY_DIR / f"{dataset_name}_rep{rep_idx:03d}_{fp.stem}_ROI_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        for order, (conc, (cx, cy, rr, conf)) in enumerate(zip(concentrations, final_rois), start=1):
            mask = circular_mask(img_rgb.shape, int(cx), int(cy), int(rr))
            pix_rgb = img_rgb[mask]
            pix_rgb = trim_pixels_by_value(pix_rgb)
            if len(pix_rgb) < 20:
                continue
            pix_hsv = cv2.cvtColor(pix_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            pix_lab_cv = cv2.cvtColor(pix_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            pix_lab = cv2_lab_to_standard(pix_lab_cv)
            R, G, B = pix_rgb[:, 0].astype(float), pix_rgb[:, 1].astype(float), pix_rgb[:, 2].astype(float)
            H = pix_hsv[:, 0].astype(float) * 2.0
            S, V = pix_hsv[:, 1].astype(float), pix_hsv[:, 2].astype(float)
            Ls, As, Bs = pix_lab[:, 0], pix_lab[:, 1], pix_lab[:, 2]
            denom = R + G + B + 1e-9
            chroma = float(np.sqrt(As.mean() ** 2 + Bs.mean() ** 2))
            hue_lab = math.degrees(math.atan2(Bs.mean(), As.mean()))
            if hue_lab < 0:
                hue_lab += 360.0
            row = {
                "dataset": dataset_name,
                "image_file": fp.name,
                "replicate": rep_idx,
                "physical_order_left_to_right": order,
                "concentration_mM": float(conc),
                "pH": PH_MAP.get(int(conc), np.nan),
                "roi_cx": int(cx), "roi_cy": int(cy), "roi_radius": int(rr), "roi_confidence": float(conf),
                "n_pixels": int(len(pix_rgb)),
                "R_mean": R.mean(), "G_mean": G.mean(), "B_mean": B.mean(),
                "R_std": R.std(ddof=1), "G_std": G.std(ddof=1), "B_std": B.std(ddof=1),
                "H_mean": H.mean(), "S_mean": S.mean(), "V_mean": V.mean(),
                "H_std": H.std(ddof=1), "S_std": S.std(ddof=1), "V_std": V.std(ddof=1),
                "Lab_L_mean": Ls.mean(), "Lab_a_mean": As.mean(), "Lab_b_mean": Bs.mean(),
                "Lab_L_std": Ls.std(ddof=1), "Lab_a_std": As.std(ddof=1), "Lab_b_std": Bs.std(ddof=1),
                "Chroma_ab": chroma, "Hue_angle_ab": hue_lab,
                "R_over_G": safe_ratio(R.mean(), G.mean()), "R_over_B": safe_ratio(R.mean(), B.mean()),
                "G_over_R": safe_ratio(G.mean(), R.mean()), "G_over_B": safe_ratio(G.mean(), B.mean()),
                "B_over_R": safe_ratio(B.mean(), R.mean()), "B_over_G": safe_ratio(B.mean(), G.mean()),
                "Normalized_R": (R / denom).mean(), "Normalized_G": (G / denom).mean(), "Normalized_B": (B / denom).mean(),
                "BlueMinusRed": B.mean() - R.mean(), "GreenMinusRed": G.mean() - R.mean(),
                "Excess_Blue_Index": 2 * B.mean() - R.mean() - G.mean(),
                "Excess_Green_Index": 2 * G.mean() - R.mean() - B.mean(),
                "RGB_hex": rgb_to_hex(np.array([R.mean(), G.mean(), B.mean()])),
            }
            rows.append(row)
            cr = int(rr * 2.8)
            x1, x2 = max(0, int(cx) - cr), min(w, int(cx) + cr)
            y1, y2 = max(0, int(cy) - cr), min(h, int(cy) + cr)
            crop = img_rgb[y1:y2, x1:x2].copy()
            cv2.circle(crop, (int(cx) - x1, int(cy) - y1), int(rr), (0, 0, 0), 2)
            cv2.imwrite(str(ROI_CROP_DIR / f"{dataset_name}_rep{rep_idx:03d}_{int(conc)}mM_{fp.stem}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    # Append to ROI table; main writes combined later.
    return pd.DataFrame(rows), pd.DataFrame(roi_template_rows)

# =============================================================================
# UV-VIS PROCESSING
# =============================================================================
def read_uvvis_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in [".xls", ".xlsx"]:
            raw = pd.read_excel(path, header=None)
        else:
            raw = pd.read_csv(path, header=None, sep=None, engine="python")
    except Exception:
        return None
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.shape[1] < 2:
        return None
    best = None
    for c1 in range(raw.shape[1] - 1):
        for c2 in range(c1 + 1, min(raw.shape[1], c1 + 5)):
            tmp = raw[[c1, c2]].copy()
            tmp.columns = ["wavelength_nm", "absorbance"]
            tmp["wavelength_nm"] = pd.to_numeric(tmp["wavelength_nm"], errors="coerce")
            tmp["absorbance"] = pd.to_numeric(tmp["absorbance"], errors="coerce")
            tmp = tmp.dropna()
            tmp = tmp[(tmp["wavelength_nm"] >= 250) & (tmp["wavelength_nm"] <= 900)]
            if len(tmp) > 30:
                if best is None or len(tmp) > len(best):
                    best = tmp
    if best is None:
        return None
    return best.sort_values("wavelength_nm").reset_index(drop=True)


def load_uvvis_dataset(folder: Optional[Path], dataset_name: str) -> pd.DataFrame:
    if folder is None:
        return pd.DataFrame()
    files = sorted(list(folder.glob("*.xls")) + list(folder.glob("*.xlsx")) + list(folder.glob("*.csv")) + list(folder.glob("*.txt")))
    rows = []
    for fp in files:
        conc = natural_concentration_from_filename(fp)
        if conc is None:
            continue
        df = read_uvvis_file(fp)
        if df is None:
            print(f"[WARNING] Could not read UV-Vis file: {fp.name}")
            continue
        df["concentration_mM"] = float(conc)
        df["dataset"] = dataset_name
        df["source_file"] = fp.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def process_uvvis(uv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if uv.empty:
        return pd.DataFrame(), pd.DataFrame()
    wl_min = int(math.ceil(uv["wavelength_nm"].min()))
    wl_max = int(math.floor(uv["wavelength_nm"].max()))
    grid = np.arange(wl_min, wl_max + 1, 1)
    spectra = []
    feats = []
    for (dataset, conc, sf), g in uv.groupby(["dataset", "concentration_mM", "source_file"]):
        x = g["wavelength_nm"].values.astype(float)
        y = g["absorbance"].values.astype(float)
        if len(np.unique(x)) < 20:
            continue
        f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
        yi = f(grid)
        valid = pd.Series(yi).interpolate(limit_direction="both").values
        y_smooth = valid.copy()
        win = min(21, len(valid) // 2 * 2 - 1)
        if win >= 7:
            y_smooth = savgol_filter(valid, win, 2)
        spectra.append(pd.DataFrame({
            "dataset": dataset, "concentration_mM": conc, "source_file": sf,
            "wavelength_nm": grid, "absorbance": y_smooth,
        }))
        # robust spectral features: avoid unstable denominator ratios.
        idx = int(np.nanargmax(y_smooth))
        feats.append({
            "dataset": dataset,
            "concentration_mM": conc,
            "source_file": sf,
            "lambda_max_nm": float(grid[idx]),
            "Amax": float(y_smooth[idx]),
            "AUC": float(np.trapezoid(np.nan_to_num(y_smooth), grid)),
            "A420": float(np.interp(420, grid, y_smooth)),
            "A520": float(np.interp(520, grid, y_smooth)),
            "A560": float(np.interp(560, grid, y_smooth)),
            "A600": float(np.interp(600, grid, y_smooth)),
            "A620": float(np.interp(620, grid, y_smooth)),
            "DeltaA520_from_blank_placeholder": np.nan,
        })
    sp = pd.concat(spectra, ignore_index=True) if spectra else pd.DataFrame()
    feat = pd.DataFrame(feats)
    if not feat.empty:
        blank = feat[feat["concentration_mM"] == 0].mean(numeric_only=True)
        for col in ["Amax", "A420", "A520", "A560", "A600", "A620", "AUC", "lambda_max_nm"]:
            if col in feat.columns and col in blank.index:
                feat[f"Delta_{col}_from_blank"] = feat[col] - float(blank[col])
    return sp, feat

# =============================================================================
# TABLE HELPERS
# =============================================================================
def format_table_png(df: pd.DataFrame, title: str, path: Path, max_rows: int = 25, font_size: int = 8, col_width: float = 1.25) -> None:
    d = df.copy().head(max_rows)
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    nrows, ncols = d.shape
    fig_w = max(7.0, min(16.0, ncols * col_width))
    fig_h = max(2.2, min(18.0, 0.35 * nrows + 1.3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    tbl = ax.table(cellText=d.values, colLabels=d.columns, loc="center", cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold")
        else:
            if r % 2 == 0:
                cell.set_facecolor("#FAFAFA")
    savefig(path, tight=True)


def aggregate_color(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby("concentration_mM")[num_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def create_main_and_supplementary_tables(color_df: pd.DataFrame, roi_qc: pd.DataFrame, uv_features: pd.DataFrame,
                                         analytical: pd.DataFrame, perf_ext: pd.DataFrame, perf_int: pd.DataFrame,
                                         pca_loadings: pd.DataFrame, uv_corr: pd.DataFrame) -> None:
    # Main Table 1: pH standards.
    tbl1 = pd.DataFrame({"Borax concentration (mM)": list(PH_MAP.keys()), "pH": list(PH_MAP.values())})
    tbl1.to_csv(TAB_MAIN / "Table1_borax_standards_and_pH.csv", index=False)
    format_table_png(tbl1, "Table 1. Borax standards and measured pH", TAB_MAIN / "Table1_borax_standards_and_pH", max_rows=25)

    # Main Table 2: analytical figures of merit.
    if not analytical.empty:
        analytical.to_csv(TAB_MAIN / "Table2_analytical_figures_of_merit.csv", index=False)
        format_table_png(analytical, "Table 2. Analytical figures of merit", TAB_MAIN / "Table2_analytical_figures_of_merit", max_rows=20)

    # Main Table 3: External ML performance.
    if not perf_ext.empty:
        table3 = perf_ext.sort_values(["validation", "RMSE_mM"]).groupby("validation").head(5)
        table3.to_csv(TAB_MAIN / "Table3_external_validation_performance.csv", index=False)
        format_table_png(table3, "Table 3. External validation performance", TAB_MAIN / "Table3_external_validation_performance", max_rows=12)

    # Supplementary tables.
    if not color_df.empty:
        color_df.to_csv(TAB_SUPP / "TableS1_extracted_color_features.csv", index=False)
        aggregate_color(color_df).to_csv(TAB_SUPP / "TableS2_concentration_level_color_summary.csv", index=False)
    if not roi_qc.empty:
        roi_qc.to_csv(TAB_SUPP / "TableS3_ROI_coordinates_and_QC.csv", index=False)
        format_table_png(roi_qc.head(30), "Table S3. ROI coordinates and QC excerpt", TAB_SUPP / "TableS3_ROI_coordinates_and_QC_excerpt", max_rows=30, font_size=6, col_width=0.85)
    if not uv_features.empty:
        uv_features.to_csv(TAB_SUPP / "TableS4_UVVis_spectral_features.csv", index=False)
    if not perf_int.empty:
        perf_int.to_csv(TAB_SUPP / "TableS5_internal_LOIO_model_performance.csv", index=False)
    if not pca_loadings.empty:
        pca_loadings.to_csv(TAB_SUPP / "TableS6_PCA_loadings.csv", index=False)
    if not uv_corr.empty:
        uv_corr.to_csv(TAB_SUPP / "TableS7_color_UVVis_correlations.csv", index=False)

# =============================================================================
# FEATURE COLUMNS AND MODELS
# =============================================================================
def feature_columns(df: pd.DataFrame, include_ph: bool = False) -> List[str]:
    cols = [
        "R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean",
        "Lab_L_mean", "Lab_a_mean", "Lab_b_mean", "Chroma_ab", "Hue_angle_ab",
        "R_over_G", "R_over_B", "G_over_R", "G_over_B", "B_over_R", "B_over_G",
        "Normalized_R", "Normalized_G", "Normalized_B", "BlueMinusRed", "GreenMinusRed",
        "Excess_Blue_Index", "Excess_Green_Index"
    ]
    if include_ph:
        cols.append("pH")
    return [c for c in cols if c in df.columns]


def build_models(n_samples: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Linear": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Huber": Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor())]),
        "PLS": Pipeline([("scaler", StandardScaler()), ("model", PLSRegression(n_components=2))]),
        "SVR-RBF": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1))]),
        "RandomForest": RandomForestRegressor(n_estimators=600, random_state=42, min_samples_leaf=1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, random_state=42, min_samples_leaf=1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=2, subsample=0.9,
                                          colsample_bytree=0.9, random_state=42, objective="reg:squarederror")
    except Exception:
        pass
    return models

# =============================================================================
# FIGURES MAIN
# =============================================================================
def figure_1_workflow() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    ax.axis("off")
    steps = [
        ("Borax\nstandard", "pH increase"),
        ("Purple cabbage\nanthocyanin", "molecular transformation"),
        ("UV–Vis\nspectroscopy", "spectral validation"),
        ("Smartphone\nimaging", "digital color response"),
        ("ROI\nextraction", "zone-centroid QC"),
        ("RGB–HSV–Lab\nfeatures", "color descriptors"),
        ("Chemometrics\n+ XAI", "interpretation"),
        ("Borax\nprediction", "quantitative output"),
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    y = 0.58
    for i, ((title, sub), x) in enumerate(zip(steps, xs)):
        box = FancyBboxPatch((x - 0.052, y - 0.15), 0.104, 0.22, boxstyle="round,pad=0.018",
                             fc="#F8F9FA", ec="#222222", lw=1.0)
        ax.add_patch(box)
        ax.text(x, y + 0.035, title, ha="center", va="center", fontsize=9.2, fontweight="bold")
        ax.text(x, y - 0.075, sub, ha="center", va="center", fontsize=7.8, color="0.25")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.057, y - 0.04), (xs[i+1] - 0.057, y - 0.04),
                                         arrowstyle="->", mutation_scale=12, lw=1.1, color="#333333"))
    ax.text(0.5, 0.16, "Multimodal sensing pipeline for anthocyanin-assisted borax quantification",
            ha="center", va="center", fontsize=13, fontweight="bold")
    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)


def figure_2_roi_and_color_gallery(color_df: pd.DataFrame) -> None:
    """Premium tighter ROI gallery.

    Improvements over the previous version:
    - Five representative concentrations instead of three.
    - Tighter crops so the liquid/ROI dominates, not the plastic cup/background.
    - Larger analytical masks and mean-color swatches for visual inspection.
    """
    if color_df.empty:
        return

    reps = [0, 5, 10, 15, 20]
    selected = []
    for c in reps:
        sub = color_df[color_df["concentration_mM"] == c]
        if not sub.empty:
            # pick the replicate with the highest confidence if available, otherwise first
            if "roi_confidence" in sub.columns:
                selected.append(sub.sort_values("roi_confidence", ascending=False).iloc[0])
            else:
                selected.append(sub.iloc[0])
    if not selected:
        return

    n = len(selected)
    fig, axes = plt.subplots(
        3, n, figsize=(2.55 * n, 6.8),
        gridspec_kw={"height_ratios": [1.18, 1.18, 0.48], "hspace": 0.20, "wspace": 0.10}
    )
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    mean_rgb = _mean_rgb_table(color_df)

    for col, row in enumerate(selected):
        conc = int(row["concentration_mM"])
        img_path = ROOT / str(row["dataset"]) / str(row["image_file"])
        if not img_path.exists():
            for folder in [first_existing_folder(PHOTO_GANJIL_CANDIDATES), first_existing_folder(PHOTO_GENAP_CANDIDATES)]:
                if folder and (folder / str(row["image_file"])).exists():
                    img_path = folder / str(row["image_file"])
                    break

        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cx, cy, rr = int(row["roi_cx"]), int(row["roi_cy"]), int(row["roi_radius"])
            # tighter crop than previous version, but still includes rim context
            cr = int(max(rr * 1.65, rr + 18))
            h, w = img_rgb.shape[:2]
            x1, x2 = max(0, cx - cr), min(w, cx + cr)
            y1, y2 = max(0, cy - cr), min(h, cy + cr)
            crop = img_rgb[y1:y2, x1:x2].copy()
            lc_x, lc_y = cx - x1, cy - y1

            axes[0, col].imshow(crop)
            axes[0, col].add_patch(Circle((lc_x, lc_y), rr, fill=False, lw=1.65, ec="black"))

            mask = circular_mask(crop.shape, lc_x, lc_y, rr)
            vis = np.ones_like(crop) * 255
            vis[mask] = crop[mask]
            axes[1, col].imshow(vis)
        else:
            axes[0, col].text(0.5, 0.5, "Image\nnot found", ha="center", va="center")
            axes[1, col].text(0.5, 0.5, "Mask\nnot found", ha="center", va="center")

        axes[0, col].set_title(f"{conc} mM", fontsize=10.7, pad=6)
        axes[0, col].axis("off")
        axes[1, col].axis("off")

        if conc in mean_rgb.index:
            rgb = np.clip(mean_rgb.loc[conc].values, 0, 255).astype(np.uint8)
            sw = np.ones((70, 135, 3), dtype=np.uint8) * rgb.reshape(1, 1, 3)
            axes[2, col].imshow(sw)
            axes[2, col].text(0.5, -0.24, rgb_to_hex(rgb), transform=axes[2, col].transAxes,
                              ha="center", va="top", fontsize=7.8)
        axes[2, col].axis("off")

    row_labels = ["ROI overlay", "Analytical mask", "Mean color"]
    for r, lab in enumerate(row_labels):
        axes[r, 0].text(-0.13, 0.5, lab, transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=9.4)

    fig.suptitle("Representative ROI validation and concentration-dependent color evolution", y=0.985, fontsize=13.3)
    savefig(FIG_MAIN / "Fig02_representative_ROI_and_color_gallery")

def figure_3_ph_hue_lab(color_df: pd.DataFrame) -> None:
    if color_df.empty:
        return
    agg = color_df.groupby("concentration_mM").agg(H=("H_mean", "mean"), Hsd=("H_mean", "std"),
                                                    Lab_a=("Lab_a_mean", "mean"), Lab_b=("Lab_b_mean", "mean"), pH=("pH", "mean")).reindex(ALL_CONC).dropna()
    x = agg.index.values.astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))
    axes[0].plot(x, agg["pH"], marker="o", lw=2.0)
    axes[0].set_xlabel("Borax concentration (mM)"); axes[0].set_ylabel("pH")
    axes[0].set_title("Borax-induced alkalization"); axes[0].grid(alpha=0.15); add_panel_label(axes[0], "a")
    axes[1].errorbar(x, agg["H"], yerr=agg["Hsd"].fillna(0), fmt="o", capsize=3.5, ms=4.8)
    try:
        popt, _ = curve_fit(lambda C, Hinf, A, k: Hinf + A*np.exp(-k*C), x, agg["H"].values,
                            p0=[agg["H"].min(), agg["H"].max()-agg["H"].min(), 0.12],
                            bounds=([0, 0, 0.0001], [360, 360, 5]), maxfev=20000)
        xs = np.linspace(x.min(), x.max(), 300); ys = popt[0] + popt[1]*np.exp(-popt[2]*xs)
        yh = popt[0] + popt[1]*np.exp(-popt[2]*x)
        axes[1].plot(xs, ys, lw=2.2, label=f"Saturation fit, R²={r2_score(agg['H'], yh):.3f}")
        axes[1].legend(frameon=False)
    except Exception:
        pass
    axes[1].set_xlabel("Borax concentration (mM)"); axes[1].set_ylabel("Hue angle (°)")
    axes[1].set_title("Hue response"); axes[1].grid(alpha=0.15); add_panel_label(axes[1], "b")
    sc = axes[2].scatter(agg["Lab_a"], agg["Lab_b"], c=x, s=55, edgecolor="black", linewidth=0.5)
    axes[2].plot(agg["Lab_a"], agg["Lab_b"], lw=1.0, alpha=0.6)
    for conc, rr in agg.iterrows():
        if int(conc) % 2 == 0:
            axes[2].annotate(str(int(conc)), (rr["Lab_a"], rr["Lab_b"]), xytext=(5, 3), textcoords="offset points", fontsize=7)
    axes[2].set_xlabel("a*"); axes[2].set_ylabel("b*"); axes[2].set_title("CIELab chromatic trajectory"); axes[2].grid(alpha=0.15)
    cb = fig.colorbar(sc, ax=axes[2], pad=0.02); cb.set_label("mM")
    add_panel_label(axes[2], "c")
    fig.suptitle("pH-mediated colorimetric response of purple cabbage anthocyanin", y=1.03)
    savefig(FIG_MAIN / "Fig03_pH_hue_Lab_mechanism")


def figure_4_uvvis_top_tier(uv_spectra: pd.DataFrame) -> None:
    if uv_spectra.empty:
        return
    avg = uv_spectra.groupby(["concentration_mM", "wavelength_nm"])["absorbance"].mean().reset_index()
    concs = sorted(avg["concentration_mM"].unique())
    selected = [c for c in SELECTED_CONC if c in concs]
    if len(selected) < 3:
        selected = concs[::max(1, len(concs)//5)]
    fig = plt.figure(figsize=(14.8, 4.8))
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.0, 1.0], figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.viridis
    for i, conc in enumerate(selected):
        g = avg[avg["concentration_mM"] == conc]
        ax.plot(g["wavelength_nm"], g["absorbance"], lw=2.0, label=f"{conc:g} mM", color=cmap(i/max(1, len(selected)-1)))
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance")
    ax.set_title("Selected UV–Vis spectra"); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.14); add_panel_label(ax, "a")
    ax2 = fig.add_subplot(gs[0, 1])
    pivot = avg.pivot_table(index="concentration_mM", columns="wavelength_nm", values="absorbance")
    im = ax2.imshow(pivot.values, aspect="auto", origin="lower",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()], cmap="magma")
    ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Borax concentration (mM)")
    ax2.set_title("Absorbance heatmap"); cb = fig.colorbar(im, ax=ax2, pad=0.02); cb.set_label("Absorbance"); add_panel_label(ax2, "b")
    ax3 = fig.add_subplot(gs[0, 2])
    blank = avg[avg["concentration_mM"] == 0][["wavelength_nm", "absorbance"]].rename(columns={"absorbance":"blank"})
    for i, conc in enumerate(selected):
        if conc == 0: continue
        g = avg[avg["concentration_mM"] == conc].merge(blank, on="wavelength_nm", how="inner")
        ax3.plot(g["wavelength_nm"], g["absorbance"] - g["blank"], lw=1.8, label=f"{conc:g} mM", color=cmap(i/max(1, len(selected)-1)))
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("Wavelength (nm)"); ax3.set_ylabel("ΔAbsorbance vs blank")
    ax3.set_title("Difference spectra"); ax3.legend(frameon=False, fontsize=8); ax3.grid(alpha=0.14); add_panel_label(ax3, "c")
    fig.suptitle("Spectroscopic validation of borax-induced anthocyanin transformation", y=1.03)
    savefig(FIG_MAIN / "Fig04_UVVis_selected_heatmap_difference")


def figure_5_uvvis_features(uv_features: pd.DataFrame) -> None:
    if uv_features.empty:
        return
    g = uv_features.groupby("concentration_mM").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    for ax, col, title, ylabel in zip(axes, ["lambda_max_nm", "Amax", "AUC"], ["λmax shift", "Peak absorbance", "Integrated absorbance"], ["λmax (nm)", "Amax", "AUC"]):
        if col not in g.columns: continue
        ax.plot(g["concentration_mM"], g[col], marker="o", lw=2.0)
        ax.set_xlabel("Borax concentration (mM)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.15)
    fig.suptitle("Robust UV–Vis spectral descriptors", y=1.04)
    savefig(FIG_MAIN / "Fig05_UVVis_robust_spectral_descriptors")


def best_linear_range_from_blank(color_df: pd.DataFrame, feature: str = "H_mean") -> pd.DataFrame:
    agg = color_df.groupby("concentration_mM").agg(mean=(feature, "mean"), sd=(feature, "std"), n=(feature, "count")).reindex(ALL_CONC).dropna()
    rows = []
    for maxc in range(5, 21):
        sub = agg.loc[agg.index <= maxc]
        if len(sub) < 5:
            continue
        x = sub.index.values.astype(float); y = sub["mean"].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1); pred = slope*x + intercept
        rows.append({"start_mM": 0, "end_mM": maxc, "n_levels": len(sub), "slope": slope, "intercept": intercept,
                     "R2": r2_score(y, pred), "RMSE_deg": math.sqrt(mean_squared_error(y, pred)), "MAE_deg": mean_absolute_error(y, pred)})
    cand = pd.DataFrame(rows)
    if cand.empty:
        return pd.DataFrame()
    eligible = cand[(cand["R2"] >= 0.95)]
    if not eligible.empty:
        # Prefer broadest range from blank, then high R2.
        best = eligible.sort_values(["end_mM", "R2"], ascending=[False, False]).head(1)
    else:
        best = cand.sort_values(["R2", "end_mM"], ascending=[False, False]).head(1)
    return best


def figure_6_analytical_calibration(color_df: pd.DataFrame) -> pd.DataFrame:
    if color_df.empty:
        return pd.DataFrame()
    agg = color_df.groupby("concentration_mM").agg(H=("H_mean", "mean"), Hsd=("H_mean", "std"), n=("H_mean", "count")).reindex(ALL_CONC).dropna()
    best = best_linear_range_from_blank(color_df, "H_mean")
    if best.empty:
        return pd.DataFrame()
    end = int(best.iloc[0]["end_mM"])
    sub = agg.loc[agg.index <= end]
    x = sub.index.values.astype(float); y = sub["H"].values.astype(float); yerr = sub["Hsd"].fillna(0).values.astype(float)
    slope, intercept = np.polyfit(x, y, 1); pred = slope*x + intercept
    r2 = r2_score(y, pred); rmse = math.sqrt(mean_squared_error(y, pred)); mae = mean_absolute_error(y, pred)
    sd_blank = color_df[color_df["concentration_mM"] == 0]["H_mean"].std(ddof=1)
    lod = 3.3 * sd_blank / abs(slope) if np.isfinite(sd_blank) and slope != 0 else np.nan
    loq = 10.0 * sd_blank / abs(slope) if np.isfinite(sd_blank) and slope != 0 else np.nan
    analytical = pd.DataFrame([{
        "Analytical feature": "HSV hue angle",
        "Linear range (mM)": f"0-{end}",
        "Sensitivity (deg mM^-1)": slope,
        "Intercept (deg)": intercept,
        "R2": r2,
        "RMSE (deg)": rmse,
        "MAE (deg)": mae,
        "Blank SD (deg)": sd_blank,
        "LOD (mM)": lod,
        "LOQ (mM)": loq,
    }])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.35), gridspec_kw={"width_ratios":[1.25,1]})
    xs = np.linspace(0, end, 200)
    axes[0].errorbar(x, y, yerr=yerr, fmt="o", capsize=3.5, ms=5.2, label="Mean ± SD")
    axes[0].plot(xs, slope*xs + intercept, lw=2.2, label="Linear fit")
    axes[0].set_xlabel("Borax concentration (mM)"); axes[0].set_ylabel("Hue angle (°)"); axes[0].set_title("Linear analytical range from blank")
    axes[0].grid(alpha=0.15); axes[0].legend(frameon=False)
    axes[0].text(0.05, 0.08, f"H = {slope:.2f}C + {intercept:.2f}\nR²={r2:.3f}; RMSE={rmse:.2f}°\nLOD={lod:.3f} mM; LOQ={loq:.3f} mM",
                 transform=axes[0].transAxes, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", lw=0.8)); add_panel_label(axes[0], "a")
    resid = y - pred
    axes[1].axhline(0, color="black", lw=0.9)
    axes[1].axhspan(-rmse, rmse, alpha=0.18, color="0.7")
    axes[1].vlines(x, 0, resid, color="0.5", lw=0.8)
    axes[1].scatter(x, resid, s=42, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Borax concentration (mM)"); axes[1].set_ylabel("Residual (°)"); axes[1].set_title("Residual diagnostics")
    axes[1].grid(alpha=0.15); add_panel_label(axes[1], "b")
    fig.suptitle("Analytical calibration and figures of merit", y=1.03)
    savefig(FIG_MAIN / "Fig06_analytical_calibration_LOD_LOQ")
    return analytical


def confidence_ellipse(ax, x, y, n_std=1.5, **kwargs):
    if len(x) < 3: return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)): return
    vals, vecs = np.linalg.eigh(cov); order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    if np.any(vals <= 0): return
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1])); width, height = 2*n_std*np.sqrt(vals)
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), width, height, angle=theta, fill=False, **kwargs))


def figure_7_chemometrics(color_df: pd.DataFrame, uv_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if color_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = feature_columns(color_df)
    Xdf = color_df[cols].fillna(color_df[cols].median())
    Xs = StandardScaler().fit_transform(Xdf.values)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)
    pca_load = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=cols).reset_index().rename(columns={"index":"feature"})
    y = color_df["concentration_mM"].values.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    sc = axes[0].scatter(Z[:,0], Z[:,1], c=y, s=32, edgecolor="black", linewidth=0.35)
    for conc in sorted(color_df["concentration_mM"].unique()):
        idx = y == conc
        if idx.sum() >= 3:
            confidence_ellipse(axes[0], Z[idx,0], Z[idx,1], n_std=1.05, lw=0.9, ec="0.25", alpha=0.65)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title("PCA score plot with confidence ellipses"); axes[0].grid(alpha=0.15)
    cb = fig.colorbar(sc, ax=axes[0], pad=0.02); cb.set_label("mM"); add_panel_label(axes[0], "a")
    # Correlation heatmap.
    agg = color_df.groupby("concentration_mM")[cols + ["pH"]].mean(numeric_only=True).reset_index()
    keep = ["concentration_mM", "pH", "H_mean", "Lab_a_mean", "Lab_b_mean", "Normalized_B", "B_over_R", "Excess_Blue_Index"]
    uv_corr_table = pd.DataFrame()
    if not uv_features.empty:
        uvagg = uv_features.groupby("concentration_mM").mean(numeric_only=True).reset_index()
        agg = agg.merge(uvagg[["concentration_mM", "lambda_max_nm", "Amax", "AUC"]], on="concentration_mM", how="left")
        keep += ["lambda_max_nm", "Amax", "AUC"]
        pairs = []
        for c1 in ["H_mean", "Lab_a_mean", "Lab_b_mean", "Normalized_B", "B_over_R", "Excess_Blue_Index"]:
            for c2 in ["lambda_max_nm", "Amax", "AUC"]:
                if c1 in agg and c2 in agg:
                    pairs.append({"color_feature": c1, "uvvis_feature": c2, "pearson_r": agg[[c1,c2]].corr().iloc[0,1]})
        uv_corr_table = pd.DataFrame(pairs).sort_values("pearson_r", key=lambda s: np.abs(s), ascending=False) if pairs else pd.DataFrame()
    keep = [c for c in keep if c in agg.columns]
    corr = agg[keep].corr()
    im = axes[1].imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_xticks(range(len(keep))); axes[1].set_yticks(range(len(keep)))
    axes[1].set_xticklabels(keep, rotation=45, ha="right", fontsize=7); axes[1].set_yticklabels(keep, fontsize=7)
    axes[1].set_title("Multimodal correlation matrix")
    for i in range(len(keep)):
        for j in range(len(keep)):
            axes[1].text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=5.8)
    cb2 = fig.colorbar(im, ax=axes[1], pad=0.02); cb2.set_label("r"); add_panel_label(axes[1], "b")
    fig.suptitle("Chemometric interpretation of colorimetric and spectral features", y=1.03)
    savefig(FIG_MAIN / "Fig07_PCA_and_multimodal_correlation")
    return pca_load, uv_corr_table


def external_validation(color_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if color_df.empty or not {"ganjil", "genap"}.issubset(set(color_df["dataset"].unique())):
        return pd.DataFrame(), pd.DataFrame()
    cols = feature_columns(color_df, include_ph=False)
    models = build_models(len(color_df))
    rows = []; pred_rows = []
    for direction, train_name, test_name in [("ganjil_to_genap", "ganjil", "genap"), ("genap_to_ganjil", "genap", "ganjil")]:
        train = color_df[color_df["dataset"] == train_name]; test = color_df[color_df["dataset"] == test_name]
        Xtr = train[cols].fillna(train[cols].median()).values; ytr = train["concentration_mM"].values.astype(float)
        Xte = test[cols].fillna(train[cols].median()).values; yte = test["concentration_mM"].values.astype(float)
        for name, model in models.items():
            try:
                m = clone(model); m.fit(Xtr, ytr); yp = np.asarray(m.predict(Xte)).ravel(); met = metric(yte, yp)
                rows.append({"validation": direction, "model": name, **met})
                for yt, yy, img in zip(yte, yp, test["image_file"]):
                    pred_rows.append({"validation": direction, "model": name, "image_file": img, "actual_mM": yt, "predicted_mM": yy, "residual_mM": yy-yt})
            except Exception:
                rows.append({"validation": direction, "model": name, "R2": np.nan, "RMSE_mM": np.nan, "MAE_mM": np.nan})
    perf = pd.DataFrame(rows).sort_values(["validation", "RMSE_mM"])
    preds = pd.DataFrame(pred_rows)
    return perf, preds


def internal_loio_validation(color_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if color_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = feature_columns(color_df, include_ph=False)
    X = color_df[cols].fillna(color_df[cols].median()).values
    y = color_df["concentration_mM"].values.astype(float)
    groups = (color_df["dataset"].astype(str) + "_" + color_df["image_file"].astype(str)).values
    models = build_models(len(color_df)); cv = LeaveOneGroupOut()
    rows = []; pred_df = pd.DataFrame({"actual_mM": y, "dataset": color_df["dataset"].values, "image_file": color_df["image_file"].values})
    for name, model in models.items():
        try:
            yp = cross_val_predict(model, X, y, cv=cv, groups=groups)
            rows.append({"model": name, **metric(y, yp)})
            pred_df[f"pred_{name}"] = yp
        except Exception:
            rows.append({"model": name, "R2": np.nan, "RMSE_mM": np.nan, "MAE_mM": np.nan})
    return pd.DataFrame(rows).sort_values("RMSE_mM"), pred_df


def figure_8_ml_external(color_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate robust external-validation ML figure.

    Revision notes
    --------------
    This version fixes the KeyError: 'validation' by forcing the external-
    validation result table to always have a stable schema. It also handles
    cases where only one dataset is available, where one validation direction
    fails, or where all models return NaN metrics.
    """
    perf_ext, pred_ext = external_validation(color_df)
    perf_int, pred_int = internal_loio_validation(color_df)

    required_perf_cols = ["validation", "model", "R2", "RMSE_mM", "MAE_mM"]
    required_pred_cols = ["validation", "model", "image_file", "actual_mM", "predicted_mM", "residual_mM"]

    if perf_ext is None or perf_ext.empty:
        perf_ext = pd.DataFrame(columns=required_perf_cols)
    else:
        perf_ext = perf_ext.copy()
        for col in required_perf_cols:
            if col not in perf_ext.columns:
                perf_ext[col] = np.nan
        perf_ext = perf_ext[required_perf_cols]
        perf_ext["validation"] = perf_ext["validation"].astype(str)
        perf_ext["model"] = perf_ext["model"].astype(str)
        perf_ext[["R2", "RMSE_mM", "MAE_mM"]] = perf_ext[["R2", "RMSE_mM", "MAE_mM"]].apply(pd.to_numeric, errors="coerce")

    if pred_ext is None or pred_ext.empty:
        pred_ext = pd.DataFrame(columns=required_pred_cols)
    else:
        pred_ext = pred_ext.copy()
        for col in required_pred_cols:
            if col not in pred_ext.columns:
                pred_ext[col] = np.nan
        pred_ext = pred_ext[required_pred_cols]
        pred_ext["validation"] = pred_ext["validation"].astype(str)
        pred_ext["model"] = pred_ext["model"].astype(str)
        pred_ext[["actual_mM", "predicted_mM", "residual_mM"]] = pred_ext[["actual_mM", "predicted_mM", "residual_mM"]].apply(pd.to_numeric, errors="coerce")

    if perf_int is None or perf_int.empty:
        perf_int = pd.DataFrame(columns=["model", "R2", "RMSE_mM", "MAE_mM"])
    if pred_int is None or pred_int.empty:
        pred_int = pd.DataFrame()

    perf_ext.to_csv(TAB_MAIN / "Table_ML_external_validation_summary.csv", index=False)
    pred_ext.to_csv(TAB_SUPP / "TableS_ML_external_validation_predictions.csv", index=False)
    perf_int.to_csv(TAB_SUPP / "TableS_internal_LOIO_model_performance.csv", index=False)
    pred_int.to_csv(TAB_SUPP / "TableS_internal_LOIO_predictions.csv", index=False)

    # Skip the main ML external figure gracefully if external validation is not available.
    valid_perf = perf_ext.dropna(subset=["RMSE_mM"]).copy()
    if valid_perf.empty or pred_ext.empty:
        msg = [
            "External validation figure was skipped.",
            "Reason: no valid ganjil-to-genap/genap-to-ganjil validation result was available.",
            "Please make sure both photo folders exist and are named 'ganjil' and 'genap' or one of the supported aliases.",
        ]
        (OUTPUT / "README_external_validation_skipped.txt").write_text("\n".join(msg), encoding="utf-8")
        return perf_ext, pred_ext, perf_int, pred_int

    # Select the best model for each validation direction.
    best = (
        valid_perf.sort_values(["validation", "RMSE_mM", "MAE_mM"], ascending=[True, True, True])
        .groupby("validation", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # Select the single best validation/model pair for predicted-vs-actual panel.
    best_row = valid_perf.sort_values(["RMSE_mM", "MAE_mM"], ascending=[True, True]).iloc[0]
    pbest = pred_ext[
        (pred_ext["validation"] == str(best_row["validation"]))
        & (pred_ext["model"] == str(best_row["model"]))
    ].dropna(subset=["actual_mM", "predicted_mM"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))

    # Panel a: best RMSE per validation direction.
    labels = best["validation"].astype(str).str.replace("_to_", " → ", regex=False).values
    axes[0].bar(labels, best["RMSE_mM"].values, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("RMSE (mM)")
    axes[0].set_title("Best model in each external validation")
    axes[0].grid(axis="y", alpha=0.15)
    axes[0].tick_params(axis="x", rotation=0)
    for i, rr in best.iterrows():
        r2txt = "NA" if pd.isna(rr["R2"]) else f"{rr['R2']:.3f}"
        rmse_y = 0 if pd.isna(rr["RMSE_mM"]) else rr["RMSE_mM"]
        axes[0].text(i, rmse_y, f"{rr['model']}\nR²={r2txt}", ha="center", va="bottom", fontsize=8)
    add_panel_label(axes[0], "a")

    # Panel b: predicted vs actual for best pair.
    if not pbest.empty:
        axes[1].scatter(pbest["actual_mM"], pbest["predicted_mM"], s=30, edgecolor="black", linewidth=0.4)
        minv = min(-1, float(np.nanmin(pbest[["actual_mM", "predicted_mM"]].values)) - 0.5)
        maxv = max(21, float(np.nanmax(pbest[["actual_mM", "predicted_mM"]].values)) + 0.5)
        axes[1].plot([minv, maxv], [minv, maxv], ls="--", lw=1.1, color="black")
        axes[1].set_xlim(minv, maxv)
        axes[1].set_ylim(minv, maxv)
        r2txt = "NA" if pd.isna(best_row["R2"]) else f"{best_row['R2']:.3f}"
        axes[1].text(
            0.05, 0.95,
            f"{str(best_row['validation']).replace('_to_', ' → ')}\n"
            f"{best_row['model']}\n"
            f"R²={r2txt}\nRMSE={best_row['RMSE_mM']:.2f} mM\nMAE={best_row['MAE_mM']:.2f} mM",
            transform=axes[1].transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", lw=0.7)
        )
    else:
        axes[1].text(0.5, 0.5, "No prediction rows available\nfor the best external model", ha="center", va="center")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Actual borax concentration (mM)")
    axes[1].set_ylabel("Predicted borax concentration (mM)")
    axes[1].set_title(f"Best external prediction: {best_row['model']}")
    axes[1].grid(alpha=0.15)
    add_panel_label(axes[1], "b")

    fig.suptitle("Machine-learning prediction under independent ganjil–genap validation", y=1.03)
    savefig(FIG_MAIN / "Fig08_ML_external_validation")
    return perf_ext, pred_ext, perf_int, pred_int

def figure_9_explainability(color_df: pd.DataFrame) -> pd.DataFrame:
    if color_df.empty:
        return pd.DataFrame()
    cols = feature_columns(color_df, include_ph=False)
    X = color_df[cols].fillna(color_df[cols].median()).values
    y = color_df["concentration_mM"].values.astype(float)
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=50, random_state=42, scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": cols, "importance_mean": result.importances_mean, "importance_sd": result.importances_std}).sort_values("importance_mean", ascending=False)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], capsize=3, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Permutation importance (increase in RMSE score)")
    ax.set_title("Explainable feature relevance for borax prediction")
    ax.grid(axis="x", alpha=0.15)
    savefig(FIG_MAIN / "Fig09_explainable_feature_importance")
    # Optional SHAP/extratrees figure.
    try:
        import shap
        tree = ExtraTreesRegressor(n_estimators=400, random_state=42, min_samples_leaf=1)
        tree.fit(X, y)
        explainer = shap.TreeExplainer(tree)
        sv = explainer.shap_values(X)
        shap_abs = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"feature": cols, "mean_abs_SHAP": shap_abs}).sort_values("mean_abs_SHAP", ascending=False)
        shap_df.to_csv(TAB_SUPP / "TableS_SHAP_feature_importance.csv", index=False)
        top2 = shap_df.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        ax.barh(top2["feature"], top2["mean_abs_SHAP"], edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("SHAP-based model explanation")
        ax.grid(axis="x", alpha=0.15)
        savefig(FIG_SUPP / "FigS09_SHAP_feature_importance")
    except Exception:
        pass
    return imp


def figure_10_bland_altman_and_permutation(color_df: pd.DataFrame, pred_ext: pd.DataFrame, perf_ext: pd.DataFrame) -> pd.DataFrame:
    perm = pd.DataFrame()
    if pred_ext.empty or perf_ext.empty:
        return perm
    best = perf_ext.iloc[0]
    p = pred_ext[(pred_ext["validation"] == best["validation"]) & (pred_ext["model"] == best["model"])].copy()
    mean_vals = (p["actual_mM"] + p["predicted_mM"]) / 2
    diff = p["predicted_mM"] - p["actual_mM"]
    bias = diff.mean(); sd = diff.std(ddof=1); loa_low, loa_high = bias - 1.96*sd, bias + 1.96*sd
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    axes[0].scatter(mean_vals, diff, s=28, edgecolor="black", linewidth=0.4)
    axes[0].axhline(bias, color="black", lw=1.0, label="Bias")
    axes[0].axhline(loa_low, color="0.35", lw=1.0, ls="--", label="95% limits")
    axes[0].axhline(loa_high, color="0.35", lw=1.0, ls="--")
    axes[0].set_xlabel("Mean of actual and predicted (mM)"); axes[0].set_ylabel("Prediction difference (mM)")
    axes[0].set_title("Bland–Altman agreement analysis"); axes[0].legend(frameon=False); axes[0].grid(alpha=0.15); add_panel_label(axes[0], "a")
    # permutation test for best validation/model.
    cols = feature_columns(color_df, include_ph=False)
    direction = best["validation"]
    train_name, test_name = direction.split("_to_")
    train = color_df[color_df["dataset"] == train_name]
    test = color_df[color_df["dataset"] == test_name]
    Xtr = train[cols].fillna(train[cols].median()).values; ytr = train["concentration_mM"].values.astype(float)
    Xte = test[cols].fillna(train[cols].median()).values; yte = test["concentration_mM"].values.astype(float)
    base_models = build_models(len(color_df)); model = base_models.get(best["model"], Ridge())
    rng = np.random.default_rng(42); scores = []
    for i in range(500):
        yp = rng.permutation(ytr)
        try:
            m = clone(model); m.fit(Xtr, yp); pred = np.asarray(m.predict(Xte)).ravel(); scores.append(r2_score(yte, pred))
        except Exception:
            pass
    scores = np.asarray(scores)
    obs = float(best["R2"])
    pval = float((np.sum(scores >= obs) + 1) / (len(scores) + 1)) if len(scores) else np.nan
    axes[1].hist(scores, bins=35, edgecolor="black", linewidth=0.35)
    axes[1].axvline(obs, color="black", lw=2.0, label=f"Observed R²={obs:.3f}")
    axes[1].set_xlabel("R² after label permutation"); axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Permutation test (p={pval:.4f})"); axes[1].legend(frameon=False); axes[1].grid(alpha=0.15); add_panel_label(axes[1], "b")
    fig.suptitle("External prediction agreement and overfitting control", y=1.03)
    savefig(FIG_MAIN / "Fig10_BlandAltman_and_permutation_test")
    perm = pd.DataFrame({"permuted_R2": scores})
    perm.attrs["observed_R2"] = obs
    return perm

# =============================================================================
# SUPPLEMENTARY FIGURES
# =============================================================================
def supplementary_figures(color_df: pd.DataFrame, uv_features: pd.DataFrame, perf_int: pd.DataFrame, pred_int: pd.DataFrame,
                          pca_loadings: pd.DataFrame, pred_ext: pd.DataFrame, perf_ext: pd.DataFrame) -> None:
    if color_df.empty:
        return
    # S1 ROI QC.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    axes[0].hist(color_df["roi_confidence"], bins=30, edgecolor="black", linewidth=0.35); axes[0].set_title("ROI confidence")
    axes[0].set_xlabel("Confidence score"); axes[0].set_ylabel("Frequency"); axes[0].grid(alpha=0.15)
    axes[1].scatter(color_df["concentration_mM"], color_df["n_pixels"], s=16, edgecolor="black", linewidth=0.25)
    axes[1].set_title("Analytical pixels per ROI"); axes[1].set_xlabel("mM"); axes[1].set_ylabel("Pixels after trimming"); axes[1].grid(alpha=0.15)
    yy = color_df.groupby(["dataset", "image_file"])["roi_cy"].std().reset_index(name="y_sd")
    axes[2].hist(yy["y_sd"], bins=20, edgecolor="black", linewidth=0.35); axes[2].set_title("Within-image row stability")
    axes[2].set_xlabel("SD of ROI y-position (px)"); axes[2].set_ylabel("Image count"); axes[2].grid(alpha=0.15)
    fig.suptitle("Figure S1. ROI extraction quality control", y=1.03)
    savefig(FIG_SUPP / "FigS01_ROI_quality_control")

    # S2 repeatability/precision.
    hue_summary = color_df.groupby("concentration_mM").agg(mean_H=("H_mean", "mean"), sd_H=("H_mean", "std"), n=("H_mean", "count")).reindex(ALL_CONC).dropna()
    hue_summary["CV_percent"] = 100 * hue_summary["sd_H"] / hue_summary["mean_H"].abs()
    hue_summary.reset_index().to_csv(TAB_SUPP / "TableS_repeatability_Hue_CV.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    data = [color_df[color_df["concentration_mM"] == c]["H_mean"].dropna().values for c in hue_summary.index]
    axes[0].boxplot(data, labels=[str(int(c)) for c in hue_summary.index], showfliers=False)
    axes[0].set_xlabel("Borax concentration (mM)"); axes[0].set_ylabel("Hue angle (°)"); axes[0].set_title("Hue repeatability")
    axes[0].tick_params(axis="x", rotation=45); axes[0].grid(axis="y", alpha=0.15)
    axes[1].bar([str(int(c)) for c in hue_summary.index], hue_summary["CV_percent"], edgecolor="black", linewidth=0.35)
    axes[1].set_xlabel("Borax concentration (mM)"); axes[1].set_ylabel("CV (%)"); axes[1].set_title("Precision by concentration")
    axes[1].tick_params(axis="x", rotation=45); axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle("Figure S2. Repeatability and precision of hue extraction", y=1.03)
    savefig(FIG_SUPP / "FigS02_repeatability_precision")

    # S3 DeltaE.
    cols = ["Lab_L_mean", "Lab_a_mean", "Lab_b_mean"]
    blank = color_df[color_df["concentration_mM"] == 0][cols].mean().values
    tmp = color_df.copy(); tmp["DeltaE_from_blank"] = np.sqrt(((tmp[cols].values - blank) ** 2).sum(axis=1))
    de = tmp.groupby("concentration_mM").agg(mean=("DeltaE_from_blank", "mean"), sd=("DeltaE_from_blank", "std"), n=("DeltaE_from_blank", "count")).reindex(ALL_CONC).dropna()
    de.reset_index().to_csv(TAB_SUPP / "TableS_DeltaE_from_blank.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(de.index, de["mean"], yerr=de["sd"].fillna(0), fmt="o-", capsize=3.5)
    ax.axhline(2.3, ls="--", lw=1.0, color="black", label="Approx. perceptibility threshold")
    ax.set_xlabel("Borax concentration (mM)"); ax.set_ylabel("ΔE*ab from blank"); ax.set_title("Figure S3. Perceptual color distance")
    ax.legend(frameon=False); ax.grid(alpha=0.15); savefig(FIG_SUPP / "FigS03_DeltaE_from_blank")

    # S4 feature trends.
    features = ["B_over_R", "B_over_G", "Normalized_B", "Excess_Blue_Index", "S_mean", "Lab_b_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0)); axes = axes.ravel()
    rows = []
    for ax, feat in zip(axes, features):
        if feat not in color_df.columns: continue
        g = color_df.groupby("concentration_mM").agg(mean=(feat, "mean"), sd=(feat, "std"), n=(feat, "count")).reindex(ALL_CONC).dropna()
        for conc, rr in g.iterrows(): rows.append({"feature": feat, "concentration_mM": conc, **rr.to_dict()})
        ax.errorbar(g.index, g["mean"], yerr=g["sd"].fillna(0), fmt="o-", capsize=3)
        ax.set_title(feat.replace("_", " ")); ax.set_xlabel("mM"); ax.grid(alpha=0.15)
    pd.DataFrame(rows).to_csv(TAB_SUPP / "TableS_feature_trends_summary.csv", index=False)
    fig.suptitle("Figure S4. Concentration-dependent feature trends", y=1.02)
    savefig(FIG_SUPP / "FigS04_feature_trends")

    # S5 Internal model comparison.
    if not perf_int.empty:
        top = perf_int.dropna().head(10)
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
        for ax, metric_name, ylabel in zip(axes, ["RMSE_mM", "MAE_mM", "R2"], ["RMSE (mM)", "MAE (mM)", "R²"]):
            pp = top.sort_values(metric_name, ascending=(metric_name != "R2"))
            ax.bar(range(len(pp)), pp[metric_name], edgecolor="black", linewidth=0.35)
            ax.set_xticks(range(len(pp))); ax.set_xticklabels(pp["model"], rotation=45, ha="right")
            ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.15)
        fig.suptitle("Figure S5. Internal leave-one-image-out model comparison", y=1.03)
        savefig(FIG_SUPP / "FigS05_internal_LOIO_model_comparison")

    # S6 Predicted vs actual internal best models.
    if not pred_int.empty and not perf_int.empty:
        top_models = perf_int.dropna().head(4)["model"].tolist()
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.8)); axes = axes.ravel()
        for ax, m in zip(axes, top_models):
            col = f"pred_{m}"
            if col not in pred_int.columns: continue
            yt = pred_int["actual_mM"].values; yp = pred_int[col].values; met = metric(yt, yp)
            ax.scatter(yt, yp, s=24, edgecolor="black", linewidth=0.35)
            ax.plot([-1, 21], [-1, 21], ls="--", color="black", lw=1.0)
            ax.set_xlim(-1, 21); ax.set_ylim(-1, 21); ax.set_title(m); ax.set_xlabel("Actual (mM)"); ax.set_ylabel("Predicted (mM)")
            ax.text(0.05, 0.93, f"R²={met['R2']:.3f}\nRMSE={met['RMSE_mM']:.2f} mM", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", lw=0.6))
            ax.grid(alpha=0.15)
        fig.suptitle("Figure S6. Internal LOIO predicted vs actual", y=1.02)
        savefig(FIG_SUPP / "FigS06_internal_LOIO_predicted_vs_actual")

    # S7 PCA loading plot.
    if not pca_loadings.empty:
        top_load = pca_loadings.copy(); top_load["abs_PC1"] = top_load["PC1"].abs(); top_load = top_load.sort_values("abs_PC1", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.barh(top_load["feature"], top_load["PC1"], edgecolor="black", linewidth=0.35)
        ax.axvline(0, color="black", lw=0.9); ax.set_xlabel("PC1 loading"); ax.set_title("Figure S7. PCA loading plot")
        ax.grid(axis="x", alpha=0.15); savefig(FIG_SUPP / "FigS07_PCA_loadings")

    # S8 color-UVVis correlation.
    if not uv_features.empty:
        uvagg = uv_features.groupby("concentration_mM").mean(numeric_only=True).reset_index()
        cagg = color_df.groupby("concentration_mM").mean(numeric_only=True).reset_index()
        m = cagg.merge(uvagg, on="concentration_mM", how="inner")
        pairs = [("H_mean", "lambda_max_nm"), ("H_mean", "Amax"), ("Lab_b_mean", "Amax"), ("Excess_Blue_Index", "AUC")]
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4)); axes = axes.ravel()
        for ax, (xcol, ycol) in zip(axes, pairs):
            if xcol not in m.columns or ycol not in m.columns: continue
            r = m[[xcol, ycol]].corr().iloc[0,1]
            ax.scatter(m[xcol], m[ycol], s=45, edgecolor="black", linewidth=0.45)
            z = np.polyfit(m[xcol], m[ycol], 1); xs = np.linspace(m[xcol].min(), m[xcol].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), lw=1.5); ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(f"r={r:.3f}"); ax.grid(alpha=0.15)
        fig.suptitle("Figure S8. Correlation between smartphone color and UV–Vis descriptors", y=1.02)
        savefig(FIG_SUPP / "FigS08_color_UVVis_correlation")

    # S10 clustering.
    cols = feature_columns(color_df)
    agg = color_df.groupby("concentration_mM")[cols].mean(numeric_only=True).reindex(ALL_CONC).dropna()
    X = StandardScaler().fit_transform(agg.values)
    Zlink = linkage(pdist(X), method="ward")
    fig = plt.figure(figsize=(12.0, 5.2)); gs = GridSpec(1, 2, width_ratios=[0.75, 1.35], figure=fig)
    axd = fig.add_subplot(gs[0,0]); dendrogram(Zlink, labels=[str(int(i)) for i in agg.index], ax=axd, orientation="left", color_threshold=None)
    axd.set_title("Hierarchical clustering"); axd.set_xlabel("Distance")
    axh = fig.add_subplot(gs[0,1]); im = axh.imshow(X, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    axh.set_yticks(range(len(agg.index))); axh.set_yticklabels([str(int(i)) for i in agg.index])
    axh.set_xticks(range(len(cols))); axh.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    axh.set_title("Standardized concentration-level features"); cb = fig.colorbar(im, ax=axh, pad=0.02); cb.set_label("z-score")
    fig.suptitle("Figure S10. Hierarchical clustering and feature heatmap", y=1.03)
    savefig(FIG_SUPP / "FigS10_hierarchical_clustering_heatmap")



# =============================================================================
# TOP-TIER ENHANCEMENT OVERRIDES
# These functions override earlier definitions to improve top-tier journal readiness.
# =============================================================================

def figure_1_workflow() -> None:
    """Premium workflow schematic with mechanism-aware analytical pipeline."""
    fig, ax = plt.subplots(figsize=(15.2, 5.7))
    ax.axis("off")
    steps = [
        ("Borax\nstandard", "alkalization", "1"),
        ("Purple cabbage\nanthocyanin", "pH-responsive probe", "2"),
        ("Molecular\nconversion", "flavylium ⇄ quinoidal", "3"),
        ("UV–Vis\nvalidation", "spectral fingerprints", "4"),
        ("Smartphone\nimaging", "controlled acquisition", "5"),
        ("ROI\nextraction", "zone-centroid QC", "6"),
        ("RGB–HSV–Lab\nfeatures", "color descriptors", "7"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", "8"),
        ("Borax\nprediction", "quantitative output", "9"),
    ]
    xs = np.linspace(0.055, 0.945, len(steps)); y=0.58
    colors = ["#EAF2FF", "#F4ECFF", "#F7F7F7", "#EDF7F2", "#FFF4E6", "#F7F7F7", "#EEF7FF", "#F2F2F2", "#EAF7EA"]
    for i, ((title, sub, num), x) in enumerate(zip(steps, xs)):
        box = FancyBboxPatch((x-0.047, y-0.17), 0.094, 0.255, boxstyle="round,pad=0.018",
                             fc=colors[i], ec="#1f1f1f", lw=1.05)
        ax.add_patch(box)
        circ = plt.Circle((x-0.034, y+0.064), 0.014, fc="#222222", ec="none")
        ax.add_patch(circ)
        ax.text(x-0.034, y+0.064, num, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
        ax.text(x, y+0.025, title, ha="center", va="center", fontsize=9.0, fontweight="bold")
        ax.text(x, y-0.095, sub, ha="center", va="center", fontsize=7.2, color="0.25")
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((x+0.051, y-0.035), (xs[i+1]-0.051, y-0.035),
                                         arrowstyle="->", mutation_scale=12, lw=1.1, color="#333333"))
    # Mechanism annotation band
    ax.add_patch(FancyBboxPatch((0.12, 0.18), 0.76, 0.10, boxstyle="round,pad=0.015", fc="white", ec="0.35", lw=0.8))
    ax.text(0.50, 0.23, "Borax-induced alkalization drives anthocyanin structural conversion, producing UV–Vis and smartphone-detectable color signatures",
            ha="center", va="center", fontsize=10.5)
    ax.text(0.5, 0.08, "Multimodal smartphone colorimetry workflow for anthocyanin-assisted borax quantification",
            ha="center", va="center", fontsize=13.2, fontweight="bold")
    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)



def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Premium mechanistic Figure 2 with improved Q1 layout.

    Final revision:
    - Panel a molecular schematic is enlarged and placed lower to remove the
      unused white space from the previous version.
    - Panel labels are separated from panel titles.
    - Panel b keeps nonlinear pH-hue coupling, 95% CI, Pearson/Spearman statistics,
      and a smaller colorbar.
    - Panel c reports representative smartphone colors, hue values, and HEX codes.
    """
    from matplotlib.patches import RegularPolygon
    from scipy.stats import pearsonr, spearmanr, t as student_t_local

    fig = plt.figure(figsize=(15.4, 5.45))
    gs = GridSpec(1, 3, width_ratios=[1.48, 1.05, 1.10], figure=fig, wspace=0.34)

    # ------------------------------------------------------------------
    # Panel a: schematic anthocyanin transformations
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.00, 1.02, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax.set_title("Anthocyanin molecular response to borax-induced alkalization",
                 fontsize=11.2, fontweight="bold", pad=12)

    def draw_species(cx, cy, scale, face_col, label, subtitle, note, oxonium=False, open_chain=False):
        if open_chain:
            ring1 = RegularPolygon((cx - 0.095*scale, cy), 6, radius=0.064*scale,
                                   orientation=np.pi/6, fc=face_col, ec="0.12", lw=1.05, alpha=0.96)
            ring2 = RegularPolygon((cx + 0.138*scale, cy + 0.008*scale), 6, radius=0.064*scale,
                                   orientation=np.pi/6, fc=face_col, ec="0.12", lw=1.05, alpha=0.96)
            ax.add_patch(ring1); ax.add_patch(ring2)
            ax.plot([cx - 0.030*scale, cx + 0.075*scale], [cy, cy + 0.006*scale], color="0.12", lw=1.2)
            ax.plot([cx + 0.008*scale, cx + 0.105*scale], [cy + 0.024*scale, cy + 0.028*scale], color="0.12", lw=1.0)
            ax.text(cx + 0.025*scale, cy + 0.072*scale, "C=C–C=O", ha="center", va="center", fontsize=7.3)
        else:
            ringA = RegularPolygon((cx - 0.095*scale, cy), 6, radius=0.064*scale,
                                   orientation=np.pi/6, fc=face_col, ec="0.12", lw=1.05, alpha=0.96)
            ringC = RegularPolygon((cx - 0.010*scale, cy), 6, radius=0.064*scale,
                                   orientation=np.pi/6, fc=face_col, ec="0.12", lw=1.05, alpha=0.96)
            ringB = RegularPolygon((cx + 0.115*scale, cy + 0.014*scale), 6, radius=0.058*scale,
                                   orientation=np.pi/6, fc=face_col, ec="0.12", lw=1.05, alpha=0.96)
            ax.add_patch(ringA); ax.add_patch(ringC); ax.add_patch(ringB)
            ax.plot([cx + 0.050*scale, cx + 0.070*scale], [cy + 0.018*scale, cy + 0.020*scale], color="0.12", lw=1.1)
            ax.text(cx - 0.010*scale, cy + 0.004*scale, "O" + ("⁺" if oxonium else ""),
                    ha="center", va="center", fontsize=8.5, fontweight="bold", color="0.05")
            ax.text(cx + 0.115*scale, cy + 0.096*scale, "OH", ha="center", va="center", fontsize=6.9)
            ax.text(cx - 0.145*scale, cy + 0.090*scale, "OH", ha="center", va="center", fontsize=6.9)
        ax.add_patch(FancyBboxPatch((cx - 0.185*scale, cy - 0.164*scale), 0.37*scale, 0.062*scale,
                                    boxstyle="round,pad=0.012,rounding_size=0.012",
                                    fc="white", ec="0.55", lw=0.65))
        ax.text(cx, cy - 0.132*scale, label, ha="center", va="center", fontsize=8.2, fontweight="bold")
        ax.text(cx, cy - 0.192*scale, subtitle, ha="center", va="center", fontsize=7.1, color="0.30")
        ax.text(cx, cy - 0.240*scale, note, ha="center", va="center", fontsize=6.5, color="0.35")

    ax.annotate("increasing borax concentration / pH", xy=(0.91, 0.82), xytext=(0.14, 0.82),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.25"),
                ha="center", va="center", fontsize=9.2, fontweight="bold")

    species = [
        (0.18, 0.50, "Flavylium\ncation", "acidic / purple", "dominant at lower pH", "#8353A1", True, False),
        (0.50, 0.50, "Quinoidal\nbase", "mild alkaline / blue", "deprotonated chromophore", "#3568B7", False, False),
        (0.82, 0.50, "Anionic /\nchalcone-like", "alkaline / green-blue", "ring-opening equilibrium", "#2D7B69", False, True),
    ]
    for args in species:
        cx, cy, lab, sub, note, col, ox, op = args
        draw_species(cx, cy, 1.05, col, lab, sub, note, oxonium=ox, open_chain=op)
    for x0, x1 in [(0.30, 0.38), (0.62, 0.70)]:
        ax.add_patch(FancyArrowPatch((x0, 0.555), (x1, 0.555), arrowstyle="-|>", mutation_scale=11, lw=1.0, color="0.20"))
        ax.add_patch(FancyArrowPatch((x1, 0.450), (x0, 0.450), arrowstyle="-|>", mutation_scale=11, lw=1.0, color="0.20"))
    ax.text(0.5, 0.055,
            "Simplified anthocyanin-species schematic; substituent pattern depends on the natural extract composition.",
            ha="center", va="center", fontsize=7.3, color="0.35")

    # ------------------------------------------------------------------
    # Panel b: pH-hue mechanistic coupling
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(-0.22, 1.02, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax2.set_title("pH–hue mechanistic coupling", fontsize=11.2, fontweight="bold", pad=12)
    if not color_df.empty and {"pH", "H_mean", "concentration_mM"}.issubset(color_df.columns):
        agg = color_df.groupby("concentration_mM").agg(
            H=("H_mean", "mean"), Hsd=("H_mean", "std"), pH=("pH", "mean")
        ).reindex(ALL_CONC).dropna()
        if "pH" not in agg.columns:
            agg["pH"] = agg["concentration_mM"].round().astype(int).map(PH_MAP).astype(float)
        x = agg["pH"].values.astype(float)
        y = agg["H"].values.astype(float)
        yerr = agg["Hsd"].fillna(0).values.astype(float)
        conc = agg.index.values.astype(float)
        sc = ax2.scatter(x, y, c=conc, cmap="viridis", s=45, edgecolor="black", linewidth=0.45, zorder=3)
        ax2.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.38", elinewidth=0.8, capsize=2.6, zorder=2)
        if len(x) >= 5 and np.nanstd(x) > 0:
            deg = 2 if len(x) >= 6 else 1
            coeff = np.polyfit(x, y, deg)
            poly = np.poly1d(coeff)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 250)
            yhat = poly(x); ys = poly(xs)
            dof = max(len(x) - (deg + 1), 1)
            s_err = np.sqrt(np.sum((y - yhat) ** 2) / dof)
            xbar = np.mean(x); sxx = np.sum((x - xbar) ** 2)
            if sxx > 0:
                tval = student_t_local.ppf(0.975, dof)
                ci = tval * s_err * np.sqrt(1.0 / len(x) + (xs - xbar) ** 2 / sxx)
                ax2.fill_between(xs, ys - ci, ys + ci, color="0.20", alpha=0.12, lw=0, zorder=1)
            ax2.plot(xs, ys, color="0.15", lw=1.9, zorder=2)
            r, p = pearsonr(x, y); rho, ps = spearmanr(x, y)
            ax2.text(0.96, 0.96,
                     f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\nR² = {r*r:.3f}\np = {p:.1e}",
                     transform=ax2.transAxes, ha="right", va="top", fontsize=8.0,
                     bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.55", lw=0.7, alpha=0.92))
        for ci_, xi, yi in zip(conc, x, y):
            if ci_ in [0, 5, 10, 15, 20]:
                ax2.annotate(str(int(ci_)), (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=7.8)
        cb = fig.colorbar(sc, ax=ax2, fraction=0.032, pad=0.018)
        cb.set_label("Borax concentration (mM)", fontsize=8.4)
        cb.ax.tick_params(labelsize=7.8)
    ax2.set_xlabel("Measured pH")
    ax2.set_ylabel("HSV hue angle (°)")
    ax2.grid(alpha=0.14, lw=0.55)

    # ------------------------------------------------------------------
    # Panel c: smartphone color trajectory and hue readout
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(-0.10, 1.02, "c", transform=ax3.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax3.set_title("Smartphone color trajectory and hue readout", fontsize=11.2, fontweight="bold", pad=12)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    if not color_df.empty:
        show = [0, 2, 5, 8, 10, 12, 15, 18, 20]
        agg2 = color_df.groupby("concentration_mM").agg(
            R=("R_mean", "mean"), G=("G_mean", "mean"), B=("B_mean", "mean"), H=("H_mean", "mean")
        )
        available = [c for c in show if c in agg2.index]
        y0, dy = 0.82, 0.080
        for i, c in enumerate(available):
            row = agg2.loc[c]
            rgb = np.clip([row["R"], row["G"], row["B"]], 0, 255).astype(int)
            y = y0 - i*dy
            ax3.text(0.05, y, f"{int(c)} mM", ha="left", va="center", fontsize=8.2, fontweight="bold")
            ax3.add_patch(FancyBboxPatch((0.19, y-0.030), 0.22, 0.056,
                                         boxstyle="round,pad=0.012,rounding_size=0.008",
                                         fc=rgb/255.0, ec="0.22", lw=0.55))
            ax3.text(0.45, y, f"H = {row['H']:.1f}°", ha="left", va="center", fontsize=8.1)
            ax3.text(0.74, y, rgb_to_hex(rgb), ha="left", va="center", fontsize=7.5, color="0.35")
        means = color_df.groupby("concentration_mM")[["R_mean", "G_mean", "B_mean"]].mean().reindex(ALL_CONC).dropna()
        n = len(means)
        if n > 0:
            for i, (conc, row) in enumerate(means.iterrows()):
                ax3.add_patch(Rectangle((0.08 + 0.82*i/n, 0.095), 0.82/n, 0.064,
                                        fc=np.clip(row.values/255.0, 0, 1), ec="white", lw=0.25))
            for c in [0, 5, 10, 15, 20]:
                if c in means.index:
                    i = list(means.index).index(c)
                    ax3.text(0.08 + 0.82*(i + 0.5)/n, 0.062, str(c), ha="center", va="top", fontsize=7.0)
            ax3.text(0.49, 0.022, "Borax concentration (mM)", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 y=1.035, fontsize=13.6, fontweight="bold")
    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale")


def _candidate_calibration_features(color_df: pd.DataFrame) -> List[str]:
    candidates = ["H_mean", "Lab_b_mean", "Lab_a_mean", "Normalized_B", "Normalized_G", "B_over_G", "B_over_R", "Excess_Blue_Index", "S_mean", "Hue_angle_ab"]
    return [c for c in candidates if c in color_df.columns]


def _best_calibration_from_blank(color_df: pd.DataFrame) -> dict:
    best = None
    for feat in _candidate_calibration_features(color_df):
        agg = color_df.groupby("concentration_mM").agg(mean=(feat,"mean"), sd=(feat,"std"), n=(feat,"count")).reindex(ALL_CONC).dropna()
        if 0 not in agg.index or len(agg) < 5:
            continue
        for end in range(5, 21):
            sub = agg.loc[agg.index <= end]
            if len(sub) < 5 or 0 not in sub.index:
                continue
            x = sub.index.values.astype(float); y = sub["mean"].values.astype(float)
            slope, intercept = np.polyfit(x,y,1); pred=slope*x+intercept
            r2 = r2_score(y,pred); rmse=math.sqrt(mean_squared_error(y,pred)); mae=mean_absolute_error(y,pred)
            # Score favors high R2, wide range, low RMSE normalized by response span.
            span = max(np.ptp(y), 1e-9)
            score = 4*r2 + 0.06*end - 0.25*(rmse/span)
            row = dict(feature=feat, end=end, x=x, y=y, ysd=sub["sd"].fillna(0).values.astype(float), slope=slope, intercept=intercept, r2=r2, rmse=rmse, mae=mae, score=score)
            if best is None or row["score"] > best["score"]:
                best = row
    return best if best is not None else {}


def figure_6_analytical_calibration(color_df: pd.DataFrame) -> pd.DataFrame:
    """Dual calibration: optimized blank-starting linear range + nonlinear full-range hue response."""
    if color_df.empty:
        return pd.DataFrame()
    best = _best_calibration_from_blank(color_df)
    if not best:
        return pd.DataFrame()
    feat = best["feature"]; x=best["x"]; y=best["y"]; yerr=best["ysd"]
    slope=best["slope"]; intercept=best["intercept"]; r2=best["r2"]; rmse=best["rmse"]; mae=best["mae"]; end=best["end"]
    sd_blank = color_df[color_df["concentration_mM"]==0][feat].std(ddof=1)
    lod = 3.3*sd_blank/abs(slope) if np.isfinite(sd_blank) and abs(slope)>1e-12 else np.nan
    loq = 10*sd_blank/abs(slope) if np.isfinite(sd_blank) and abs(slope)>1e-12 else np.nan
    analytical = pd.DataFrame([{
        "Analytical feature": feat,
        "Linear range (mM)": f"0-{end}",
        "Sensitivity (feature unit mM^-1)": slope,
        "Intercept": intercept,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Blank SD": sd_blank,
        "LOD (mM)": lod,
        "LOQ (mM)": loq,
    }])

    # Full-range hue nonlinear fit.
    hue = color_df.groupby("concentration_mM").agg(H=("H_mean","mean"), Hsd=("H_mean","std")).reindex(ALL_CONC).dropna()
    xh=hue.index.values.astype(float); yh=hue["H"].values.astype(float); yherr=hue["Hsd"].fillna(0).values.astype(float)
    def exp_model(C, Hinf, A, k):
        return Hinf + A*np.exp(-k*C)
    try:
        p0=[float(np.min(yh)), float(np.max(yh)-np.min(yh)), 0.08]
        popt,_=curve_fit(exp_model,xh,yh,p0=p0,bounds=([0,-500,0.0001],[360,500,5]),maxfev=20000)
        yhfit=exp_model(xh,*popt); r2h=r2_score(yh,yhfit); rmseh=math.sqrt(mean_squared_error(yh,yhfit))
        xs_h=np.linspace(xh.min(),xh.max(),300); ys_h=exp_model(xs_h,*popt)
        eq_h=f"H={popt[0]:.1f}+{popt[1]:.1f}e$^{{-{popt[2]:.3f}C}}$"
    except Exception:
        coeff=np.polyfit(xh,yh,2); poly=np.poly1d(coeff); yhfit=poly(xh); r2h=r2_score(yh,yhfit); rmseh=math.sqrt(mean_squared_error(yh,yhfit)); xs_h=np.linspace(xh.min(),xh.max(),300); ys_h=poly(xs_h); eq_h="Quadratic fit"

    fig, axes = plt.subplots(1,3,figsize=(15.6,4.35),gridspec_kw={"width_ratios":[1.15,0.95,1.1]})
    xs=np.linspace(0,end,200)
    axes[0].errorbar(x,y,yerr=yerr,fmt="o",capsize=3.5,ms=5.2,label="Mean ± SD")
    axes[0].plot(xs,slope*xs+intercept,lw=2.2,label="Linear fit")
    axes[0].set_xlabel("Borax concentration (mM)"); axes[0].set_ylabel(feat.replace("_"," "))
    axes[0].set_title("Optimized linear analytical range")
    axes[0].grid(alpha=0.15); axes[0].legend(frameon=False)
    axes[0].text(0.05,0.08,f"Feature: {feat}\nRange = 0–{end} mM\ny = {slope:.3f}C + {intercept:.2f}\nR²={r2:.3f}; RMSE={rmse:.2f}\nLOD={lod:.3f} mM; LOQ={loq:.3f} mM", transform=axes[0].transAxes, fontsize=8.2, bbox=dict(boxstyle="round,pad=0.32",fc="white",ec="0.55",lw=0.75))
    add_panel_label(axes[0],"a")
    resid=y-(slope*x+intercept)
    axes[1].axhline(0,color="black",lw=0.9); axes[1].axhspan(-rmse,rmse,alpha=0.18,color="0.7")
    axes[1].vlines(x,0,resid,color="0.55",lw=0.8); axes[1].scatter(x,resid,s=38,edgecolor="black",linewidth=0.45)
    axes[1].set_xlabel("Borax concentration (mM)"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residual diagnostics")
    axes[1].grid(alpha=0.15); add_panel_label(axes[1],"b")
    axes[2].errorbar(xh,yh,yerr=yherr,fmt="o",capsize=3.2,ms=5.0,label="Mean ± SD")
    axes[2].plot(xs_h,ys_h,lw=2.2,label="Nonlinear fit")
    axes[2].set_xlabel("Borax concentration (mM)"); axes[2].set_ylabel("Hue angle (°)"); axes[2].set_title("Full-range nonlinear hue response")
    axes[2].grid(alpha=0.15); axes[2].legend(frameon=False)
    axes[2].text(0.05,0.08,f"{eq_h}\nR²={r2h:.3f}; RMSE={rmseh:.2f}°", transform=axes[2].transAxes, fontsize=8.2, bbox=dict(boxstyle="round,pad=0.32",fc="white",ec="0.55",lw=0.75))
    add_panel_label(axes[2],"c")
    fig.suptitle("Analytical calibration without forcing nonlinear full-range response into one linear model", y=1.03)
    savefig(FIG_MAIN / "Fig06_analytical_calibration_dual_linear_nonlinear")
    return analytical


def figure_9_explainability(color_df: pd.DataFrame) -> pd.DataFrame:
    """Top-tier explainability: SHAP beeswarm when available; otherwise permutation importance."""
    if color_df.empty:
        return pd.DataFrame()
    cols = feature_columns(color_df, include_ph=False)
    Xdf = color_df[cols].fillna(color_df[cols].median())
    X = Xdf.values; y = color_df["concentration_mM"].values.astype(float)
    imp = pd.DataFrame()
    try:
        import shap
        model = ExtraTreesRegressor(n_estimators=500, random_state=42, min_samples_leaf=1)
        model.fit(X,y)
        explainer=shap.TreeExplainer(model)
        sv=explainer.shap_values(X)
        imp = pd.DataFrame({"feature":cols,"mean_abs_SHAP":np.abs(sv).mean(axis=0)}).sort_values("mean_abs_SHAP",ascending=False)
        imp.to_csv(TAB_MAIN / "Table3_SHAP_feature_importance.csv", index=False)
        # Summary beeswarm-style using shap if possible.
        plt.figure(figsize=(8.2,5.6))
        shap.summary_plot(sv, Xdf, plot_type="dot", show=False, max_display=12)
        plt.title("SHAP explanation of borax prediction model", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIG_MAIN / "Fig09_SHAP_explainable_feature_importance.png", dpi=FIG_DPI, bbox_inches="tight")
        if SAVE_PDF:
            plt.savefig(FIG_MAIN / "Fig09_SHAP_explainable_feature_importance.pdf", bbox_inches="tight")
        plt.close()
        # Also save clean bar plot for table readability.
        top=imp.head(12).iloc[::-1]
        fig, ax=plt.subplots(figsize=(7.4,5.0))
        ax.barh(top["feature"],top["mean_abs_SHAP"],edgecolor="black",linewidth=0.4)
        ax.set_xlabel("Mean absolute SHAP value"); ax.set_title("SHAP-based feature relevance")
        ax.grid(axis="x",alpha=0.15); savefig(FIG_SUPP / "FigS_SHAP_bar_feature_importance")
    except Exception:
        model=Pipeline([("scaler",StandardScaler()),("model",Ridge(alpha=1.0))]); model.fit(X,y)
        result=permutation_importance(model,X,y,n_repeats=80,random_state=42,scoring="neg_root_mean_squared_error")
        imp=pd.DataFrame({"feature":cols,"importance_mean":result.importances_mean,"importance_sd":result.importances_std}).sort_values("importance_mean",ascending=False)
        top=imp.head(12).iloc[::-1]
        fig, ax=plt.subplots(figsize=(7.4,5.0))
        ax.barh(top["feature"],top["importance_mean"],xerr=top["importance_sd"],capsize=3,edgecolor="black",linewidth=0.4)
        ax.set_xlabel("Permutation importance (increase in RMSE score)"); ax.set_title("Explainable feature relevance for borax prediction")
        ax.grid(axis="x",alpha=0.15); savefig(FIG_MAIN / "Fig09_explainable_feature_importance")
    return imp


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# FINAL PREMIUM OVERRIDES: reviewer-oriented figures for Q1 top-tier submission
# =============================================================================
def _mean_rgb_table(color_df: pd.DataFrame) -> pd.DataFrame:
    if color_df.empty:
        return pd.DataFrame()
    return color_df.groupby("concentration_mM")[["R_mean", "G_mean", "B_mean"]].mean().sort_index()


def _draw_mean_color_strip(ax, color_df: pd.DataFrame, concs: Optional[List[int]] = None, label_every: int = 5) -> None:
    means = _mean_rgb_table(color_df)
    if means.empty:
        ax.axis("off"); return
    if concs is None:
        concs = [int(c) for c in means.index if int(c) in ALL_CONC]
    for i, c in enumerate(concs):
        if c not in means.index:
            continue
        ax.add_patch(Rectangle((i, 0), 1, 1, fc=np.clip(means.loc[c].values / 255, 0, 1), ec="white", lw=0.45))
        if (c % label_every == 0) or c in [0, 20]:
            ax.text(i + 0.5, -0.13, str(c), ha="center", va="top", fontsize=8)
    ax.set_xlim(0, len(concs)); ax.set_ylim(-0.28, 1.02); ax.axis("off")


def _get_avg_uvvis(uv_spectra: pd.DataFrame) -> pd.DataFrame:
    if uv_spectra.empty:
        return pd.DataFrame()
    return uv_spectra.groupby(["concentration_mM", "wavelength_nm"])["absorbance"].mean().reset_index()


def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Premium two-row graphical abstract-style workflow.

    Final Q1 revision:
    - The workflow is split into two readable rows.
    - The curved UV–Vis-to-smartphone arrow is removed because UV–Vis is an
      independent validation layer, not the causal input to smartphone imaging.
    - A clean divider separates chemistry validation from digital quantification.
    - Lower panels remain data-derived: smartphone color scale, UV–Vis spectra,
      and prediction/interpretable AI block.
    """
    fig = plt.figure(figsize=(16.2, 8.0))
    gs = GridSpec(
        3, 12, figure=fig,
        height_ratios=[1.40, 0.13, 1.22],
        width_ratios=[1]*12,
        hspace=0.28, wspace=0.58
    )

    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    chemistry_steps = [
        ("Borax\nstandard", "alkalization", "#e9f3ff"),
        ("Purple cabbage\nanthocyanin", "natural pH probe", "#f2e9fb"),
        ("Anthocyanin\nconversion", "flavylium ⇄ quinoidal", "#f7f7f7"),
        ("UV–Vis\nvalidation", "independent spectral fingerprint", "#edf8f2"),
    ]
    digital_steps = [
        ("Smartphone\nimaging", "controlled acquisition", "#fff2df"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", "#ecf6fd"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", "#f5f5f5"),
        ("Borax\nprediction", "quantitative mM output", "#edf9ed"),
    ]

    def draw_row(steps, y, label, start_number):
        x0, x1 = 0.075, 0.925
        gap = 0.035
        box_w = (x1 - x0 - gap * (len(steps)-1)) / len(steps)
        box_h = 0.245
        # row label as a clean vertical badge, not competing with the boxes
        ax.text(x0 - 0.045, y + box_h/2, label, rotation=90, ha="center", va="center",
                fontsize=9.0, fontweight="bold", color="0.18")
        centers = []
        for i, (title, subtitle, fc) in enumerate(steps):
            x = x0 + i * (box_w + gap)
            centers.append((x + box_w/2, y + box_h/2))
            box = FancyBboxPatch((x, y), box_w, box_h,
                                 boxstyle="round,pad=0.018,rounding_size=0.024",
                                 fc=fc, ec="0.16", lw=1.0)
            ax.add_patch(box)
            ax.add_patch(Circle((x + 0.026, y + box_h - 0.044), 0.017, fc="0.15", ec="none"))
            ax.text(x + 0.026, y + box_h - 0.044, str(start_number + i),
                    ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")
            ax.text(x + box_w/2, y + 0.155, title, ha="center", va="center",
                    fontsize=10.0, fontweight="bold", linespacing=0.82)
            ax.text(x + box_w/2, y + 0.054, subtitle, ha="center", va="center",
                    fontsize=7.8, color="0.28")
            if i < len(steps)-1:
                ax.add_patch(FancyArrowPatch((x + box_w + 0.006, y + box_h/2),
                                             (x + box_w + gap - 0.006, y + box_h/2),
                                             arrowstyle="-|>", mutation_scale=12,
                                             lw=1.0, color="0.22"))
        return centers

    draw_row(chemistry_steps, 0.62, "Chemistry\nvalidation", 1)
    draw_row(digital_steps, 0.25, "Digital\nquantification", 5)

    # Clean separator showing that UV–Vis validates the chemistry layer, while
    # smartphone/AI form a separate quantification layer.
    ax.plot([0.075, 0.925], [0.525, 0.525], color="0.70", lw=0.8, ls=(0, (3, 3)))
    ax.text(0.925, 0.535, "independent validation + prediction layers", ha="right", va="bottom",
            fontsize=7.6, color="0.35")

    ax.text(0.5, 0.075,
            "Borax-induced alkalization transforms anthocyanin chemistry into spectroscopically validated smartphone color signatures for explainable quantitative prediction",
            ha="center", va="center", fontsize=10.6, fontweight="bold")

    # ------------------------------------------------------------------
    # Data-derived lower panels
    # ------------------------------------------------------------------
    ax_sw = fig.add_subplot(gs[2, 0:3])
    ax_uv = fig.add_subplot(gs[2, 3:8])
    ax_ml = fig.add_subplot(gs[2, 8:12])

    ax_sw.set_title("Data-derived smartphone color scale", fontsize=10.7, pad=8)
    _draw_mean_color_strip(ax_sw, color_df, ALL_CONC, label_every=5)
    ax_sw.text(0.5, -0.31, "Borax concentration (mM)", transform=ax_sw.transAxes,
               ha="center", va="top", fontsize=8.8)
    ax_sw.text(-0.08, 1.10, "a", transform=ax_sw.transAxes, fontsize=12, fontweight="bold")

    ax_uv.set_title("Representative UV–Vis fingerprints", fontsize=10.7, pad=8)
    avg = _get_avg_uvvis(uv_spectra)
    if not avg.empty:
        for c in [0, 5, 10, 15, 20]:
            if c in set(avg["concentration_mM"].astype(float)):
                g = avg[np.isclose(avg["concentration_mM"].astype(float), float(c))]
                ax_uv.plot(g["wavelength_nm"], g["absorbance"], lw=1.45, label=f"{c} mM")
        ax_uv.set_xlabel("Wavelength (nm)")
        ax_uv.set_ylabel("Absorbance")
        ax_uv.legend(frameon=False, fontsize=7.6, loc="upper right")
        ax_uv.grid(alpha=0.13, lw=0.6)
    else:
        ax_uv.text(0.5, 0.5, "UV–Vis data not available", ha="center", va="center")
        ax_uv.axis("off")
    ax_uv.text(-0.08, 1.10, "b", transform=ax_uv.transAxes, fontsize=12, fontweight="bold")

    ax_ml.set_title("Independent prediction and interpretation", fontsize=10.7, pad=8)
    ax_ml.set_xlim(0, 1); ax_ml.set_ylim(0, 1); ax_ml.axis("off")
    nodes = [
        ("Color feature matrix", "RGB, HSV, CIELab"),
        ("Independent validation", "Set A ↔ Set B"),
        ("Feature attribution", "Permutation / SHAP"),
        ("Borax concentration", "mM output"),
    ]
    ys = [0.80, 0.58, 0.36, 0.14]
    for (title, sub), y in zip(nodes, ys):
        ax_ml.add_patch(FancyBboxPatch((0.12, y - 0.065), 0.76, 0.115,
                                       boxstyle="round,pad=0.025,rounding_size=0.018",
                                       fc="white", ec="0.28", lw=0.9))
        ax_ml.text(0.50, y + 0.014, title, ha="center", va="center", fontsize=8.8, fontweight="bold")
        ax_ml.text(0.50, y - 0.032, sub, ha="center", va="center", fontsize=7.4, color="0.33")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax_ml.add_patch(FancyArrowPatch((0.5, y0 - 0.075), (0.5, y1 + 0.075),
                                        arrowstyle="->", mutation_scale=11, lw=0.9, color="0.25"))
    ax_ml.text(-0.08, 1.10, "c", transform=ax_ml.transAxes, fontsize=12, fontweight="bold")

    fig.suptitle("Multimodal smartphone colorimetry workflow for anthocyanin-assisted borax quantification",
                 y=0.985, fontsize=14.2, fontweight="bold")
    savefig(FIG_MAIN / "Fig01_workflow_schematic")


def figure_2_roi_and_color_gallery(color_df: pd.DataFrame) -> None:
    """Tighter ROI gallery with five representative concentrations plus mean colors."""
    if color_df.empty:
        return
    reps = [0, 5, 10, 15, 20]
    selected = []
    for c in reps:
        sub = color_df[color_df["concentration_mM"] == c]
        if not sub.empty:
            selected.append(sub.iloc[0])
    if not selected:
        return
    fig, axes = plt.subplots(3, len(selected), figsize=(3.1 * len(selected), 7.4), gridspec_kw={"height_ratios":[1.1,1.1,0.55]})
    if len(selected) == 1:
        axes = np.array(axes).reshape(3,1)
    mean_rgb = _mean_rgb_table(color_df)
    for col, row in enumerate(selected):
        img_path = ROOT / str(row["dataset"]) / str(row["image_file"])
        if not img_path.exists():
            # fallback to folder candidates
            for folder in [first_existing_folder(PHOTO_GANJIL_CANDIDATES), first_existing_folder(PHOTO_GENAP_CANDIDATES)]:
                if folder and (folder / str(row["image_file"])).exists():
                    img_path = folder / str(row["image_file"]); break
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        conc = int(row["concentration_mM"])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cx, cy, rr = int(row["roi_cx"]), int(row["roi_cy"]), int(row["roi_radius"])
            cr = int(rr * 2.15)  # tighter crop
            h, w = img_rgb.shape[:2]
            x1, x2 = max(0, cx-cr), min(w, cx+cr); y1, y2 = max(0, cy-cr), min(h, cy+cr)
            crop = img_rgb[y1:y2, x1:x2].copy()
            lc_x, lc_y = cx-x1, cy-y1
            axes[0, col].imshow(crop)
            axes[0, col].add_patch(Circle((lc_x, lc_y), rr, fill=False, lw=1.5, ec="black"))
            mask = circular_mask(crop.shape, lc_x, lc_y, rr)
            vis = np.ones_like(crop) * 255
            vis[mask] = crop[mask]
            axes[1, col].imshow(vis)
        else:
            axes[0, col].text(0.5,0.5,"Image not found",ha="center",va="center"); axes[1,col].axis("off")
        axes[0, col].set_title(f"{conc} mM", fontsize=11)
        axes[0, col].axis("off"); axes[1, col].axis("off")
        if conc in mean_rgb.index:
            rgb = np.clip(mean_rgb.loc[conc].values, 0, 255).astype(np.uint8)
            sw = np.ones((90, 160, 3), dtype=np.uint8) * rgb.reshape(1,1,3)
            axes[2, col].imshow(sw)
            axes[2, col].text(0.5, -0.18, rgb_to_hex(rgb), transform=axes[2,col].transAxes, ha="center", va="top", fontsize=8)
        axes[2, col].axis("off")
    axes[0,0].text(-0.10,0.5,"ROI overlay", transform=axes[0,0].transAxes, rotation=90, ha="right", va="center", fontsize=10)
    axes[1,0].text(-0.10,0.5,"Analytical mask", transform=axes[1,0].transAxes, rotation=90, ha="right", va="center", fontsize=10)
    axes[2,0].text(-0.10,0.5,"Mean color", transform=axes[2,0].transAxes, rotation=90, ha="right", va="center", fontsize=10)
    fig.suptitle("Representative ROI validation and concentration-dependent color evolution", y=0.98)
    savefig(FIG_MAIN / "Fig02_representative_ROI_and_color_gallery")


def figure_6_analytical_calibration(color_df: pd.DataFrame) -> pd.DataFrame:
    """Clean dual calibration: blank-anchored linear range plus full-range nonlinear response."""
    if color_df.empty:
        return pd.DataFrame()
    # Prioritize physically interpretable features; choose best blank-anchored linear window with at least 6 levels.
    candidate_features = ["H_mean", "Lab_b_mean", "Excess_Blue_Index", "Normalized_B", "B_over_G", "S_mean"]
    candidate_features = [c for c in candidate_features if c in color_df.columns]
    records = []
    grouped = color_df.groupby("concentration_mM")
    for feat in candidate_features:
        g = grouped.agg(mean=(feat,"mean"), sd=(feat,"std"), n=(feat,"count")).reindex(ALL_CONC).dropna()
        for maxc in range(8, 21):
            sub = g.loc[g.index <= maxc]
            if len(sub) < 6 or 0 not in sub.index:
                continue
            x = sub.index.values.astype(float); y = sub["mean"].values.astype(float)
            slope, intercept = np.polyfit(x, y, 1); yp = slope*x + intercept
            records.append({"feature":feat,"range":f"0-{maxc}","maxc":maxc,"R2":r2_score(y,yp),"RMSE":math.sqrt(mean_squared_error(y,yp)),"slope":slope,"intercept":intercept,"n":len(sub)})
    cand = pd.DataFrame(records)
    if cand.empty:
        return pd.DataFrame()
    # Prefer R2 above 0.95; otherwise highest R2 while keeping blank-anchored range.
    elig = cand[(cand["R2"] >= 0.95) & (cand["n"] >= 8)]
    row = (elig.sort_values(["maxc","R2"], ascending=[False, False]).iloc[0] if not elig.empty else cand.sort_values(["R2","maxc"], ascending=[False, False]).iloc[0])
    feat = row["feature"]
    g = grouped.agg(mean=(feat,"mean"), sd=(feat,"std"), n=(feat,"count")).reindex(ALL_CONC).dropna()
    sub = g.loc[g.index <= int(row["maxc"])]
    x = sub.index.values.astype(float); y = sub["mean"].values.astype(float); yerr = sub["sd"].fillna(0).values.astype(float)
    slope, intercept = np.polyfit(x, y, 1); yp = slope*x + intercept
    r2lin = r2_score(y, yp); rmse = math.sqrt(mean_squared_error(y, yp)); mae = mean_absolute_error(y, yp)
    sd_blank = color_df[color_df["concentration_mM"] == 0][feat].std(ddof=1)
    lod = 3.3 * sd_blank / abs(slope) if np.isfinite(sd_blank) and slope != 0 else np.nan
    loq = 10.0 * sd_blank / abs(slope) if np.isfinite(sd_blank) and slope != 0 else np.nan
    # Nonlinear response on hue, because it is mechanistically interpretable.
    gh = grouped.agg(mean=("H_mean","mean"), sd=("H_mean","std")).reindex(ALL_CONC).dropna()
    xh = gh.index.values.astype(float); yh = gh["mean"].values.astype(float); yhsd = gh["sd"].fillna(0).values.astype(float)
    try:
        def sat(C, Hinf, A, k): return Hinf + A*np.exp(-k*C)
        popt, _ = curve_fit(sat, xh, yh, p0=[yh.min(), yh.max()-yh.min(), 0.08], bounds=([0,0,0.0001],[360,360,5]), maxfev=20000)
        xs_h = np.linspace(xh.min(), xh.max(), 300); yfit_h = sat(xs_h, *popt); yhhat = sat(xh, *popt)
        r2h = r2_score(yh, yhhat); rmseh = math.sqrt(mean_squared_error(yh, yhhat)); model_txt = f"H={popt[0]:.1f}+{popt[1]:.1f}e$^{{-{popt[2]:.3f}C}}$\nR²={r2h:.3f}; RMSE={rmseh:.2f}°"
    except Exception:
        coeff = np.polyfit(xh, yh, 2); xs_h = np.linspace(xh.min(), xh.max(), 300); yfit_h = np.poly1d(coeff)(xs_h); r2h=np.nan; rmseh=np.nan; model_txt="Nonlinear fit"
    # Plot
    fig, axes = plt.subplots(1,3,figsize=(14.2,4.4))
    xs = np.linspace(x.min(), x.max(), 300)
    axes[0].errorbar(x, y, yerr=yerr, fmt="o", capsize=3.5, label="Mean ± SD")
    axes[0].plot(xs, slope*xs + intercept, lw=2.1, label="Linear fit")
    axes[0].set_xlabel("Borax concentration (mM)"); axes[0].set_ylabel(feat.replace("_"," "))
    axes[0].set_title("Blank-anchored linear analytical range"); axes[0].grid(alpha=0.15); axes[0].legend(frameon=False)
    axes[0].text(0.04,0.08, f"Feature: {feat}\nRange = {row['range']} mM\ny = {slope:.3f}C + {intercept:.2f}\nR²={r2lin:.3f}; RMSE={rmse:.2f}\nLOD={lod:.3f} mM; LOQ={loq:.3f} mM", transform=axes[0].transAxes, fontsize=8.1, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", lw=0.7))
    add_panel_label(axes[0], "a")
    resid = y - yp
    axes[1].axhline(0, color="black", lw=0.9); axes[1].axhspan(-rmse, rmse, color="0.90", alpha=0.7)
    axes[1].vlines(x, 0, resid, color="0.5", lw=0.8); axes[1].scatter(x, resid, s=36, edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Borax concentration (mM)"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residual diagnostics"); axes[1].grid(alpha=0.15); add_panel_label(axes[1], "b")
    axes[2].errorbar(xh, yh, yerr=yhsd, fmt="o", capsize=3.3, label="Mean ± SD")
    axes[2].plot(xs_h, yfit_h, lw=2.1, label="Nonlinear fit")
    axes[2].set_xlabel("Borax concentration (mM)"); axes[2].set_ylabel("Hue angle (°)"); axes[2].set_title("Full-range nonlinear hue response"); axes[2].grid(alpha=0.15); axes[2].legend(frameon=False)
    axes[2].text(0.04,0.08, model_txt, transform=axes[2].transAxes, fontsize=8.1, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", lw=0.7)); add_panel_label(axes[2], "c")
    fig.suptitle("Analytical calibration: blank-anchored linear range and full-range nonlinear response", y=1.03)
    savefig(FIG_MAIN / "Fig06_analytical_calibration_dual_linear_nonlinear")
    out = pd.DataFrame([{"selected_linear_feature":feat,"linear_range_mM":row['range'],"slope":slope,"intercept":intercept,"R2":r2lin,"RMSE":rmse,"MAE":mae,"blank_sd":sd_blank,"LOD_mM":lod,"LOQ_mM":loq}])
    out.to_csv(TAB_MAIN / "Table3_analytical_figures_of_merit.csv", index=False)
    return out


def figure_8_ml_external(color_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """External validation shown as actual-vs-predicted for both independent directions."""
    perf_ext, pred_ext = external_validation(color_df)
    perf_int, pred_int = internal_loio_validation(color_df)
    if perf_ext.empty or pred_ext.empty:
        return perf_ext, pred_ext, perf_int, pred_int
    best = perf_ext.sort_values("RMSE_mM").groupby("validation", as_index=False).first()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    dirs = ["ganjil_to_genap", "genap_to_ganjil"]
    titles = ["Training set A → Test set B", "Training set B → Test set A"]
    for ax, direction, title in zip(axes, dirs, titles):
        if direction not in set(best["validation"]):
            ax.axis("off"); continue
        rr = best[best["validation"] == direction].iloc[0]
        p = pred_ext[(pred_ext["validation"] == direction) & (pred_ext["model"] == rr["model"])]
        ax.scatter(p["actual_mM"], p["predicted_mM"], s=30, edgecolor="black", linewidth=0.4)
        ax.plot([-1, 21], [-1, 21], ls="--", lw=1.1, color="black")
        ax.set_xlim(-1, 21); ax.set_ylim(-1, 21); ax.grid(alpha=0.15)
        ax.set_xlabel("Actual borax concentration (mM)"); ax.set_ylabel("Predicted borax concentration (mM)")
        ax.set_title(title)
        ax.text(0.05,0.95, f"{rr['model']}\nR²={rr['R2']:.3f}\nRMSE={rr['RMSE_mM']:.2f} mM\nMAE={rr['MAE_mM']:.2f} mM", transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", lw=0.7))
    add_panel_label(axes[0], "a"); add_panel_label(axes[1], "b")
    fig.suptitle("Independent external validation of smartphone-based borax prediction", y=1.03)
    savefig(FIG_MAIN / "Fig08_ML_external_validation")
    return perf_ext, pred_ext, perf_int, pred_int


def figure_9_explainability(color_df: pd.DataFrame) -> pd.DataFrame:
    """Main XAI figure: true SHAP beeswarm if available, otherwise robust permutation importance."""
    if color_df.empty:
        return pd.DataFrame()
    cols = feature_columns(color_df, include_ph=False)
    Xdf = color_df[cols].fillna(color_df[cols].median())
    X = Xdf.values; y = color_df["concentration_mM"].values.astype(float)
    try:
        import shap
        model = ExtraTreesRegressor(n_estimators=500, random_state=42, min_samples_leaf=1)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        shap_df = pd.DataFrame({"feature": cols, "mean_abs_SHAP": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_SHAP", ascending=False)
        shap_df.to_csv(TAB_SUPP / "TableS_SHAP_feature_importance.csv", index=False)
        # beeswarm style summary
        fig, ax = plt.subplots(figsize=(8.0, 5.6))
        top_feats = shap_df.head(12)["feature"].tolist()[::-1]
        rng = np.random.default_rng(42)
        for yi, feat in enumerate(top_feats):
            j = cols.index(feat)
            vals = sv[:, j]
            raw = Xdf[feat].values
            norm = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw) + 1e-9)
            jitter = rng.normal(0, 0.055, size=len(vals))
            ax.scatter(vals, yi + jitter, c=norm, cmap="viridis", s=15, edgecolor="none", alpha=0.82)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.set_yticks(range(len(top_feats))); ax.set_yticklabels(top_feats)
        ax.set_xlabel("SHAP value impact on predicted concentration (mM)")
        ax.set_title("SHAP summary of colorimetric feature contributions")
        sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0, vmax=1)); cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label("Feature value (low → high)")
        ax.grid(axis="x", alpha=0.15)
        savefig(FIG_MAIN / "Fig09_explainable_feature_importance")
        return shap_df
    except Exception:
        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]); model.fit(X, y)
        result = permutation_importance(model, X, y, n_repeats=50, random_state=42, scoring="neg_root_mean_squared_error")
        imp = pd.DataFrame({"feature": cols, "importance_mean": result.importances_mean, "importance_sd": result.importances_std}).sort_values("importance_mean", ascending=False)
        top = imp.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Permutation importance (increase in RMSE score)")
        ax.set_title("Model interpretability by permutation feature importance")
        ax.grid(axis="x", alpha=0.15)
        ax.text(0.98, 0.05, "Install shap for SHAP beeswarm\n(pip install shap)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2, bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.6", lw=0.65))
        savefig(FIG_MAIN / "Fig09_explainable_feature_importance")
        return imp


def figure_11_color_uvvis_correlation(color_df: pd.DataFrame, uv_features: pd.DataFrame) -> pd.DataFrame:
    """Cross-modal smartphone–UV-Vis agreement with clean right-side annotations.

    Final Q1 revision:
    - Statistical boxes are placed at the right side of each panel to avoid the
      low-concentration points that occupy the left side.
    - 95% confidence bands, Pearson r, Spearman rho, R², and p-values are shown.
    - The colorbar is compact and placed outside the third panel.
    """
    from scipy.stats import pearsonr, spearmanr, t as student_t_local

    if color_df.empty or uv_features.empty:
        return pd.DataFrame()

    color_agg = color_df.groupby("concentration_mM").agg(
        H_mean=("H_mean", "mean"),
        Excess_Blue_Index=("Excess_Blue_Index", "mean"),
        Lab_b_mean=("Lab_b_mean", "mean")
    ).reset_index()
    uv_agg = uv_features.groupby("concentration_mM").agg(
        lambda_max_nm=("lambda_max_nm", "mean"),
        AUC=("AUC", "mean"),
        Amax=("Amax", "mean")
    ).reset_index()
    merged = pd.merge(color_agg, uv_agg, on="concentration_mM", how="inner").dropna()
    if merged.empty:
        return pd.DataFrame()

    pairs = [
        ("H_mean", "lambda_max_nm", "Hue angle (°)", "λmax (nm)"),
        ("Excess_Blue_Index", "AUC", "Excess blue index", "Integrated absorbance (AUC)"),
        ("Lab_b_mean", "Amax", "CIELab b*", "Peak absorbance, Amax"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), gridspec_kw={"wspace": 0.28})
    stats_rows = []
    last_sc = None

    for ax, (xcol, ycol, xlabel, ylabel), label in zip(axes, pairs, ["a", "b", "c"]):
        x = merged[xcol].values.astype(float)
        y = merged[ycol].values.astype(float)
        c = merged["concentration_mM"].values.astype(float)
        last_sc = ax.scatter(x, y, c=c, cmap="viridis", s=52, edgecolor="black", linewidth=0.50, zorder=3)

        # Linear fit + confidence band
        if len(x) >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 250)
            ys = slope * xs + intercept
            yhat = slope * x + intercept
            dof = max(len(x) - 2, 1)
            s_err = np.sqrt(np.sum((y - yhat) ** 2) / dof)
            xbar = np.mean(x); sxx = np.sum((x - xbar) ** 2)
            if sxx > 0:
                tval = student_t_local.ppf(0.975, dof)
                ci = tval * s_err * np.sqrt(1.0 / len(x) + (xs - xbar) ** 2 / sxx)
                ax.fill_between(xs, ys - ci, ys + ci, color="0.35", alpha=0.13, lw=0, zorder=1)
            ax.plot(xs, ys, color="0.20", lw=1.8, zorder=2)
            r, p = pearsonr(x, y)
            rho, ps = spearmanr(x, y)
            stats_rows.append({
                "x_feature": xcol, "y_feature": ycol,
                "pearson_r": r, "pearson_p": p,
                "spearman_rho": rho, "spearman_p": ps,
                "R2": r*r,
            })
            # Put annotation on the right side to avoid covering the dense data on left.
            ax.text(0.96, 0.95,
                    f"r = {r:.3f}\nρ = {rho:.3f}\nR² = {r*r:.3f}\np = {p:.1e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.0,
                    bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.55", lw=0.7, alpha=0.92),
                    zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.14, lw=0.55)
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold")

    if last_sc is not None:
        cb = fig.colorbar(last_sc, ax=axes, fraction=0.025, pad=0.018)
        cb.set_label("Borax concentration (mM)")
        cb.ax.tick_params(labelsize=8)

    fig.suptitle("Cross-modal agreement between smartphone colorimetry and UV–Vis spectroscopy",
                 y=1.02, fontsize=14.0)
    savefig(FIG_MAIN / "Fig11_color_UVVis_correlation")
    out = pd.DataFrame(stats_rows)
    if not out.empty:
        out.to_csv(TAB_SUPP / "TableS_color_UVVis_cross_modal_statistics.csv", index=False)
    return out


def main() -> None:
    set_premium_style()
    print("============================================================")
    print("BORAX Q1 PREMIUM FIGURE + TABLE GENERATOR")
    print("============================================================")
    ganjil_folder = first_existing_folder(PHOTO_GANJIL_CANDIDATES)
    genap_folder = first_existing_folder(PHOTO_GENAP_CANDIDATES)
    uv_ganjil_folder = first_existing_folder(UVVIS_GANJIL_CANDIDATES)
    uv_genap_folder = first_existing_folder(UVVIS_GENAP_CANDIDATES)
    print(f"Photo ganjil folder : {ganjil_folder}")
    print(f"Photo genap folder  : {genap_folder}")
    print(f"UV-Vis ganjil folder: {uv_ganjil_folder}")
    print(f"UV-Vis genap folder : {uv_genap_folder}")

    manual = load_manual_roi_override(MANUAL_ROI_OVERRIDE_CSV)
    if manual:
        print(f"Manual ROI override loaded: {len(manual)} ROI entries")

    print("\n[1/8] Extracting smartphone color features...")
    out_g = extract_color_features_from_set("ganjil", ganjil_folder, GANJIL_CONC, manual)
    out_e = extract_color_features_from_set("genap", genap_folder, GENAP_CONC, manual)
    dfs = []; rois = []
    for out in [out_g, out_e]:
        if isinstance(out, tuple):
            df, rq = out
            if not df.empty: dfs.append(df)
            if not rq.empty: rois.append(rq)
    color_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    roi_qc = pd.concat(rois, ignore_index=True) if rois else pd.DataFrame()
    if not color_df.empty:
        color_df.to_csv(PROC_DIR / "processed_color_features.csv", index=False)
        aggregate_color(color_df).to_csv(PROC_DIR / "concentration_level_color_summary.csv", index=False)
        print(f"Extracted {len(color_df)} ROI records from {color_df['image_file'].nunique()} images.")
    if not roi_qc.empty:
        roi_qc.to_csv(TAB_SUPP / "TableS_ROI_coordinates_and_QC.csv", index=False)

    print("\n[2/8] Loading and processing UV-Vis data...")
    uv_raw = pd.concat([load_uvvis_dataset(uv_ganjil_folder, "ganjil"), load_uvvis_dataset(uv_genap_folder, "genap")], ignore_index=True)
    uv_spectra, uv_features = process_uvvis(uv_raw) if not uv_raw.empty else (pd.DataFrame(), pd.DataFrame())
    if not uv_spectra.empty:
        uv_spectra.to_csv(PROC_DIR / "uvvis_spectra_interpolated.csv", index=False)
        uv_features.to_csv(PROC_DIR / "uvvis_features.csv", index=False)
        print(f"Processed UV-Vis spectra: {uv_features.shape[0]} files/features.")
    else:
        print("UV-Vis data not found/readable. UV-Vis figures will be skipped.")

    print("\n[3/8] Generating main figures...")
    figure_1_workflow_graphical(color_df, uv_spectra)
    # Additional journal-style graphical abstract generated with the final booktabs/table patch.
    try:
        graphical_abstract_q1(color_df, uv_spectra)
    except NameError:
        pass
    figure_0_molecular_mechanism(color_df)
    figure_2_roi_and_color_gallery(color_df)
    figure_3_ph_hue_lab(color_df)
    figure_4_uvvis_top_tier(uv_spectra)
    figure_5_uvvis_features(uv_features)
    analytical = figure_6_analytical_calibration(color_df)
    pca_loadings, uv_corr = figure_7_chemometrics(color_df, uv_features)
    perf_ext, pred_ext, perf_int, pred_int = figure_8_ml_external(color_df)
    imp = figure_9_explainability(color_df)
    uv_bridge = figure_11_color_uvvis_correlation(color_df, uv_features)
    perm = figure_10_bland_altman_and_permutation(color_df, pred_ext, perf_ext)

    print("\n[4/8] Generating supplementary figures...")
    supplementary_figures(color_df, uv_features, perf_int, pred_int, pca_loadings, pred_ext, perf_ext)

    print("\n[5/8] Saving prediction and model tables...")
    if not perf_ext.empty:
        perf_ext.to_csv(PROC_DIR / "external_validation_performance_all_models.csv", index=False)
    if not pred_ext.empty:
        pred_ext.to_csv(PROC_DIR / "external_validation_predictions_all_models.csv", index=False)
    if not perf_int.empty:
        perf_int.to_csv(PROC_DIR / "internal_LOIO_performance_all_models.csv", index=False)
    if not pred_int.empty:
        pred_int.to_csv(PROC_DIR / "internal_LOIO_predictions_all_models.csv", index=False)
    if not imp.empty:
        imp.to_csv(TAB_SUPP / "TableS_feature_importance_permutation_Ridge.csv", index=False)
    if not perm.empty:
        perm.to_csv(TAB_SUPP / "TableS_permutation_test_R2_distribution.csv", index=False)

    print("\n[6/8] Generating publication-style tables...")
    create_main_and_supplementary_tables(color_df, roi_qc, uv_features, analytical, perf_ext, perf_int, pca_loadings, uv_corr)

    print("\n[7/8] Writing figure/table map...")
    figure_map = pd.DataFrame([
        ["Fig. 1", "Two-row workflow schematic", "figures_main/Fig01_workflow_schematic"],
        ["Fig. 2", "Representative ROI and color gallery", "figures_main/Fig02_representative_ROI_and_color_gallery"],
        ["Fig. 3", "pH, hue, and CIELab mechanism", "figures_main/Fig03_pH_hue_Lab_mechanism"],
        ["Fig. 4", "UV-Vis spectra, heatmap, and difference spectra", "figures_main/Fig04_UVVis_selected_heatmap_difference"],
        ["Fig. 5", "Robust UV-Vis descriptors", "figures_main/Fig05_UVVis_robust_spectral_descriptors"],
        ["Fig. 6", "Analytical calibration and LOD/LOQ", "figures_main/Fig06_analytical_calibration_LOD_LOQ"],
        ["Fig. 7", "PCA and multimodal correlation", "figures_main/Fig07_PCA_and_multimodal_correlation"],
        ["Fig. 8", "ML external validation", "figures_main/Fig08_ML_external_validation"],
        ["Fig. 9", "Explainable feature importance", "figures_main/Fig09_explainable_feature_importance"],
        ["Fig. 10", "Bland-Altman and permutation test", "figures_main/Fig10_BlandAltman_and_permutation_test"],
        ["Fig. 11", "Smartphone color and UV-Vis cross-modal correlation", "figures_main/Fig11_color_UVVis_correlation"],
    ], columns=["Item", "Description", "File base"])
    figure_map.to_csv(OUTPUT / "recommended_main_figure_map.csv", index=False)
    supp_map = pd.DataFrame([
        ["Fig. S1", "ROI extraction quality control", "figures_supplementary/FigS01_ROI_quality_control"],
        ["Fig. S2", "Repeatability and precision", "figures_supplementary/FigS02_repeatability_precision"],
        ["Fig. S3", "DeltaE from blank", "figures_supplementary/FigS03_DeltaE_from_blank"],
        ["Fig. S4", "Feature trends", "figures_supplementary/FigS04_feature_trends"],
        ["Fig. S5", "Internal LOIO model comparison", "figures_supplementary/FigS05_internal_LOIO_model_comparison"],
        ["Fig. S6", "Internal LOIO predicted vs actual", "figures_supplementary/FigS06_internal_LOIO_predicted_vs_actual"],
        ["Fig. S7", "PCA loadings", "figures_supplementary/FigS07_PCA_loadings"],
        ["Fig. S8", "Color-UVVis correlations", "figures_supplementary/FigS08_color_UVVis_correlation"],
        ["Fig. S9", "SHAP feature importance if shap installed", "figures_supplementary/FigS09_SHAP_feature_importance"],
        ["Fig. S10", "Hierarchical clustering heatmap", "figures_supplementary/FigS10_hierarchical_clustering_heatmap"],
    ], columns=["Item", "Description", "File base"])
    supp_map.to_csv(OUTPUT / "recommended_supplementary_figure_map.csv", index=False)

    print("\n[8/8] Writing README...")
    readme = f"""
BORAX Q1 PREMIUM FIGURE AND TABLE OUTPUT
========================================

Output folder:
{OUTPUT}

Main figure folder:
{FIG_MAIN}

Supplementary figure folder:
{FIG_SUPP}

Main table folder:
{TAB_MAIN}

Supplementary table folder:
{TAB_SUPP}

Recommended workflow:
1. Inspect all ROI overlays in roi_overlays/.
2. If any ROI is off-center, edit tables_supplementary/TableS_ROI_coordinates_and_QC.csv and save a copy as manual_roi_override.csv beside this script.
3. Rerun the script.
4. Use PDF/TIFF files for journal submission and PNG for manuscript drafting.

Important manuscript note:
The script saves both visual figures and the numerical tables behind each figure, improving reproducibility and reviewer confidence.
"""
    (OUTPUT / "README_Q1_premium_outputs.txt").write_text(readme.strip(), encoding="utf-8")
    print("\nDone.")
    print(f"Output folder: {OUTPUT}")



# ============================================================================
# FINAL VISUAL REVISION OVERRIDES FOR FIGURE 1 AND FIGURE 2
# These definitions intentionally override earlier versions above.  They keep
# all downstream analysis unchanged and only improve journal-quality layout.
# ============================================================================


def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Final Q1-premium Figure 1.

    Design principles:
    1. Two-row workflow with large whitespace; no touching boxes.
    2. Chemistry validation is visually separated from digital quantification.
    3. Lower row contains data-derived evidence: color scale, UV-Vis spectra,
       and independent prediction/interpretation block.
    4. Avoids PowerPoint-like cramped long horizontal flowcharts.
    """
    fig = plt.figure(figsize=(16.4, 8.8))
    gs = GridSpec(
        4, 12, figure=fig,
        height_ratios=[0.16, 1.26, 0.08, 1.32],
        hspace=0.45, wspace=0.72
    )

    # ------------------------------------------------------------------
    # Top graphical workflow
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    chem_steps = [
        ("Borax\nstandard", "alkalization", "#EAF4FF"),
        ("Purple cabbage\nanthocyanin", "natural pH probe", "#F3EAFB"),
        ("Anthocyanin\nconversion", "pH-responsive forms", "#F7F7F7"),
        ("UV–Vis\nvalidation", "spectral fingerprint", "#EEF8F2"),
    ]
    digi_steps = [
        ("Smartphone\nimaging", "controlled acquisition", "#FFF2DF"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", "#ECF6FD"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", "#F5F5F5"),
        ("Borax\nprediction", "quantitative output", "#EFF9EF"),
    ]

    def row_header(y, text, color):
        ax.add_patch(FancyBboxPatch((0.025, y - 0.105), 0.095, 0.21,
                                    boxstyle="round,pad=0.012,rounding_size=0.020",
                                    fc=color, ec="0.35", lw=0.85))
        ax.text(0.0725, y, text, ha="center", va="center", fontsize=9.1,
                fontweight="bold", rotation=90, linespacing=0.88, color="0.14")

    def draw_steps(steps, y, start_no):
        # Deliberate large gaps so boxes never visually merge.
        x0, x1 = 0.155, 0.940
        n = len(steps)
        gap = 0.055
        bw = (x1 - x0 - gap*(n-1))/n
        bh = 0.185
        cy = y
        for i, (title, subtitle, fc) in enumerate(steps):
            x = x0 + i*(bw + gap)
            # subtle shadow
            ax.add_patch(FancyBboxPatch((x+0.006, cy-bh/2-0.006), bw, bh,
                                        boxstyle="round,pad=0.020,rounding_size=0.022",
                                        fc="0.75", ec="none", alpha=0.13, zorder=0))
            ax.add_patch(FancyBboxPatch((x, cy-bh/2), bw, bh,
                                        boxstyle="round,pad=0.020,rounding_size=0.022",
                                        fc=fc, ec="0.12", lw=0.95, zorder=1))
            # numbered dot outside/above left of each box: clearer, not on text.
            dotx, doty = x + 0.028, cy + bh/2 - 0.022
            ax.add_patch(Circle((dotx, doty), 0.016, fc="0.13", ec="none", zorder=3))
            ax.text(dotx, doty, str(start_no+i), ha="center", va="center",
                    fontsize=7.2, color="white", fontweight="bold", zorder=4)
            ax.text(x + bw/2, cy + 0.026, title, ha="center", va="center",
                    fontsize=10.0, fontweight="bold", linespacing=0.82, zorder=3)
            ax.text(x + bw/2, cy - 0.061, subtitle, ha="center", va="center",
                    fontsize=7.9, color="0.28", zorder=3)
            if i < n-1:
                ax.add_patch(FancyArrowPatch((x + bw + 0.012, cy),
                                             (x + bw + gap - 0.012, cy),
                                             arrowstyle="-|>", mutation_scale=14,
                                             lw=1.25, color="0.22", zorder=2))

    row_header(0.70, "Chemistry\nvalidation", "#F6FAFF")
    row_header(0.37, "Digital\nquantification", "#FFF9EF")
    draw_steps(chem_steps, 0.70, 1)
    draw_steps(digi_steps, 0.37, 5)

    # separator and central statement, with no causal arrow from UV-Vis to smartphone.
    ax.plot([0.145, 0.945], [0.535, 0.535], color="0.70", lw=0.9, ls=(0, (3.5, 3.5)))
    ax.text(0.545, 0.535, "independent spectral validation and smartphone-based quantification",
            ha="center", va="bottom", fontsize=8.0, color="0.34")
    ax.text(0.5, 0.085,
            "Borax-induced alkalization converts anthocyanin chemistry into UV–Vis-validated smartphone color signatures for explainable quantitative prediction",
            ha="center", va="center", fontsize=10.6, fontweight="bold")

    # ------------------------------------------------------------------
    # Data-evidence panels
    # ------------------------------------------------------------------
    ax_sw = fig.add_subplot(gs[3, 0:3])
    ax_uv = fig.add_subplot(gs[3, 3:8])
    ax_ml = fig.add_subplot(gs[3, 8:12])

    ax_sw.set_title("Data-derived smartphone color scale", fontsize=10.6, pad=7)
    _draw_mean_color_strip(ax_sw, color_df, ALL_CONC, label_every=5)
    ax_sw.text(0.5, -0.27, "Borax concentration (mM)", transform=ax_sw.transAxes,
               ha="center", va="top", fontsize=8.6)
    ax_sw.text(-0.10, 1.10, "a", transform=ax_sw.transAxes, fontsize=12, fontweight="bold")

    ax_uv.set_title("Representative UV–Vis fingerprints", fontsize=10.6, pad=7)
    avg = _get_avg_uvvis(uv_spectra)
    if not avg.empty:
        for c in [0, 5, 10, 15, 20]:
            if c in set(avg["concentration_mM"].astype(float)):
                g = avg[np.isclose(avg["concentration_mM"].astype(float), float(c))]
                ax_uv.plot(g["wavelength_nm"], g["absorbance"], lw=1.55, label=f"{c} mM")
        ax_uv.set_xlabel("Wavelength (nm)")
        ax_uv.set_ylabel("Absorbance")
        ax_uv.legend(frameon=False, fontsize=7.7, loc="upper right")
        ax_uv.grid(alpha=0.12, lw=0.55)
    else:
        ax_uv.text(0.5, 0.5, "UV–Vis data not available", ha="center", va="center")
        ax_uv.axis("off")
    ax_uv.text(-0.08, 1.10, "b", transform=ax_uv.transAxes, fontsize=12, fontweight="bold")

    ax_ml.set_title("Independent prediction and interpretation", fontsize=10.6, pad=7)
    ax_ml.set_xlim(0, 1); ax_ml.set_ylim(0, 1); ax_ml.axis("off")
    nodes = [
        ("Color feature matrix", "RGB, HSV, CIELab"),
        ("Independent validation", "Set A ↔ Set B"),
        ("Feature attribution", "Permutation / SHAP"),
        ("Borax concentration", "mM output"),
    ]
    ys = [0.80, 0.58, 0.36, 0.14]
    for (title, sub), y in zip(nodes, ys):
        ax_ml.add_patch(FancyBboxPatch((0.14, y - 0.060), 0.72, 0.105,
                                       boxstyle="round,pad=0.022,rounding_size=0.016",
                                       fc="white", ec="0.30", lw=0.85))
        ax_ml.text(0.50, y + 0.013, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax_ml.text(0.50, y - 0.030, sub, ha="center", va="center", fontsize=7.2, color="0.33")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax_ml.add_patch(FancyArrowPatch((0.5, y0 - 0.071), (0.5, y1 + 0.071),
                                        arrowstyle="->", mutation_scale=11, lw=0.9, color="0.25"))
    ax_ml.text(-0.08, 1.10, "c", transform=ax_ml.transAxes, fontsize=12, fontweight="bold")

    fig.suptitle("Multimodal smartphone colorimetry workflow for anthocyanin-assisted borax quantification",
                 y=0.982, fontsize=14.0, fontweight="bold")
    savefig(FIG_MAIN / "Fig01_workflow_schematic")


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Final Q1-premium Figure 2.

    Design principles:
    1. Panel a uses a clean publication-style molecular pathway without cramped
       pseudo-structure labels.
    2. Panel b contains pH-hue coupling with nonlinear fit, 95% CI, and concise
       statistics.
    3. Panel c shows a color trajectory instead of a dense table, with only
       selected concentration labels.
    """
    from matplotlib.patches import RegularPolygon
    from scipy.stats import pearsonr, spearmanr, t as student_t_local

    fig = plt.figure(figsize=(15.6, 5.6))
    gs = GridSpec(1, 3, width_ratios=[1.55, 1.08, 0.98], figure=fig, wspace=0.34)

    # ------------------------------------------------------------------
    # Panel a: cleaner molecular transformation schematic
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(-0.02, 1.02, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax.set_title("pH-driven anthocyanin structural transformation", fontsize=11.3, fontweight="bold", pad=10)

    # pH arrow kept compact and above the structures.
    ax.annotate("increasing borax concentration / pH", xy=(0.84, 0.82), xytext=(0.22, 0.82),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.25"),
                ha="center", va="center", fontsize=9.2, fontweight="bold")

    def anthocyanin_icon(cx, cy, col, label, form, note, charge=""):
        # simplified yet more chemical-looking flavylium/flavonoid scaffold
        r = 0.044
        # fused rings A/C
        hex1 = RegularPolygon((cx-0.065, cy), 6, radius=r, orientation=np.pi/6, fc=col, ec="0.13", lw=1.0, alpha=0.95)
        hex2 = RegularPolygon((cx-0.005, cy), 6, radius=r, orientation=np.pi/6, fc=col, ec="0.13", lw=1.0, alpha=0.92)
        # B ring
        hex3 = RegularPolygon((cx+0.085, cy+0.012), 6, radius=r*0.88, orientation=np.pi/6, fc=col, ec="0.13", lw=1.0, alpha=0.92)
        ax.add_patch(hex1); ax.add_patch(hex2); ax.add_patch(hex3)
        ax.plot([cx+0.035, cx+0.052], [cy+0.007, cy+0.010], color="0.13", lw=1.0)
        ax.text(cx-0.005, cy+0.002, "O"+charge, ha="center", va="center", fontsize=8.0, fontweight="bold")
        ax.text(cx-0.090, cy+0.073, "OH", ha="center", va="center", fontsize=6.5)
        ax.text(cx+0.084, cy+0.073, "OH", ha="center", va="center", fontsize=6.5)
        # form labels
        ax.add_patch(FancyBboxPatch((cx-0.145, cy-0.160), 0.29, 0.055,
                                    boxstyle="round,pad=0.012,rounding_size=0.010",
                                    fc="white", ec="0.55", lw=0.65))
        ax.text(cx, cy-0.132, label, ha="center", va="center", fontsize=8.0, fontweight="bold")
        ax.text(cx, cy-0.193, form, ha="center", va="center", fontsize=7.2, color="0.30")
        ax.text(cx, cy-0.243, note, ha="center", va="center", fontsize=6.6, color="0.38")

    def chalcone_icon(cx, cy, col, label, form, note):
        r = 0.044
        hex1 = RegularPolygon((cx-0.095, cy), 6, radius=r, orientation=np.pi/6, fc=col, ec="0.13", lw=1.0, alpha=0.95)
        hex2 = RegularPolygon((cx+0.105, cy+0.010), 6, radius=r, orientation=np.pi/6, fc=col, ec="0.13", lw=1.0, alpha=0.92)
        ax.add_patch(hex1); ax.add_patch(hex2)
        ax.plot([cx-0.042, cx+0.035], [cy+0.003, cy+0.010], color="0.13", lw=1.1)
        ax.plot([cx-0.020, cx+0.052], [cy+0.027, cy+0.035], color="0.13", lw=0.9)
        ax.text(cx+0.008, cy+0.075, "C=C–C=O", ha="center", va="center", fontsize=6.8)
        ax.text(cx-0.130, cy+0.074, "OH", ha="center", va="center", fontsize=6.5)
        ax.add_patch(FancyBboxPatch((cx-0.145, cy-0.160), 0.29, 0.055,
                                    boxstyle="round,pad=0.012,rounding_size=0.010",
                                    fc="white", ec="0.55", lw=0.65))
        ax.text(cx, cy-0.132, label, ha="center", va="center", fontsize=8.0, fontweight="bold")
        ax.text(cx, cy-0.193, form, ha="center", va="center", fontsize=7.2, color="0.30")
        ax.text(cx, cy-0.243, note, ha="center", va="center", fontsize=6.6, color="0.38")

    xs = [0.20, 0.50, 0.80]
    cy = 0.50
    anthocyanin_icon(xs[0], cy, "#7C519B", "Flavylium cation", "purple", "dominant at lower pH", charge="⁺")
    anthocyanin_icon(xs[1], cy, "#3568B7", "Quinoidal base", "blue", "deprotonated chromophore")
    chalcone_icon(xs[2], cy, "#2F806F", "Anionic/chalcone-like", "green-blue", "ring-opening equilibrium")
    for x0, x1 in [(0.31, 0.39), (0.61, 0.69)]:
        ax.add_patch(FancyArrowPatch((x0, cy+0.040), (x1, cy+0.040), arrowstyle="-|>", mutation_scale=12, lw=1.05, color="0.18"))
        ax.add_patch(FancyArrowPatch((x1, cy-0.040), (x0, cy-0.040), arrowstyle="-|>", mutation_scale=12, lw=1.05, color="0.18"))
    ax.text(0.5, 0.075, "Simplified anthocyanin-species schematic; substituent pattern depends on the natural extract composition.",
            ha="center", va="center", fontsize=7.3, color="0.36")

    # ------------------------------------------------------------------
    # Panel b: pH-hue coupling
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(-0.18, 1.02, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax2.set_title("pH–hue mechanistic coupling", fontsize=11.3, fontweight="bold", pad=10)
    if not color_df.empty and {"pH", "H_mean", "concentration_mM"}.issubset(color_df.columns):
        agg = color_df.groupby("concentration_mM").agg(
            H=("H_mean", "mean"), Hsd=("H_mean", "std"), pH=("pH", "mean")
        ).reindex(ALL_CONC).dropna()
        if "pH" not in agg.columns:
            agg["pH"] = agg["concentration_mM"].round().astype(int).map(PH_MAP).astype(float)
        x = agg["pH"].values.astype(float)
        y = agg["H"].values.astype(float)
        yerr = agg["Hsd"].fillna(0).values.astype(float)
        conc = agg.index.values.astype(float)
        sc = ax2.scatter(x, y, c=conc, cmap="viridis", s=42, edgecolor="black", linewidth=0.45, zorder=3)
        ax2.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.38", elinewidth=0.75, capsize=2.2, zorder=2)
        if len(x) >= 5 and np.nanstd(x) > 0:
            deg = 2 if len(x) >= 6 else 1
            coeff = np.polyfit(x, y, deg)
            poly = np.poly1d(coeff)
            xs_fit = np.linspace(np.nanmin(x), np.nanmax(x), 250)
            yhat = poly(x)
            ys_fit = poly(xs_fit)
            dof = max(len(x) - (deg + 1), 1)
            s_err = np.sqrt(np.sum((y - yhat) ** 2) / dof)
            xbar = np.mean(x); sxx = np.sum((x - xbar) ** 2)
            if sxx > 0:
                tval = student_t_local.ppf(0.975, dof)
                ci = tval * s_err * np.sqrt(1.0 / len(x) + (xs_fit - xbar) ** 2 / sxx)
                ax2.fill_between(xs_fit, ys_fit - ci, ys_fit + ci, color="0.20", alpha=0.12, lw=0, zorder=1)
            ax2.plot(xs_fit, ys_fit, color="0.16", lw=2.0, zorder=2)
            r, p = pearsonr(x, y); rho, ps = spearmanr(x, y)
            ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
            ax2.text(0.96, 0.96, f"r = {r:.3f}\nρ = {rho:.3f}\nR² = {r*r:.3f}\n{ptxt}",
                     transform=ax2.transAxes, ha="right", va="top", fontsize=8.2,
                     bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.55", lw=0.7, alpha=0.92))
        for ci_, xi, yi in zip(conc, x, y):
            if ci_ in [0, 5, 10, 15, 20]:
                ax2.annotate(str(int(ci_)), (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=7.6)
        cb = fig.colorbar(sc, ax=ax2, fraction=0.026, pad=0.014)
        cb.set_label("mM", fontsize=8.0)
        cb.ax.tick_params(labelsize=7.4)
    ax2.set_xlabel("Measured pH")
    ax2.set_ylabel("HSV hue angle (°)")
    ax2.grid(alpha=0.13, lw=0.55)

    # ------------------------------------------------------------------
    # Panel c: color trajectory, not dense table
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(-0.16, 1.02, "c", transform=ax3.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax3.set_title("Smartphone color trajectory", fontsize=11.3, fontweight="bold", pad=10)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    if not color_df.empty:
        means = color_df.groupby("concentration_mM")[["R_mean", "G_mean", "B_mean", "H_mean"]].mean().reindex(ALL_CONC).dropna()
        n = len(means)
        # large vertical color trajectory
        x0, w = 0.22, 0.28
        y0, h = 0.14, 0.74
        if n > 0:
            for i, (conc, row) in enumerate(means.iterrows()):
                y = y0 + h * (1 - (i+1)/n)
                ax3.add_patch(Rectangle((x0, y), w, h/n,
                                        fc=np.clip(row[["R_mean", "G_mean", "B_mean"]].values/255.0, 0, 1),
                                        ec="white", lw=0.35))
            for c in [0, 5, 10, 15, 20]:
                if c in means.index:
                    i = list(means.index).index(c)
                    y = y0 + h * (1 - (i + 0.5)/n)
                    ax3.text(x0 - 0.05, y, f"{int(c)}", ha="right", va="center", fontsize=8.2, fontweight="bold")
                    ax3.plot([x0 - 0.035, x0 - 0.006], [y, y], color="0.25", lw=0.7)
                    ax3.text(x0 + w + 0.055, y, f"H = {means.loc[c, 'H_mean']:.1f}°", ha="left", va="center", fontsize=8.0)
            ax3.text(x0 - 0.05, y0 + h + 0.035, "mM", ha="right", va="center", fontsize=8.2, fontweight="bold")
            ax3.text(x0 + w/2, y0 - 0.055, "mean ROI color", ha="center", va="top", fontsize=8.2)
            ax3.annotate("increasing\nborax", xy=(x0 + w + 0.19, y0 + 0.08), xytext=(x0 + w + 0.19, y0 + h - 0.08),
                         arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.28"),
                         ha="center", va="center", fontsize=8.0, color="0.25")
            ax3.text(0.5, 0.045, "Color transition extracted directly from smartphone ROIs", ha="center", va="center", fontsize=7.3, color="0.35")

    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 y=0.990, fontsize=13.8, fontweight="bold")
    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale")



# =============================================================================
# FINAL CLEAN VISUAL OVERRIDES FOR FIGURE 1 AND FIGURE 2
# These definitions intentionally override earlier auto-generated versions.
# Goal: cleaner Q1-style layout, no crowded boxes, no overlapping panel labels.
# =============================================================================

def _rgb01_from_row(row):
    return np.clip(np.array([row["R_mean"], row["G_mean"], row["B_mean"]], dtype=float) / 255.0, 0, 1)


def _draw_round_box(ax, xy, wh, title, subtitle="", fc="#F7F7F7", ec="0.18", lw=1.0,
                    title_size=10.2, subtitle_size=7.2):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.012,rounding_size=0.030",
                           fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", linespacing=0.90)
    if subtitle:
        ax.text(x + w/2, y + h*0.25, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color="0.34", linespacing=0.92)
    return patch


def _draw_arrow(ax, p1, p2, color="0.22", lw=1.15):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, color=color, shrinkA=2, shrinkB=2,
                                 connectionstyle="arc3,rad=0"))


def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Clean Q1-style workflow with two separated layers and larger data evidence panels."""
    fig = plt.figure(figsize=(14.8, 7.2), constrained_layout=False)
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.15, 0.18, 1.65],
                  hspace=0.34, wspace=0.38)

    # ------------------------------------------------------------------
    # Top workflow axis: simple, spacious, two layers
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fig.suptitle("Multimodal smartphone colorimetry for anthocyanin-assisted borax quantification",
                 y=0.975, fontsize=15.2, fontweight="bold")

    # Layer labels
    ax.text(0.045, 0.72, "Chemistry\nvalidation", rotation=90, ha="center", va="center",
            fontsize=8.3, fontweight="bold", color="0.18")
    ax.text(0.045, 0.30, "Digital\nquantification", rotation=90, ha="center", va="center",
            fontsize=8.3, fontweight="bold", color="0.18")
    ax.plot([0.09, 0.94], [0.50, 0.50], color="0.78", lw=0.8, ls=(0, (3, 3)))

    top = [
        ("Borax\nstandard", "alkalization", "#E7F0FA"),
        ("Purple cabbage\nanthocyanin", "natural pH probe", "#F0E7F8"),
        ("Anthocyanin\nconversion", "pH-responsive forms", "#F7F7F7"),
        ("UV–Vis\nvalidation", "spectral fingerprint", "#E9F6EF"),
    ]
    bottom = [
        ("Smartphone\nimaging", "controlled acquisition", "#FFF2DF"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", "#EAF4FB"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", "#F3F3F3"),
        ("Borax\nprediction", "quantitative mM output", "#EAF8EC"),
    ]

    x0, w, gap = 0.10, 0.17, 0.035
    y_top, y_bot, h = 0.62, 0.20, 0.22
    for i, (title, sub, fc) in enumerate(top):
        x = x0 + i*(w+gap)
        _draw_round_box(ax, (x, y_top), (w, h), title, sub, fc=fc)
        ax.add_patch(Circle((x+0.020, y_top+h-0.030), 0.014, fc="0.12", ec="none"))
        ax.text(x+0.020, y_top+h-0.030, str(i+1), ha="center", va="center",
                color="white", fontsize=6.6, fontweight="bold")
        if i < 3:
            _draw_arrow(ax, (x+w+0.006, y_top+h/2), (x+w+gap-0.010, y_top+h/2))

    for i, (title, sub, fc) in enumerate(bottom):
        x = x0 + i*(w+gap)
        _draw_round_box(ax, (x, y_bot), (w, h), title, sub, fc=fc)
        ax.add_patch(Circle((x+0.020, y_bot+h-0.030), 0.014, fc="0.12", ec="none"))
        ax.text(x+0.020, y_bot+h-0.030, str(i+5), ha="center", va="center",
                color="white", fontsize=6.6, fontweight="bold")
        if i < 3:
            _draw_arrow(ax, (x+w+0.006, y_bot+h/2), (x+w+gap-0.010, y_bot+h/2))

    # Independent validation annotation instead of confusing arrow
    ax.text(0.50, 0.515, "independent chemical validation supports digital quantification",
            ha="center", va="bottom", fontsize=7.4, color="0.40",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.90))
    ax.text(0.50, 0.075,
            "Borax-induced alkalization converts anthocyanin chemistry into UV–Vis-validated smartphone color signatures for explainable quantitative prediction",
            ha="center", va="center", fontsize=10.5, fontweight="bold")

    # ------------------------------------------------------------------
    # Bottom evidence panels: color scale, UV-Vis, prediction pipeline
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[2, 0:3])
    ax2 = fig.add_subplot(gs[2, 3:8])
    ax3 = fig.add_subplot(gs[2, 8:12])
    for lab, axp in zip(["a", "b", "c"], [ax1, ax2, ax3]):
        axp.text(-0.12, 1.06, lab, transform=axp.transAxes, fontsize=12,
                 fontweight="bold", va="bottom")

    # Panel a: thicker color strip
    ax1.set_title("Data-derived smartphone color scale", fontsize=9.8, pad=8)
    ax1.set_xlim(0, 21); ax1.set_ylim(0, 1); ax1.axis("off")
    if not color_df.empty:
        means = color_df.groupby("concentration_mM")[["R_mean", "G_mean", "B_mean"]].mean().reindex(ALL_CONC).dropna()
        for c, row in means.iterrows():
            ax1.add_patch(Rectangle((float(c), 0.22), 0.96, 0.58,
                                    fc=_rgb01_from_row(row), ec="white", lw=0.35))
    ax1.text(0, 0.08, "0", fontsize=8, ha="center")
    ax1.text(5, 0.08, "5", fontsize=8, ha="center")
    ax1.text(10, 0.08, "10", fontsize=8, ha="center")
    ax1.text(15, 0.08, "15", fontsize=8, ha="center")
    ax1.text(20, 0.08, "20", fontsize=8, ha="center")
    ax1.text(10.5, -0.14, "Borax concentration (mM)", ha="center", va="top", fontsize=8.2)

    # Panel b: UV-Vis larger
    ax2.set_title("Representative UV–Vis fingerprints", fontsize=9.8, pad=8)
    if not uv_spectra.empty:
        selected = [0, 5, 10, 15, 20]
        for c in selected:
            dfc = uv_spectra[np.isclose(uv_spectra["concentration_mM"], c)]
            if not dfc.empty:
                s = dfc.groupby("wavelength_nm")["absorbance"].mean().reset_index()
                ax2.plot(s["wavelength_nm"], s["absorbance"], lw=1.35, label=f"{c} mM")
        ax2.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax2.set_xlabel("Wavelength (nm)", fontsize=9)
    ax2.set_ylabel("Absorbance", fontsize=9)
    ax2.grid(alpha=0.14, lw=0.55)
    ax2.tick_params(labelsize=8)

    # Panel c: simplified vertical evidence path, no cramped boxes
    ax3.set_title("Independent prediction and interpretation", fontsize=9.8, pad=8)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    items = [
        ("Color feature matrix", "RGB, HSV, CIELab"),
        ("Independent validation", "Set A ↔ Set B"),
        ("Feature attribution", "permutation / SHAP"),
        ("Borax concentration", "mM output"),
    ]
    y_positions = [0.76, 0.55, 0.34, 0.13]
    for i, ((title, sub), y) in enumerate(zip(items, y_positions)):
        _draw_round_box(ax3, (0.18, y), (0.64, 0.115), title, sub, fc="white", ec="0.32",
                        lw=0.8, title_size=8.4, subtitle_size=6.8)
        if i < len(items)-1:
            _draw_arrow(ax3, (0.50, y-0.006), (0.50, y_positions[i+1]+0.132), lw=0.85)

    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)


def _draw_hex(ax, x, y, r=0.045, fc="#8E5BA6", ec="0.15", lw=1.0):
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6
    pts = np.column_stack([x + r*np.cos(angles), y + r*np.sin(angles)])
    ax.add_patch(plt.Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw))


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Cleaner mechanism figure: no overlapping titles, simplified chemistry, readable color trajectory."""
    from scipy.stats import pearsonr, spearmanr
    from scipy.stats import t as student_t_local

    fig = plt.figure(figsize=(14.8, 6.6), constrained_layout=False)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.55, 1.05, 1.00], wspace=0.36)
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 y=0.975, fontsize=15.2, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel a: clean schematic, large enough, no long table-like labels
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    ax1.text(-0.03, 0.98, "a", transform=ax1.transAxes, fontsize=13, fontweight="bold", va="top")
    ax1.set_title("pH-driven anthocyanin structural transformation", fontsize=11.3, fontweight="bold", pad=12)

    ax1.text(0.08, 0.80, "increasing borax concentration / pH", fontsize=9.6, fontweight="bold", ha="left")
    _draw_arrow(ax1, (0.48, 0.80), (0.90, 0.80), lw=1.1)

    y = 0.48
    # flavylium-like fused rings
    _draw_hex(ax1, 0.18, y, fc="#8D5DA9")
    _draw_hex(ax1, 0.25, y, fc="#8D5DA9")
    ax1.text(0.215, y, "O⁺", ha="center", va="center", fontsize=9, fontweight="bold")
    ax1.text(0.18, y+0.11, "OH", fontsize=7.2, ha="center")
    ax1.text(0.25, y+0.11, "OH", fontsize=7.2, ha="center")
    _draw_arrow(ax1, (0.30, y), (0.39, y), lw=0.9)

    # quinoidal-like rings
    _draw_hex(ax1, 0.45, y, fc="#2E65AE")
    _draw_hex(ax1, 0.52, y, fc="#3E74BC")
    ax1.text(0.485, y, "O", ha="center", va="center", fontsize=9, fontweight="bold")
    ax1.text(0.45, y+0.11, "OH", fontsize=7.2, ha="center")
    _draw_arrow(ax1, (0.57, y), (0.66, y), lw=0.9)

    # chalcone-like open chain
    _draw_hex(ax1, 0.72, y, fc="#2E8074")
    ax1.plot([0.76, 0.83, 0.90], [y, y+0.035, y+0.035], color="0.12", lw=1.2)
    ax1.text(0.82, y+0.085, "C=C–C=O", fontsize=7.4, ha="center")
    _draw_hex(ax1, 0.94, y, fc="#3B8E7D")

    # species labels in neat row
    species = [
        (0.215, "Flavylium cation", "purple", "dominant at lower pH"),
        (0.485, "Quinoidal base", "blue", "deprotonated chromophore"),
        (0.820, "Anionic / chalcone-like", "green-blue", "ring-opening equilibrium"),
    ]
    for x, name, color, note in species:
        ax1.add_patch(FancyBboxPatch((x-0.13, 0.27), 0.26, 0.075,
                                     boxstyle="round,pad=0.012,rounding_size=0.012",
                                     fc="white", ec="0.60", lw=0.7))
        ax1.text(x, 0.314, name, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax1.text(x, 0.238, color, ha="center", va="center", fontsize=8.1, color="0.35")
        ax1.text(x, 0.185, note, ha="center", va="center", fontsize=7.1, color="0.38")
    ax1.text(0.50, 0.065,
             "Simplified anthocyanin-species schematic; substituent pattern depends on the natural extract composition.",
             ha="center", fontsize=7.5, color="0.40")

    # ------------------------------------------------------------------
    # Panel b: pH-hue coupling, simplified statistics
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(-0.13, 1.03, "b", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="bottom")
    ax2.set_title("pH–hue mechanistic coupling", fontsize=11.3, fontweight="bold", pad=12)
    # Robust concentration-level summary. Do NOT use aggregate_color() here because
    # it flattens names into H_mean_mean / H_mean_std, which caused KeyError: H_mean.
    if not color_df.empty and {"concentration_mM", "H_mean"}.issubset(color_df.columns):
        agg = (color_df.groupby("concentration_mM")
               .agg(H_mean=("H_mean", "mean"),
                    H_std=("H_mean", "std"),
                    pH=("pH", "mean") if "pH" in color_df.columns else ("concentration_mM", "mean"))
               .reindex(ALL_CONC)
               .dropna(subset=["H_mean"]))
        if "pH" not in agg.columns or agg["pH"].isna().all():
            agg["pH"] = pd.Series(agg.index, index=agg.index).round().astype(int).map(PH_MAP).astype(float)
        else:
            agg["pH"] = agg["pH"].fillna(pd.Series(agg.index, index=agg.index).round().astype(int).map(PH_MAP).astype(float))
    else:
        agg = pd.DataFrame()
    if not agg.empty:
        x = agg["pH"].values.astype(float)
        yv = agg["H_mean"].values.astype(float)
        yerr = agg["H_std"].fillna(0).values.astype(float)
        conc = agg.index.values.astype(float)
        sc = ax2.scatter(x, yv, c=conc, cmap="viridis", s=44, edgecolor="k", linewidth=0.35, zorder=3)
        ax2.errorbar(x, yv, yerr=yerr, fmt="none", ecolor="0.35", elinewidth=0.7, capsize=2, zorder=2)
        order = np.argsort(x)
        xs, ys = x[order], yv[order]
        coef = np.polyfit(xs, ys, deg=2)
        xx = np.linspace(xs.min(), xs.max(), 250)
        yy = np.polyval(coef, xx)
        ax2.plot(xx, yy, color="0.15", lw=2.0, zorder=4)
        resid = ys - np.polyval(coef, xs)
        dof = max(len(xs) - 3, 1)
        s_err = np.sqrt(np.sum(resid**2) / dof)
        ci = student_t_local.ppf(0.975, dof) * s_err * 0.75
        ax2.fill_between(xx, yy-ci, yy+ci, color="0.15", alpha=0.13, lw=0, zorder=1)
        r, p = pearsonr(x, yv); rho, _ = spearmanr(x, yv)
        ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
        ax2.text(0.96, 0.95, f"r = {r:.3f}\nρ = {rho:.3f}\nR² = {r*r:.3f}\n{ptxt}",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8.0,
                 bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.60", alpha=0.92))
        for c in [0, 5, 10, 15, 20]:
            idxs = np.where(np.isclose(conc, c))[0]
            if len(idxs):
                i = idxs[0]
                ax2.annotate(str(int(c)), (x[i], yv[i]), xytext=(5, 4), textcoords="offset points", fontsize=7.2)
        cb = fig.colorbar(sc, ax=ax2, fraction=0.030, pad=0.018)
        cb.set_label("mM", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    ax2.set_xlabel("Measured pH")
    ax2.set_ylabel("HSV hue angle (°)")
    ax2.grid(alpha=0.12, lw=0.55)
    ax2.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Panel c: clean concentration-color trajectory only
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(-0.13, 1.03, "c", transform=ax3.transAxes, fontsize=13, fontweight="bold", va="bottom")
    ax3.set_title("Smartphone color trajectory", fontsize=11.3, fontweight="bold", pad=12)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    if not color_df.empty:
        means = color_df.groupby("concentration_mM")[["R_mean", "G_mean", "B_mean", "H_mean"]].mean().reindex(ALL_CONC).dropna()
        n = len(means)
        x0, y0, w, h = 0.33, 0.16, 0.26, 0.72
        for i, (conc, row) in enumerate(means.iterrows()):
            y = y0 + h * (1 - (i+1)/n)
            ax3.add_patch(Rectangle((x0, y), w, h/n, fc=_rgb01_from_row(row), ec="white", lw=0.32))
        for c in [0, 5, 10, 15, 20]:
            if c in means.index:
                idx = list(means.index).index(c)
                yy = y0 + h * (1 - (idx + 0.5)/n)
                ax3.text(x0-0.04, yy, f"{int(c)}", ha="right", va="center", fontsize=8.5, fontweight="bold")
                ax3.plot([x0-0.03, x0-0.006], [yy, yy], color="0.28", lw=0.75)
                ax3.text(x0+w+0.06, yy, f"H = {means.loc[c, 'H_mean']:.1f}°", ha="left", va="center", fontsize=8.4)
        ax3.text(x0-0.04, y0+h+0.035, "mM", ha="right", va="center", fontsize=8.5, fontweight="bold")
        ax3.annotate("increasing\nborax", xy=(0.78, y0+0.10), xytext=(0.78, y0+h-0.10),
                     ha="center", va="center", fontsize=8.0, color="0.28",
                     arrowprops=dict(arrowstyle="-|>", lw=1.05, color="0.28"))
        ax3.text(x0 + w/2, y0-0.06, "mean ROI color", ha="center", fontsize=8.3)
        ax3.text(0.50, 0.045, "Color transition extracted directly from smartphone ROIs",
                 ha="center", fontsize=7.4, color="0.35")

    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)


# =============================================================================
# FINAL Q1-CLEAN OVERRIDES FOR FIGURE 1 AND FIGURE 2
# These functions intentionally reduce text density and increase whitespace.
# They overwrite earlier definitions while preserving all other figures/tables.
# =============================================================================

def _safe_color_summary_for_visual(color_df: pd.DataFrame) -> pd.DataFrame:
    """Return concentration-level mean RGB/Hue/pH with robust column handling."""
    if color_df is None or color_df.empty:
        rows=[]
        for c in ALL_CONC:
            rows.append({"concentration_mM":c,"R_mean":40,"G_mean":60,"B_mean":95,"H_mean":210,"pH":PH_MAP.get(c, np.nan)})
        return pd.DataFrame(rows).set_index("concentration_mM")
    df=color_df.copy()
    if "pH" not in df.columns:
        df["pH"] = df["concentration_mM"].map(PH_MAP)
    # handle possible alternative hue names
    if "H_mean" not in df.columns:
        for alt in ["H_mean_mean", "Hue_mean", "HSV_H_mean", "H"]:
            if alt in df.columns:
                df["H_mean"] = df[alt]
                break
    if "H_mean" not in df.columns:
        df["H_mean"] = np.nan
    needed=[]
    for c in ["R_mean","G_mean","B_mean","H_mean","pH"]:
        if c in df.columns:
            needed.append(c)
    agg=df.groupby("concentration_mM")[needed].mean(numeric_only=True)
    for c in ["R_mean","G_mean","B_mean","H_mean","pH"]:
        if c not in agg.columns:
            if c=="pH": agg[c]=pd.Series({k:PH_MAP.get(k,np.nan) for k in agg.index})
            elif c=="H_mean": agg[c]=np.nan
            else: agg[c]=60
    return agg.sort_index()


def _rgb01_from_series(row) -> tuple:
    try:
        return (float(row["R_mean"])/255.0, float(row["G_mean"])/255.0, float(row["B_mean"])/255.0)
    except Exception:
        return (0.10,0.20,0.32)


def _draw_q1_box(ax, xy, wh, title, subtitle, number, fc="#F7F7F7", ec="0.15"):
    x,y=xy; w,h=wh
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.025",
                         fc=fc, ec=ec, lw=1.15, zorder=2)
    ax.add_patch(patch)
    circ=Circle((x+0.028,y+h-0.033),0.017,fc="0.12",ec="none",zorder=3)
    ax.add_patch(circ)
    ax.text(x+0.028,y+h-0.033,str(number),ha="center",va="center",fontsize=8.0,
            color="white",fontweight="bold",zorder=4)
    ax.text(x+w/2,y+h*0.61,title,ha="center",va="center",fontsize=10.0,
            fontweight="bold",linespacing=0.94,zorder=4)
    ax.text(x+w/2,y+h*0.28,subtitle,ha="center",va="center",fontsize=7.5,
            color="0.33",linespacing=0.98,zorder=4)
    return patch


def _arrow_between(ax, x1, y1, x2, y2, color="0.20"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.35, color=color,
                                shrinkA=1, shrinkB=1, mutation_scale=10), zorder=1)


def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Clean two-layer graphical workflow with generous spacing for Q1 submission."""
    set_premium_style()
    summary=_safe_color_summary_for_visual(color_df)
    fig=plt.figure(figsize=(17.8,9.2), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Multimodal smartphone colorimetry for anthocyanin-assisted borax quantification",
                 fontsize=18, fontweight="bold", y=0.965)

    # ===================== Top workflow axis =====================
    ax=fig.add_axes([0.055,0.595,0.89,0.275])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.045,0.70,"Chemical\nvalidation",rotation=90,ha="center",va="center",
            fontsize=9.8,fontweight="bold",color="0.18",linespacing=0.92)
    ax.text(0.045,0.29,"Digital\nquantification",rotation=90,ha="center",va="center",
            fontsize=9.8,fontweight="bold",color="0.18",linespacing=0.92)
    ax.plot([0.09,0.96],[0.50,0.50],ls=(0,(4,4)),lw=0.85,color="0.78")
    ax.text(0.525,0.515,"independent chemical validation supports smartphone-based quantification",
            ha="center",va="bottom",fontsize=8.0,color="0.40")

    row1=[
        ("Borax\nstandard","alkalization",1,"#EAF4FF"),
        ("Purple cabbage\nanthocyanin","natural pH probe",2,"#F2EAFB"),
        ("Anthocyanin\nconversion","pH-responsive forms",3,"#F8F8F8"),
        ("UV–Vis\nvalidation","spectral fingerprint",4,"#ECF8F1"),
    ]
    row2=[
        ("Smartphone\nimaging","controlled acquisition",5,"#FFF4E2"),
        ("ROI + color\nfeatures","RGB–HSV–CIELab",6,"#EAF6FF"),
        ("Chemometrics\n+ XAI","PCA, ML, SHAP",7,"#F8F8F8"),
        ("Borax\nprediction","quantitative output",8,"#ECFAEC"),
    ]
    xs=[0.105,0.320,0.535,0.750]; w=0.165; h=0.235
    y1=0.62; y2=0.20
    for i,item in enumerate(row1):
        _draw_q1_box(ax,(xs[i],y1),(w,h),*item)
    for i,item in enumerate(row2):
        _draw_q1_box(ax,(xs[i],y2),(w,h),*item)
    for i in range(3):
        _arrow_between(ax,xs[i]+w+0.010,y1+h/2,xs[i+1]-0.010,y1+h/2)
        _arrow_between(ax,xs[i]+w+0.010,y2+h/2,xs[i+1]-0.010,y2+h/2)
    # Layer transition is shown by dashed line, not a misleading process arrow.
    ax.text(0.50,0.015,
            "Borax-induced alkalization converts anthocyanin chemistry into UV–Vis-validated smartphone color signatures for explainable prediction",
            ha="center",va="bottom",fontsize=10.3,fontweight="bold")

    # ===================== Lower data panels =====================
    ax1=fig.add_axes([0.07,0.150,0.235,0.275])
    ax2=fig.add_axes([0.365,0.145,0.315,0.300])
    ax3=fig.add_axes([0.745,0.150,0.205,0.275])

    # Panel a: color scale
    ax1.text(-0.12,1.10,"a",transform=ax1.transAxes,fontsize=13,fontweight="bold")
    ax1.set_title("Smartphone color scale",fontsize=11,pad=8)
    concs=[c for c in ALL_CONC if c in summary.index]
    if concs:
        arr=np.array([[ _rgb01_from_series(summary.loc[c]) for c in concs ]])
        ax1.imshow(arr,aspect="auto",extent=[min(concs),max(concs),0,1])
        for c in concs:
            ax1.axvline(c,color="white",lw=0.45,alpha=0.72)
        ax1.set_xlim(min(concs),max(concs)); ax1.set_ylim(0,1)
    ax1.set_yticks([]); ax1.set_xticks([0,5,10,15,20])
    ax1.set_xlabel("Borax concentration (mM)",fontsize=9,labelpad=14)
    for s in ax1.spines.values(): s.set_visible(False)

    # Panel b: UV-Vis selected spectra
    ax2.text(-0.10,1.07,"b",transform=ax2.transAxes,fontsize=13,fontweight="bold")
    ax2.set_title("Representative UV–Vis fingerprints",fontsize=11,pad=8)
    if uv_spectra is not None and not uv_spectra.empty:
        for c in [0,5,10,15,20]:
            sub=uv_spectra[np.isclose(uv_spectra["concentration_mM"],c)]
            if not sub.empty:
                g=sub.groupby("wavelength_nm",as_index=False)["absorbance"].mean()
                ax2.plot(g["wavelength_nm"],g["absorbance"],lw=1.3,label=f"{c} mM")
        ax2.legend(frameon=False,fontsize=7.5,loc="upper right")
    ax2.set_xlabel("Wavelength (nm)",fontsize=9)
    ax2.set_ylabel("Absorbance",fontsize=9)
    ax2.grid(True,alpha=0.18,lw=0.5)

    # Panel c: prediction schematic, minimalist
    ax3.text(-0.12,1.10,"c",transform=ax3.transAxes,fontsize=13,fontweight="bold")
    ax3.set_title("Prediction and interpretation",fontsize=11,pad=8)
    ax3.set_xlim(0,1); ax3.set_ylim(0,1); ax3.axis("off")
    steps=[("Color feature matrix","RGB, HSV, CIELab"),
           ("Independent validation","Set A ↔ Set B"),
           ("Feature attribution","permutation / SHAP"),
           ("Borax concentration","mM output")]
    yy=[0.78,0.57,0.36,0.15]
    for (title,sub),y in zip(steps,yy):
        ax3.add_patch(FancyBboxPatch((0.13,y),0.74,0.13,boxstyle="round,pad=0.018,rounding_size=0.018",
                                     fc="white",ec="0.25",lw=0.8))
        ax3.text(0.50,y+0.082,title,ha="center",va="center",fontsize=8.6,fontweight="bold")
        ax3.text(0.50,y+0.038,sub,ha="center",va="center",fontsize=7.2,color="0.36")
    for y in [0.70,0.49,0.28]:
        _arrow_between(ax3,0.50,y,0.50,y-0.07,color="0.25")

    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)


def _draw_simple_molecule(ax, cx, cy, color, label, subtitle, note, charged=False, chalcone=False, scale=1.0):
    """Clean schematic anthocyanin species; simplified intentionally for natural extract."""
    # aromatic rings as clean hexagons; minimized text to avoid collisions
    r=0.045*scale
    def hexagon(x,y,fc):
        ang=np.linspace(0,2*np.pi,7)+np.pi/6
        pts=np.column_stack([x+r*np.cos(ang), y+r*np.sin(ang)])
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="0.12", lw=1.0))
    if chalcone:
        hexagon(cx-0.055*scale,cy,color)
        ax.plot([cx-0.010*scale,cx+0.075*scale],[cy+0.005*scale,cy+0.045*scale],color="0.12",lw=1.1)
        ax.text(cx+0.010*scale,cy+0.070*scale,"C=C–C=O",fontsize=7.2,ha="center")
        hexagon(cx+0.120*scale,cy,color)
    else:
        hexagon(cx-0.045*scale,cy,color)
        hexagon(cx+0.045*scale,cy,color)
        ax.text(cx,cy,"O⁺" if charged else "O",ha="center",va="center",fontsize=8.5,fontweight="bold")
        ax.text(cx-0.065*scale,cy+0.075*scale,"OH",fontsize=7,ha="center")
        ax.text(cx+0.065*scale,cy+0.075*scale,"OH",fontsize=7,ha="center")
    # label box
    ax.add_patch(FancyBboxPatch((cx-0.125,cy-0.165),0.25,0.075,boxstyle="round,pad=0.012,rounding_size=0.012",
                                fc="white",ec="0.55",lw=0.75))
    ax.text(cx,cy-0.125,label,ha="center",va="center",fontsize=8.2,fontweight="bold")
    ax.text(cx,cy-0.195,subtitle,ha="center",va="center",fontsize=7.3,color="0.35")
    ax.text(cx,cy-0.250,note,ha="center",va="center",fontsize=6.6,color="0.42")


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Clean, low-density mechanism figure: molecular schematic + pH/hue + compact color trajectory."""
    set_premium_style()
    summary=_safe_color_summary_for_visual(color_df)
    fig=plt.figure(figsize=(18.0,7.2),dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 fontsize=18,fontweight="bold",y=0.975)

    axA=fig.add_axes([0.055,0.120,0.370,0.745])
    axB=fig.add_axes([0.505,0.145,0.260,0.690])
    axC=fig.add_axes([0.815,0.145,0.145,0.690])

    # Panel A: molecule schematic
    axA.set_xlim(0,1); axA.set_ylim(0,1); axA.axis("off")
    axA.text(-0.04,1.02,"a",transform=axA.transAxes,fontsize=14,fontweight="bold",va="bottom")
    axA.set_title("pH-driven anthocyanin structural transformation",fontsize=12.2,fontweight="bold",pad=8)
    axA.annotate("increasing borax concentration / pH", xy=(0.83,0.72), xytext=(0.18,0.72),
                 ha="center", va="center", fontsize=9.6, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>",lw=1.2,color="0.25"))
    y_mol=0.43
    xs=[0.24,0.50,0.76]
    _draw_simple_molecule(axA,xs[0],y_mol,"#8A59A5","Flavylium cation","purple","dominant at lower pH",charged=True,scale=1.0)
    _draw_simple_molecule(axA,xs[1],y_mol,"#326EB6","Quinoidal base","blue","deprotonated chromophore",scale=1.0)
    _draw_simple_molecule(axA,xs[2],y_mol,"#2B8A78","Chalcone-like form","green-blue","ring-opening equilibrium",chalcone=True,scale=1.0)
    _arrow_between(axA,xs[0]+0.095,y_mol,xs[1]-0.095,y_mol,color="0.20")
    _arrow_between(axA,xs[1]+0.095,y_mol,xs[2]-0.120,y_mol,color="0.20")
    axA.text(0.50,0.045,"Simplified anthocyanin-species schematic; substituent pattern depends on the natural extract composition.",
             ha="center",va="center",fontsize=7.4,color="0.38")

    # Panel B: pH-hue coupling
    axB.text(-0.18,1.04,"b",transform=axB.transAxes,fontsize=14,fontweight="bold",va="bottom")
    axB.set_title("pH–hue mechanistic coupling",fontsize=12.2,fontweight="bold",pad=8)
    valid=summary.dropna(subset=["pH","H_mean"])
    if not valid.empty and len(valid)>=4:
        x=valid["pH"].astype(float).values
        y=valid["H_mean"].astype(float).values
        c=valid.index.astype(float).values
        # polynomial fit for visual mechanistic trend
        order=2 if len(x)>=6 else 1
        coef=np.polyfit(x,y,order)
        xs_fit=np.linspace(np.nanmin(x),np.nanmax(x),220)
        ys_fit=np.polyval(coef,xs_fit)
        resid=y-np.polyval(coef,x)
        band=1.96*np.nanstd(resid) if np.isfinite(np.nanstd(resid)) else 1.0
        axB.fill_between(xs_fit,ys_fit-band,ys_fit+band,color="0.88",alpha=1.0,lw=0,zorder=0)
        axB.plot(xs_fit,ys_fit,color="0.15",lw=2.0,zorder=1)
        sc=axB.scatter(x,y,c=c,cmap="viridis",s=48,edgecolor="black",lw=0.55,zorder=3)
        # error bars if available in raw data group summary
        if color_df is not None and not color_df.empty and "H_mean" in color_df.columns:
            tmp=color_df.copy(); tmp["pH"] = tmp["concentration_mM"].map(PH_MAP) if "pH" not in tmp.columns else tmp["pH"]
            sd=tmp.groupby("concentration_mM")["H_mean"].std()
            for conc,row in valid.iterrows():
                if conc in sd.index and np.isfinite(sd.loc[conc]):
                    axB.errorbar(row["pH"],row["H_mean"],yerr=sd.loc[conc],fmt="none",ecolor="0.35",elinewidth=0.8,capsize=2,zorder=2)
        for conc in [0,5,10,15,20]:
            if conc in valid.index:
                axB.text(valid.loc[conc,"pH"]+0.02,valid.loc[conc,"H_mean"]+0.65,str(conc),fontsize=8.0)
        try:
            from scipy.stats import pearsonr, spearmanr
            rr,pp=pearsonr(x,y); rho,sp=spearmanr(x,y)
            txt=f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\np < 0.001" if pp<0.001 else f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\np = {pp:.3g}"
        except Exception:
            rr=np.corrcoef(x,y)[0,1]
            txt=f"r = {rr:.3f}\nR² = {rr**2:.3f}"
        axB.text(0.96,0.96,txt,transform=axB.transAxes,ha="right",va="top",fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.28",fc="white",ec="0.60",alpha=0.92))
        cb=fig.colorbar(sc,ax=axB,fraction=0.035,pad=0.016)
        cb.set_label("Borax concentration (mM)",fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axB.set_xlabel("Measured pH",fontsize=10)
    axB.set_ylabel("HSV hue angle (°)",fontsize=10)
    axB.grid(True,alpha=0.18,lw=0.5)

    # Panel C: clean color trajectory
    axC.text(-0.22,1.04,"c",transform=axC.transAxes,fontsize=14,fontweight="bold",va="bottom")
    axC.set_title("Smartphone color trajectory",fontsize=12.2,fontweight="bold",pad=8)
    axC.set_xlim(0,1); axC.set_ylim(0,1); axC.axis("off")
    means=summary.reindex([c for c in ALL_CONC if c in summary.index]).dropna(subset=["R_mean","G_mean","B_mean"])
    if not means.empty:
        n=len(means)
        x0,y0,w,h=0.26,0.16,0.34,0.70
        for i,(conc,row) in enumerate(means.iterrows()):
            yy=y0+h*(1-(i+1)/n)
            axC.add_patch(Rectangle((x0,yy),w,h/n,fc=_rgb01_from_series(row),ec="white",lw=0.35))
        axC.text(x0-0.06,y0+h+0.035,"mM",ha="right",va="center",fontsize=9,fontweight="bold")
        for conc in [0,5,10,15,20]:
            if conc in means.index:
                idx=list(means.index).index(conc)
                yy=y0+h*(1-(idx+0.5)/n)
                hue=means.loc[conc,"H_mean"] if "H_mean" in means.columns else np.nan
                axC.text(x0-0.045,yy,f"{int(conc)}",ha="right",va="center",fontsize=8.5,fontweight="bold")
                axC.plot([x0-0.030,x0-0.006],[yy,yy],color="0.28",lw=0.75)
                if np.isfinite(hue):
                    axC.text(x0+w+0.08,yy,f"H = {hue:.1f}°",ha="left",va="center",fontsize=8.3)
        axC.annotate("increasing\nborax",xy=(0.86,y0+0.10),xytext=(0.86,y0+h-0.12),
                     ha="center",va="center",fontsize=8.0,color="0.30",
                     arrowprops=dict(arrowstyle="-|>",lw=1.05,color="0.30"))
        axC.text(x0+w/2,y0-0.065,"mean ROI color",ha="center",fontsize=8.3)
    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)




# =====================================================================
# FINAL VISUAL REVISION: clean Q1-style Figure 1 and Figure 2
# These definitions override earlier versions before main() is called.
# =====================================================================

def _draw_q1_box(ax, xy, wh, title, subtitle, step=None, fc="#F7F7F7", ec="0.18"):
    """Clean workflow box without distracting black numbered circles."""
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        fc=fc, ec=ec, lw=1.05, zorder=2
    )
    ax.add_patch(patch)
    if step is not None:
        ax.text(
            x + 0.018, y + h - 0.020, f"Step {step}",
            ha="left", va="top", fontsize=6.6,
            color="0.38", fontweight="bold", zorder=4
        )
    ax.text(
        x + w/2, y + h*0.60, title,
        ha="center", va="center", fontsize=9.6,
        fontweight="bold", linespacing=0.92, zorder=4
    )
    ax.text(
        x + w/2, y + h*0.30, subtitle,
        ha="center", va="center", fontsize=7.2,
        color="0.35", linespacing=0.96, zorder=4
    )
    return patch


def _arrow_between(ax, x1, y1, x2, y2, color="0.25", lw=1.25, ms=10):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", lw=lw, color=color,
            shrinkA=0, shrinkB=0, mutation_scale=ms
        ),
        zorder=1
    )


def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Final clean graphical workflow: no black circles, tighter panel balance, no arrow-text collisions."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(17.5, 8.6), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Multimodal smartphone colorimetry for anthocyanin-assisted borax quantification",
        fontsize=18, fontweight="bold", y=0.965
    )

    # -------------------- Two-layer workflow --------------------
    ax = fig.add_axes([0.075, 0.620, 0.850, 0.250])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Subtle layer labels outside the boxes
    ax.text(0.035, 0.710, "Chemical\nvalidation", rotation=90,
            ha="center", va="center", fontsize=9.2, fontweight="bold",
            color="0.22", linespacing=0.85)
    ax.text(0.035, 0.295, "Digital\nquantification", rotation=90,
            ha="center", va="center", fontsize=9.2, fontweight="bold",
            color="0.22", linespacing=0.85)

    ax.plot([0.085, 0.965], [0.505, 0.505], ls=(0, (4, 4)), lw=0.8, color="0.78")
    ax.text(0.525, 0.515,
            "independent chemical validation supports smartphone-based quantification",
            ha="center", va="bottom", fontsize=7.7, color="0.44")

    row1 = [
        ("Borax\nstandard", "alkalization", 1, "#EAF4FF"),
        ("Purple cabbage\nanthocyanin", "natural pH probe", 2, "#F2EAFB"),
        ("Anthocyanin\nconversion", "pH-responsive forms", 3, "#F8F8F8"),
        ("UV–Vis\nvalidation", "spectral fingerprint", 4, "#ECF8F1"),
    ]
    row2 = [
        ("Smartphone\nimaging", "controlled acquisition", 5, "#FFF4E2"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", 6, "#EAF6FF"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", 7, "#F8F8F8"),
        ("Borax\nprediction", "quantitative output", 8, "#ECFAEC"),
    ]

    xs = [0.100, 0.315, 0.530, 0.745]
    w = 0.165
    h = 0.205
    y1 = 0.635
    y2 = 0.215

    for i, item in enumerate(row1):
        title, sub, step, fc = item
        _draw_q1_box(ax, (xs[i], y1), (w, h), title, sub, step=step, fc=fc)
    for i, item in enumerate(row2):
        title, sub, step, fc = item
        _draw_q1_box(ax, (xs[i], y2), (w, h), title, sub, step=step, fc=fc)

    for i in range(3):
        _arrow_between(ax, xs[i] + w + 0.010, y1 + h/2, xs[i+1] - 0.010, y1 + h/2)
        _arrow_between(ax, xs[i] + w + 0.010, y2 + h/2, xs[i+1] - 0.010, y2 + h/2)

    ax.text(
        0.525, 0.040,
        "Borax-induced alkalization generates UV–Vis-validated smartphone color signatures for explainable prediction",
        ha="center", va="center", fontsize=10.5, fontweight="bold"
    )

    # -------------------- Lower evidence panels --------------------
    # Balanced: color scale and prediction schematic are compact; UV-Vis gets the widest panel.
    ax1 = fig.add_axes([0.070, 0.145, 0.260, 0.305])
    ax2 = fig.add_axes([0.365, 0.145, 0.365, 0.305])
    ax3 = fig.add_axes([0.780, 0.145, 0.165, 0.305])

    # Panel a: color scale
    ax1.text(-0.12, 1.08, "a", transform=ax1.transAxes, fontsize=13, fontweight="bold")
    ax1.set_title("Smartphone color scale", fontsize=11.3, pad=8)
    concs = [c for c in ALL_CONC if c in summary.index]
    if concs:
        arr = np.array([[_rgb01_from_series(summary.loc[c]) for c in concs]])
        ax1.imshow(arr, aspect="auto", extent=[min(concs), max(concs), 0, 1])
        for c in concs:
            ax1.axvline(c, color="white", lw=0.45, alpha=0.70)
        ax1.set_xlim(min(concs), max(concs))
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_xlabel("Borax concentration (mM)", fontsize=8.8, labelpad=12)
    for s in ax1.spines.values():
        s.set_visible(False)

    # Panel b: UV-Vis fingerprints
    ax2.text(-0.105, 1.08, "b", transform=ax2.transAxes, fontsize=13, fontweight="bold")
    ax2.set_title("Representative UV–Vis fingerprints", fontsize=11.3, pad=8)
    if uv_spectra is not None and not uv_spectra.empty:
        for c in [0, 5, 10, 15, 20]:
            sub = uv_spectra[np.isclose(uv_spectra["concentration_mM"], c)]
            if not sub.empty:
                g = sub.groupby("wavelength_nm", as_index=False)["absorbance"].mean()
                ax2.plot(g["wavelength_nm"], g["absorbance"], lw=1.35, label=f"{c} mM")
        ax2.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax2.set_xlabel("Wavelength (nm)", fontsize=9.2)
    ax2.set_ylabel("Absorbance", fontsize=9.2)
    ax2.grid(True, alpha=0.16, lw=0.5)

    # Panel c: no arrows crossing text
    ax3.text(-0.18, 1.08, "c", transform=ax3.transAxes, fontsize=13, fontweight="bold")
    ax3.set_title("Prediction workflow", fontsize=11.3, pad=8)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    steps = [
        ("Color features", "RGB, HSV, CIELab"),
        ("Validation", "Set A ↔ Set B"),
        ("Attribution", "Permutation / SHAP"),
        ("Prediction", "mM output"),
    ]
    yy = [0.80, 0.58, 0.36, 0.14]
    box_x, box_w, box_h = 0.08, 0.84, 0.135
    for (title, sub), y in zip(steps, yy):
        ax3.add_patch(FancyBboxPatch(
            (box_x, y), box_w, box_h,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            fc="white", ec="0.25", lw=0.82
        ))
        ax3.text(0.50, y + 0.085, title, ha="center", va="center",
                 fontsize=8.2, fontweight="bold")
        ax3.text(0.50, y + 0.038, sub, ha="center", va="center",
                 fontsize=6.9, color="0.36")
    # arrows only in the gaps, not through the boxes
    for y_top, y_bot in [(0.80, 0.58), (0.58, 0.36), (0.36, 0.14)]:
        _arrow_between(ax3, 0.50, y_top - 0.010, 0.50, y_bot + box_h + 0.012,
                       color="0.25", lw=1.1, ms=9)

    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)


def _draw_species_card(ax, x, y, w, h, color, title, state, note, charged=False, chalcone=False):
    """Premium clean species card: color + simplified chromophore schematic + concise label."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        fc="white", ec="0.65", lw=0.85
    ))
    # color chip
    ax.add_patch(Rectangle((x + 0.020, y + h - 0.060), w - 0.040, 0.035,
                           fc=color, ec="0.25", lw=0.45))
    # schematic area
    cx = x + w/2
    cy = y + h*0.53
    r = min(w, h) * 0.105

    def ring(cx0, cy0, fc):
        ang = np.linspace(0, 2*np.pi, 7) + np.pi/6
        pts = np.column_stack([cx0 + r*np.cos(ang), cy0 + r*np.sin(ang)])
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="0.15", lw=1.0))

    if chalcone:
        ring(cx - 0.055, cy, color)
        ax.plot([cx - 0.015, cx + 0.070], [cy, cy + 0.045], color="0.15", lw=1.15)
        ax.plot([cx + 0.060, cx + 0.105], [cy + 0.045, cy + 0.065], color="0.15", lw=1.15)
        ring(cx + 0.120, cy + 0.070, color)
        ax.text(cx + 0.035, cy + 0.085, "opened ring", ha="center", va="bottom", fontsize=6.3, color="0.25")
    else:
        ring(cx - 0.050, cy, color)
        ring(cx + 0.050, cy, color)
        ax.text(cx, cy, "O⁺" if charged else "O", ha="center", va="center",
                fontsize=8.0, fontweight="bold", color="0.10")
    # concise chemistry labels
    ax.text(cx, y + 0.075, title, ha="center", va="center", fontsize=8.7, fontweight="bold")
    ax.text(cx, y + 0.045, state, ha="center", va="center", fontsize=7.1, color="0.35")
    ax.text(cx, y + 0.018, note, ha="center", va="center", fontsize=6.6, color="0.48")


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Final clean mechanism figure: no text collisions, clear color-mechanism-data linkage."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(18.0, 7.0), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 fontsize=18, fontweight="bold", y=0.970)

    axA = fig.add_axes([0.055, 0.130, 0.390, 0.735])
    axB = fig.add_axes([0.520, 0.145, 0.245, 0.690])
    axC = fig.add_axes([0.825, 0.150, 0.135, 0.680])

    # -------------------- Panel A --------------------
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.axis("off")
    axA.text(-0.045, 1.025, "a", transform=axA.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axA.set_title("pH-driven anthocyanin transformation", fontsize=12.5, fontweight="bold", pad=10)

    axA.annotate(
        "increasing borax concentration / alkalinity",
        xy=(0.86, 0.83), xytext=(0.17, 0.83),
        ha="left", va="center", fontsize=9.2, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.15, color="0.25")
    )

    card_y, card_h, card_w = 0.300, 0.385, 0.245
    xs = [0.075, 0.375, 0.675]
    _draw_species_card(axA, xs[0], card_y, card_w, card_h,
                       "#8A59A5", "Flavylium", "cationic / purple", "low-pH dominant", charged=True)
    _draw_species_card(axA, xs[1], card_y, card_w, card_h,
                       "#326EB6", "Quinoidal", "neutral / blue", "deprotonated form")
    _draw_species_card(axA, xs[2], card_y, card_w, card_h,
                       "#2B8A78", "Chalcone-like", "open-ring / green-blue", "alkaline form", chalcone=True)
    _arrow_between(axA, xs[0] + card_w + 0.020, card_y + card_h*0.56,
                   xs[1] - 0.020, card_y + card_h*0.56, lw=1.2, ms=10)
    _arrow_between(axA, xs[1] + card_w + 0.020, card_y + card_h*0.56,
                   xs[2] - 0.020, card_y + card_h*0.56, lw=1.2, ms=10)

    axA.text(0.50, 0.075,
             "Simplified species schematic for natural anthocyanin extract; detailed substituents may vary.",
             ha="center", va="center", fontsize=7.5, color="0.40")

    # -------------------- Panel B --------------------
    axB.text(-0.19, 1.04, "b", transform=axB.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axB.set_title("pH–hue coupling", fontsize=12.5, fontweight="bold", pad=10)
    valid = summary.dropna(subset=["pH", "H_mean"])
    if not valid.empty and len(valid) >= 4:
        x = valid["pH"].astype(float).values
        y = valid["H_mean"].astype(float).values
        c = valid.index.astype(float).values
        order = 2 if len(x) >= 6 else 1
        coef = np.polyfit(x, y, order)
        xs_fit = np.linspace(np.nanmin(x), np.nanmax(x), 220)
        ys_fit = np.polyval(coef, xs_fit)
        resid = y - np.polyval(coef, x)
        band = 1.65 * np.nanstd(resid) if np.isfinite(np.nanstd(resid)) else 1.0
        axB.fill_between(xs_fit, ys_fit-band, ys_fit+band, color="0.88", alpha=1.0, lw=0, zorder=0)
        axB.plot(xs_fit, ys_fit, color="0.15", lw=2.0, zorder=1)
        sc = axB.scatter(x, y, c=c, cmap="viridis", s=46, edgecolor="black", lw=0.55, zorder=3)
        if color_df is not None and not color_df.empty and "H_mean" in color_df.columns:
            tmp = color_df.copy()
            tmp["pH"] = tmp["concentration_mM"].map(PH_MAP) if "pH" not in tmp.columns else tmp["pH"]
            sd = tmp.groupby("concentration_mM")["H_mean"].std()
            for conc, row in valid.iterrows():
                if conc in sd.index and np.isfinite(sd.loc[conc]):
                    axB.errorbar(row["pH"], row["H_mean"], yerr=sd.loc[conc], fmt="none",
                                 ecolor="0.35", elinewidth=0.8, capsize=2, zorder=2)
        for conc in [0, 5, 10, 15, 20]:
            if conc in valid.index:
                axB.text(valid.loc[conc, "pH"] + 0.018,
                         valid.loc[conc, "H_mean"] + 0.55,
                         str(conc), fontsize=7.8)
        try:
            from scipy.stats import pearsonr, spearmanr
            rr, pp = pearsonr(x, y)
            rho, sp = spearmanr(x, y)
            txt = f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\np < 0.001" if pp < 0.001 else f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\np = {pp:.3g}"
        except Exception:
            rr = np.corrcoef(x, y)[0, 1]
            txt = f"r = {rr:.3f}\nR² = {rr**2:.3f}"
        axB.text(0.955, 0.955, txt, transform=axB.transAxes, ha="right", va="top", fontsize=8.2,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.60", alpha=0.92))
        cb = fig.colorbar(sc, ax=axB, fraction=0.033, pad=0.015)
        cb.set_label("mM", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axB.set_xlabel("Measured pH", fontsize=10)
    axB.set_ylabel("HSV hue angle (°)", fontsize=10)
    axB.grid(True, alpha=0.18, lw=0.5)

    # -------------------- Panel C --------------------
    axC.text(-0.24, 1.04, "c", transform=axC.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axC.set_title("Color trajectory", fontsize=12.5, fontweight="bold", pad=10)
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.axis("off")
    means = summary.reindex([c for c in ALL_CONC if c in summary.index]).dropna(subset=["R_mean", "G_mean", "B_mean"])
    if not means.empty:
        n = len(means)
        x0, y0, w, h = 0.26, 0.24, 0.36, 0.60
        for i, (conc, row) in enumerate(means.iterrows()):
            yy = y0 + h * (1 - (i + 1) / n)
            axC.add_patch(Rectangle((x0, yy), w, h/n, fc=_rgb01_from_series(row), ec="white", lw=0.35))
        axC.text(x0-0.05, y0+h+0.045, "mM", ha="right", va="center", fontsize=8.8, fontweight="bold")
        for conc in [0, 5, 10, 15, 20]:
            if conc in means.index:
                idx = list(means.index).index(conc)
                yy = y0 + h * (1 - (idx + 0.5) / n)
                hue = means.loc[conc, "H_mean"] if "H_mean" in means.columns else np.nan
                axC.text(x0-0.045, yy, f"{int(conc)}", ha="right", va="center", fontsize=8.3, fontweight="bold")
                axC.plot([x0-0.028, x0-0.006], [yy, yy], color="0.28", lw=0.75)
                if np.isfinite(hue):
                    axC.text(x0+w+0.060, yy, f"H = {hue:.1f}°", ha="left", va="center", fontsize=8.0)
        # Text separate from arrow; arrow never crosses the words.
        axC.text(0.90, y0+h*0.72, "increasing\nborax", ha="center", va="center", fontsize=7.8, color="0.30")
        axC.annotate("", xy=(0.90, y0+h*0.20), xytext=(0.90, y0+h*0.60),
                     arrowprops=dict(arrowstyle="-|>", lw=1.05, color="0.30"))
        axC.text(x0+w/2, y0-0.060, "mean ROI color", ha="center", fontsize=8.0)
    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)



# =====================================================================
# FINAL STRICT Q1 VISUAL OVERRIDES (clean Fig. 1 and Fig. 2)
# - Fig. 1: wider boxes, no black numbered circles, no text collisions,
#           all internal labels dark/readable, prediction arrows do not cross text.
# - Fig. 2: simplified species-symbol cards, no problematic chemical cartoons,
#           no top concentration arrow, no arrow/text collision in color trajectory.
# These override earlier definitions before main() is called.
# =====================================================================

def _draw_q1_box(ax, xy, wh, title, subtitle="", step=None, fc="#F7F7F7", ec="0.20"):
    """Publication-style workflow box with readable dark text and no distracting markers."""
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        fc=fc, ec=ec, lw=1.05, zorder=2
    )
    ax.add_patch(patch)
    if step is not None:
        ax.text(x + 0.018, y + h - 0.026, f"Step {step}",
                ha="left", va="top", fontsize=7.1,
                color="0.25", fontweight="bold", zorder=4)
    # Use separated y positions so bold and normal text never collide.
    ax.text(x + w/2, y + h*0.60, title,
            ha="center", va="center", fontsize=9.8,
            fontweight="bold", color="0.02", linespacing=0.92, zorder=4)
    if subtitle:
        ax.text(x + w/2, y + h*0.245, subtitle,
                ha="center", va="center", fontsize=7.6,
                color="0.18", fontweight="medium", linespacing=0.96, zorder=4)
    return patch


def _arrow_between(ax, x1, y1, x2, y2, color="0.25", lw=1.25, ms=10):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0, mutation_scale=ms),
                zorder=1)


def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Q1-clean graphical workflow with wider boxes and tighter lower-panel spacing."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(17.5, 8.3), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Multimodal smartphone colorimetry for anthocyanin-assisted borax quantification",
                 fontsize=18, fontweight="bold", y=0.965)

    # Workflow block
    ax = fig.add_axes([0.055, 0.615, 0.890, 0.275])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.030, 0.735, "Chemical\nvalidation", rotation=90,
            ha="center", va="center", fontsize=9.0, fontweight="bold",
            color="0.12", linespacing=0.86)
    ax.text(0.030, 0.300, "Digital\nquantification", rotation=90,
            ha="center", va="center", fontsize=9.0, fontweight="bold",
            color="0.12", linespacing=0.86)

    # Subtle separator; no text over the boxes.
    ax.plot([0.080, 0.965], [0.505, 0.505], ls=(0, (4, 4)), lw=0.8, color="0.78")
    ax.text(0.525, 0.523, "independent chemical validation supports smartphone-based quantification",
            ha="center", va="bottom", fontsize=8.0, color="0.32")

    row1 = [
        ("Borax\nstandard", "alkalization", 1, "#EAF4FF"),
        ("Purple cabbage\nanthocyanin", "natural pH probe", 2, "#F2EAFB"),
        ("Anthocyanin\nconversion", "pH-responsive forms", 3, "#F8F8F8"),
        ("UV–Vis\nvalidation", "spectral fingerprint", 4, "#ECF8F1"),
    ]
    row2 = [
        ("Smartphone\nimaging", "controlled acquisition", 5, "#FFF4E2"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", 6, "#EAF6FF"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", 7, "#F8F8F8"),
        ("Borax\nprediction", "quantitative output", 8, "#ECFAEC"),
    ]

    # Wider boxes and larger gaps; no internal collision.
    xs = [0.095, 0.305, 0.515, 0.725]
    w, h = 0.178, 0.205
    y1, y2 = 0.650, 0.220
    for i, (title, sub, step, fc) in enumerate(row1):
        _draw_q1_box(ax, (xs[i], y1), (w, h), title, sub, step=step, fc=fc)
    for i, (title, sub, step, fc) in enumerate(row2):
        _draw_q1_box(ax, (xs[i], y2), (w, h), title, sub, step=step, fc=fc)
    for i in range(3):
        _arrow_between(ax, xs[i] + w + 0.012, y1 + h/2, xs[i+1] - 0.012, y1 + h/2, lw=1.15, ms=10)
        _arrow_between(ax, xs[i] + w + 0.012, y2 + h/2, xs[i+1] - 0.012, y2 + h/2, lw=1.15, ms=10)

    ax.text(0.525, 0.055,
            "Borax-induced alkalization generates UV–Vis-validated smartphone color signatures for explainable prediction",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="0.02")

    # Lower evidence panels: tighter, more balanced composition.
    ax1 = fig.add_axes([0.070, 0.140, 0.290, 0.310])
    ax2 = fig.add_axes([0.395, 0.140, 0.395, 0.310])
    ax3 = fig.add_axes([0.835, 0.140, 0.130, 0.310])

    ax1.text(-0.12, 1.08, "a", transform=ax1.transAxes, fontsize=13, fontweight="bold")
    ax1.set_title("Smartphone color scale", fontsize=11.2, pad=8)
    concs = [c for c in ALL_CONC if c in summary.index]
    if concs:
        arr = np.array([[_rgb01_from_series(summary.loc[c]) for c in concs]])
        ax1.imshow(arr, aspect="auto", extent=[min(concs), max(concs), 0, 1])
        for c in concs:
            ax1.axvline(c, color="white", lw=0.45, alpha=0.68)
        ax1.set_xlim(min(concs), max(concs))
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_xlabel("Borax concentration (mM)", fontsize=8.8, labelpad=12)
    for s in ax1.spines.values():
        s.set_visible(False)

    ax2.text(-0.095, 1.08, "b", transform=ax2.transAxes, fontsize=13, fontweight="bold")
    ax2.set_title("Representative UV–Vis fingerprints", fontsize=11.2, pad=8)
    if uv_spectra is not None and not uv_spectra.empty:
        for c in [0, 5, 10, 15, 20]:
            sub = uv_spectra[np.isclose(uv_spectra["concentration_mM"], c)]
            if not sub.empty:
                g = sub.groupby("wavelength_nm", as_index=False)["absorbance"].mean()
                ax2.plot(g["wavelength_nm"], g["absorbance"], lw=1.35, label=f"{c} mM")
        ax2.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax2.set_xlabel("Wavelength (nm)", fontsize=9.2)
    ax2.set_ylabel("Absorbance", fontsize=9.2)
    ax2.grid(True, alpha=0.16, lw=0.5)

    ax3.text(-0.22, 1.08, "c", transform=ax3.transAxes, fontsize=13, fontweight="bold")
    ax3.set_title("Prediction workflow", fontsize=11.2, pad=8)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    steps = [("Color features", "RGB, HSV, CIELab"), ("Validation", "Set A ↔ Set B"),
             ("Attribution", "Permutation / SHAP"), ("Prediction", "mM output")]
    ys = [0.80, 0.58, 0.36, 0.14]
    box_x, box_w, box_h = 0.05, 0.90, 0.135
    for (title, sub), y in zip(steps, ys):
        ax3.add_patch(FancyBboxPatch((box_x, y), box_w, box_h,
                                     boxstyle="round,pad=0.014,rounding_size=0.018",
                                     fc="white", ec="0.25", lw=0.85))
        ax3.text(0.50, y+0.086, title, ha="center", va="center", fontsize=7.8, fontweight="bold", color="0.02")
        ax3.text(0.50, y+0.038, sub, ha="center", va="center", fontsize=6.7, color="0.25")
    # Gap-only arrows, never crossing text.
    for y_top, y_bot in [(0.80,0.58),(0.58,0.36),(0.36,0.14)]:
        _arrow_between(ax3, 0.50, y_top-0.006, 0.50, y_bot+box_h+0.008, lw=1.0, ms=8)

    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)


def _draw_species_card_symbol(ax, x, y, w, h, color, symbol, title, subtitle, note):
    """Species card without pseudo-chemical hexagons; uses symbol + color chip only."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.020",
                                fc="white", ec="0.64", lw=0.90))
    # Strong color chip retained as requested.
    ax.add_patch(Rectangle((x + 0.030, y + h - 0.090), w - 0.060, 0.052,
                           fc=color, ec="0.25", lw=0.45))
    # Symbol instead of fake structure.
    ax.text(x + w/2, y + h*0.55, symbol,
            ha="center", va="center", fontsize=20, fontweight="bold", color="0.08")
    ax.text(x + w/2, y + 0.112, title,
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="0.02")
    ax.text(x + w/2, y + 0.072, subtitle,
            ha="center", va="center", fontsize=7.1, color="0.25")
    ax.text(x + w/2, y + 0.035, note,
            ha="center", va="center", fontsize=6.6, color="0.45")


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Q1-clean mechanistic figure: no bad chemistry cartoons and no arrow/text collisions."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(18.0, 7.0), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 fontsize=18, fontweight="bold", y=0.970)

    axA = fig.add_axes([0.055, 0.130, 0.400, 0.730])
    axB = fig.add_axes([0.525, 0.145, 0.245, 0.685])
    axC = fig.add_axes([0.830, 0.155, 0.125, 0.665])

    # Panel A: clean symbolic transformation, no concentration arrow.
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.axis("off")
    axA.text(-0.045, 1.025, "a", transform=axA.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axA.set_title("pH-driven anthocyanin transformation", fontsize=12.5, fontweight="bold", pad=10)

    axA.text(0.50, 0.835, "Increasing borax concentration increases alkalinity",
             ha="center", va="center", fontsize=9.4, fontweight="bold", color="0.08")

    card_y, card_h, card_w = 0.305, 0.405, 0.255
    xs = [0.055, 0.372, 0.690]
    _draw_species_card_symbol(axA, xs[0], card_y, card_w, card_h,
                              "#8A59A5", "AH⁺", "Flavylium", "cationic / purple", "low-pH dominant")
    _draw_species_card_symbol(axA, xs[1], card_y, card_w, card_h,
                              "#326EB6", "A", "Quinoidal", "neutral / blue", "deprotonated form")
    _draw_species_card_symbol(axA, xs[2], card_y, card_w, card_h,
                              "#2B8A78", "A⁻ / C", "Chalcone-like", "open-ring / green-blue", "alkaline form")

    # Transformation symbols between cards, not arrows.
    axA.text((xs[0]+card_w+xs[1])/2, card_y + card_h*0.58, "⇌",
             ha="center", va="center", fontsize=18, fontweight="bold", color="0.20")
    axA.text((xs[1]+card_w+xs[2])/2, card_y + card_h*0.58, "⇌",
             ha="center", va="center", fontsize=18, fontweight="bold", color="0.20")

    axA.text(0.50, 0.075,
             "Simplified species notation for natural anthocyanin extract; detailed substituents may vary.",
             ha="center", va="center", fontsize=7.5, color="0.38")

    # Panel B unchanged in concept, with compact stats.
    axB.text(-0.19, 1.04, "b", transform=axB.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axB.set_title("pH–hue coupling", fontsize=12.5, fontweight="bold", pad=10)
    valid = summary.dropna(subset=["pH", "H_mean"])
    if not valid.empty and len(valid) >= 4:
        x = valid["pH"].astype(float).values
        y = valid["H_mean"].astype(float).values
        c = valid.index.astype(float).values
        order = 2 if len(x) >= 6 else 1
        coef = np.polyfit(x, y, order)
        xs_fit = np.linspace(np.nanmin(x), np.nanmax(x), 220)
        ys_fit = np.polyval(coef, xs_fit)
        resid = y - np.polyval(coef, x)
        band = 1.65 * np.nanstd(resid) if np.isfinite(np.nanstd(resid)) else 1.0
        axB.fill_between(xs_fit, ys_fit-band, ys_fit+band, color="0.88", alpha=1.0, lw=0, zorder=0)
        axB.plot(xs_fit, ys_fit, color="0.15", lw=2.0, zorder=1)
        sc = axB.scatter(x, y, c=c, cmap="viridis", s=46, edgecolor="black", lw=0.55, zorder=3)
        if color_df is not None and not color_df.empty and "H_mean" in color_df.columns:
            tmp = color_df.copy()
            tmp["pH"] = tmp["concentration_mM"].map(PH_MAP) if "pH" not in tmp.columns else tmp["pH"]
            sd = tmp.groupby("concentration_mM")["H_mean"].std()
            for conc, row in valid.iterrows():
                if conc in sd.index and np.isfinite(sd.loc[conc]):
                    axB.errorbar(row["pH"], row["H_mean"], yerr=sd.loc[conc], fmt="none",
                                 ecolor="0.35", elinewidth=0.8, capsize=2, zorder=2)
        for conc in [0,5,10,15,20]:
            if conc in valid.index:
                axB.text(valid.loc[conc,"pH"]+0.018, valid.loc[conc,"H_mean"]+0.55, str(conc), fontsize=7.8)
        try:
            from scipy.stats import pearsonr, spearmanr
            rr, pp = pearsonr(x, y)
            rho, sp = spearmanr(x, y)
            txt = f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\np < 0.001" if pp < 0.001 else f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\np = {pp:.3g}"
        except Exception:
            rr = np.corrcoef(x,y)[0,1]
            txt = f"r = {rr:.3f}\nR² = {rr**2:.3f}"
        axB.text(0.955,0.955,txt,transform=axB.transAxes,ha="right",va="top",fontsize=8.2,
                 bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="0.60",alpha=0.92))
        cb = fig.colorbar(sc, ax=axB, fraction=0.033, pad=0.015)
        cb.set_label("mM", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axB.set_xlabel("Measured pH", fontsize=10)
    axB.set_ylabel("HSV hue angle (°)", fontsize=10)
    axB.grid(True, alpha=0.18, lw=0.5)

    # Panel C: no arrow. Clean color trajectory with endpoints and hue readout.
    axC.text(-0.24, 1.04, "c", transform=axC.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axC.set_title("Color trajectory", fontsize=12.5, fontweight="bold", pad=10)
    axC.set_xlim(0,1)
    axC.set_ylim(0,1)
    axC.axis("off")
    means = summary.reindex([c for c in ALL_CONC if c in summary.index]).dropna(subset=["R_mean","G_mean","B_mean"])
    if not means.empty:
        n = len(means)
        x0, y0, w, h = 0.28, 0.25, 0.36, 0.58
        for i, (conc, row) in enumerate(means.iterrows()):
            yy = y0 + h*(1-(i+1)/n)
            axC.add_patch(Rectangle((x0, yy), w, h/n, fc=_rgb01_from_series(row), ec="white", lw=0.35))
        axC.text(x0-0.055, y0+h+0.045, "mM", ha="right", va="center", fontsize=8.8, fontweight="bold")
        label_concs = [0,5,10,15,20]
        for conc in label_concs:
            if conc in means.index:
                idx = list(means.index).index(conc)
                yy = y0 + h*(1-(idx+0.5)/n)
                hue = means.loc[conc,"H_mean"] if "H_mean" in means.columns else np.nan
                axC.text(x0-0.050, yy, f"{int(conc)}", ha="right", va="center", fontsize=8.4, fontweight="bold")
                axC.plot([x0-0.030,x0-0.006],[yy,yy], color="0.28", lw=0.75)
                if np.isfinite(hue):
                    axC.text(x0+w+0.058, yy, f"H = {hue:.1f}°", ha="left", va="center", fontsize=8.0)
        # Clear direction statement without an arrow colliding with text.
        axC.text(x0+w/2, y0+h+0.015, "0 → 20 mM", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="0.15")
        axC.text(x0+w/2, y0-0.062, "mean ROI color", ha="center", fontsize=8.0)
    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)



# -----------------------------------------------------------------------------
# FINAL OVERRIDE: Figure 2 redesigned for cleaner Q1 top-tier presentation
# - Smaller symbolic anthocyanin species cards
# - Smaller symbols AH+, A, A-/C
# - Lighter confidence band
# - Smaller colorbar
# - Panel C simplified to a full horizontal color trajectory without redundant H labels
# -----------------------------------------------------------------------------
def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from scipy.stats import pearsonr, spearmanr
    from scipy.optimize import curve_fit

    df = color_df.copy()

    # Robust pH handling
    if "pH" not in df.columns:
        conc_col = "concentration_mM" if "concentration_mM" in df.columns else "conc_mM"
        df["pH"] = df[conc_col].map(PH_MAP)

    # Robust hue/RGB column handling
    conc_col = "concentration_mM" if "concentration_mM" in df.columns else "conc_mM"
    hue_candidates = ["H_mean", "H", "Hue", "HSV_H", "HSV_H_mean", "hue_mean"]
    hue_col = next((c for c in hue_candidates if c in df.columns), None)
    if hue_col is None:
        # try any numeric hue-like column
        hue_col = next((c for c in df.columns if "hue" in c.lower() or c.lower().startswith("h_")), None)
    if hue_col is None:
        raise KeyError("No hue column found for Figure 2. Expected H_mean/Hue/HSV_H-like column.")

    # Mean by concentration, avoid flattened-column-name mismatch
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = [conc_col, "pH", hue_col] + [c for c in ["R_mean", "G_mean", "B_mean", "R", "G", "B"] if c in df.columns]
    keep = [c for c in keep if c in df.columns]
    summary = df[keep].groupby(conc_col).mean(numeric_only=True).sort_index()

    # Normalize RGB column names for color strips
    rgb_map = {}
    for base in ["R", "G", "B"]:
        if f"{base}_mean" in summary.columns:
            rgb_map[base] = f"{base}_mean"
        elif base in summary.columns:
            rgb_map[base] = base
    if len(rgb_map) < 3:
        # fallback from existing helper if available
        try:
            agg = aggregate_color(df)
            for base in ["R", "G", "B"]:
                if f"{base}_mean" in agg.columns:
                    summary[f"{base}_mean"] = agg[f"{base}_mean"]
                    rgb_map[base] = f"{base}_mean"
        except Exception:
            pass

    pH = summary["pH"].astype(float).values
    hue = summary[hue_col].astype(float).values
    conc = summary.index.astype(float).values
    valid = np.isfinite(pH) & np.isfinite(hue)
    pH, hue, conc = pH[valid], hue[valid], conc[valid]

    # Figure layout: more compact panel A and C, data-focused panel B
    fig = plt.figure(figsize=(15.8, 6.4), dpi=300)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.15, 1.05, 0.62],
        left=0.045, right=0.985, top=0.82, bottom=0.15,
        wspace=0.24
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry", fontsize=19, fontweight="bold", y=0.96)

    # Panel A: symbolic species cards, no pseudo-chemical hexagons
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.axis("off")
    axA.text(-0.055, 1.035, "a", transform=axA.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axA.set_title("Borax-induced anthocyanin species shift", fontsize=12.5, fontweight="bold", pad=12)
    axA.text(0.50, 0.90, "Increasing borax concentration increases alkalinity", ha="center", va="center", fontsize=10.5, fontweight="bold", color="0.15")

    species = [
        ("AH$^+$", "Flavylium", "cationic / purple", "low-pH dominant", "#8A59A5"),
        ("A", "Quinoidal", "neutral / blue", "deprotonated form", "#2F6DB2"),
        ("A$^-$ / C", "Chalcone-like", "open-ring / green-blue", "alkaline form", "#2A8C7A"),
    ]
    card_w, card_h = 0.245, 0.42
    xs = [0.08, 0.38, 0.68]
    y0 = 0.33
    for i, (sym, title, sub, note, col) in enumerate(species):
        x0 = xs[i]
        axA.add_patch(FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            fc="white", ec="0.70", lw=0.95
        ))
        # Color swatch is the visual anchor, not pseudo-structure
        axA.add_patch(Rectangle((x0+0.035, y0+card_h-0.105), card_w-0.070, 0.055, fc=col, ec="0.45", lw=0.4))
        axA.text(x0+card_w/2, y0+0.250, sym, ha="center", va="center", fontsize=22, fontweight="bold", color="0.08")
        axA.text(x0+card_w/2, y0+0.140, title, ha="center", va="center", fontsize=10.0, fontweight="bold")
        axA.text(x0+card_w/2, y0+0.098, sub, ha="center", va="center", fontsize=8.3, color="0.30")
        axA.text(x0+card_w/2, y0+0.058, note, ha="center", va="center", fontsize=7.5, color="0.42")
        if i < 2:
            axA.annotate("", xy=(xs[i+1]-0.030, y0+card_h*0.52), xytext=(x0+card_w+0.024, y0+card_h*0.52),
                         arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.25", shrinkA=0, shrinkB=0))
            axA.text((x0+card_w+xs[i+1])/2, y0+card_h*0.60, "pH ↑", ha="center", va="center", fontsize=8.2, color="0.35")
    axA.text(0.50, 0.17, "Simplified species notation for natural anthocyanin extract; detailed substituents may vary.",
             ha="center", fontsize=7.8, color="0.42")

    # Panel B: pH-hue coupling with lighter CI and compact stats
    axB.text(-0.14, 1.035, "b", transform=axB.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axB.set_title("pH–hue coupling", fontsize=12.5, fontweight="bold", pad=8)

    if len(pH) >= 4:
        def quad(x, a, b, c):
            return a*x*x + b*x + c
        order = np.argsort(pH)
        xsfit = np.linspace(np.nanmin(pH), np.nanmax(pH), 250)
        try:
            popt, pcov = curve_fit(quad, pH, hue, maxfev=20000)
            yfit = quad(xsfit, *popt)
            yhat = quad(pH, *popt)
            # residual-based approximate CI, intentionally light and not visually dominant
            resid = hue - yhat
            s = np.nanstd(resid, ddof=min(3, max(1, len(resid)-1))) if len(resid) > 3 else np.nanstd(resid)
            ci = 1.20 * s if np.isfinite(s) else 1.5
            axB.fill_between(xsfit, yfit-ci, yfit+ci, color="0.82", alpha=0.45, lw=0, zorder=1)
            axB.plot(xsfit, yfit, color="0.12", lw=1.9, zorder=3)
        except Exception:
            coeff = np.polyfit(pH, hue, deg=2)
            yfit = np.polyval(coeff, xsfit)
            axB.plot(xsfit, yfit, color="0.12", lw=1.9, zorder=3)

    sc = axB.scatter(pH, hue, c=conc, cmap="viridis", s=44, edgecolor="k", linewidth=0.45, zorder=4)
    for target in [0, 5, 10, 15, 20]:
        if target in conc:
            idx = np.where(conc == target)[0][0]
            axB.text(pH[idx]+0.025, hue[idx]+0.55, str(target), fontsize=8.4, color="0.15")

    try:
        rr, pp = pearsonr(pH, hue)
        rho, psp = spearmanr(pH, hue)
        r2 = rr**2
        ptxt = "p < 0.001" if pp < 0.001 else f"p = {pp:.3f}"
        axB.text(0.965, 0.965, f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {r2:.3f}\n{ptxt}", transform=axB.transAxes,
                 ha="right", va="top", fontsize=8.4,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.65", alpha=0.93))
    except Exception:
        pass

    cb = fig.colorbar(sc, ax=axB, fraction=0.026, pad=0.012)
    cb.set_label("mM", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    axB.set_xlabel("Measured pH", fontsize=10)
    axB.set_ylabel("HSV hue angle (°)", fontsize=10)
    axB.grid(True, alpha=0.16, lw=0.5)

    # Panel C: horizontal color trajectory; no redundant labels that collide
    axC.text(-0.18, 1.035, "c", transform=axC.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axC.set_title("Color trajectory", fontsize=12.5, fontweight="bold", pad=8)
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.axis("off")

    means = summary.copy()
    if len(rgb_map) == 3:
        means = means.reindex([c for c in ALL_CONC if c in means.index]).dropna(subset=[rgb_map["R"], rgb_map["G"], rgb_map["B"]])
        if not means.empty:
            n = len(means)
            x0, y0c, w, h = 0.10, 0.47, 0.80, 0.18
            for i, (cc, row) in enumerate(means.iterrows()):
                xx = x0 + w*i/n
                rgb = np.array([row[rgb_map["R"]], row[rgb_map["G"]], row[rgb_map["B"]]], dtype=float)
                if np.nanmax(rgb) > 1.5:
                    rgb = np.clip(rgb/255.0, 0, 1)
                else:
                    rgb = np.clip(rgb, 0, 1)
                axC.add_patch(Rectangle((xx, y0c), w/n, h, fc=rgb, ec="white", lw=0.25))
            axC.text(x0, y0c+h+0.095, "0 mM", ha="left", va="center", fontsize=9, fontweight="bold")
            axC.text(x0+w, y0c+h+0.095, "20 mM", ha="right", va="center", fontsize=9, fontweight="bold")
            axC.annotate("", xy=(x0+w, y0c+h+0.055), xytext=(x0, y0c+h+0.055),
                         arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.2"))
            axC.text(x0+w/2, y0c-0.085, "mean ROI color from smartphone images", ha="center", fontsize=8.2, color="0.35")
            # Hue endpoints only, enough to connect to panel B without clutter
            try:
                h0 = means.loc[means.index.min(), hue_col]
                h1 = means.loc[means.index.max(), hue_col]
                axC.text(x0, y0c-0.18, f"Hue {h0:.1f}°", ha="left", fontsize=8.4, color="0.20")
                axC.text(x0+w, y0c-0.18, f"{h1:.1f}°", ha="right", fontsize=8.4, color="0.20")
            except Exception:
                pass
    else:
        axC.text(0.5, 0.5, "RGB color trajectory unavailable", ha="center", va="center", fontsize=9, color="0.35")

    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)



# -----------------------------------------------------------------------------
# FINAL OVERRIDE: Figure 1 redesigned after final review
# - Boxes are narrower but taller for efficient typography
# - No black number markers; step labels are small and unobtrusive
# - All internal text uses readable dark weights, not thin grey text
# - Lower panels use tighter spacing with less empty gap
# - Panel C arrows are located only in gaps and never cross text
# -----------------------------------------------------------------------------
def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(17.2, 8.1), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Multimodal smartphone colorimetry for anthocyanin-assisted borax quantification",
        fontsize=17.2, fontweight="bold", y=0.965
    )

    # ------------------------ Upper workflow ------------------------
    ax = fig.add_axes([0.055, 0.590, 0.890, 0.315])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def clean_box(x, y, w, h, step, title, subtitle, fc):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            fc=fc, ec="0.18", lw=1.0
        ))
        # small unobtrusive step label
        ax.text(x + 0.018, y + h - 0.030, f"Step {step}",
                ha="left", va="top", fontsize=7.2, fontweight="bold", color="0.22")
        # larger vertical spacing between title and subtitle
        ax.text(x + w/2, y + h*0.610, title,
                ha="center", va="center", fontsize=9.3, fontweight="bold",
                color="0.00", linespacing=0.92)
        ax.text(x + w/2, y + h*0.265, subtitle,
                ha="center", va="center", fontsize=7.45, fontweight="semibold",
                color="0.20", linespacing=0.95)

    def arr(x1, y1, x2, y2, ms=9):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="0.22", lw=1.15,
                                    mutation_scale=ms, shrinkA=0, shrinkB=0))

    # Two scientific layers, label outside the boxes with enough clearance.
    ax.text(0.045, 0.715, "Chemical\nvalidation", rotation=90,
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="0.10", linespacing=0.88)
    ax.text(0.045, 0.300, "Digital\nquantification", rotation=90,
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="0.10", linespacing=0.88)

    # Subtle separator and concise bridge text.
    ax.plot([0.095, 0.945], [0.505, 0.505], ls=(0, (3, 4)), lw=0.85, color="0.76")
    ax.text(0.520, 0.518, "chemical validation supports smartphone-based quantification",
            ha="center", va="bottom", fontsize=7.8, color="0.30", fontweight="semibold")

    # Narrower but taller boxes.
    w, h = 0.157, 0.168
    xs = [0.115, 0.335, 0.555, 0.775]
    y_top, y_bot = 0.635, 0.225
    top = [
        (1, "Borax\nstandard", "alkalization", "#EAF4FF"),
        (2, "Purple cabbage\nanthocyanin", "natural pH probe", "#F2EAFB"),
        (3, "Anthocyanin\nconversion", "pH-responsive forms", "#F8F8F8"),
        (4, "UV–Vis\nvalidation", "spectral fingerprint", "#ECF8F1"),
    ]
    bot = [
        (5, "Smartphone\nimaging", "controlled acquisition", "#FFF4E2"),
        (6, "ROI + color\nfeatures", "RGB–HSV–CIELab", "#EAF6FF"),
        (7, "Chemometrics\n+ XAI", "PCA, ML, SHAP", "#F8F8F8"),
        (8, "Borax\nprediction", "quantitative output", "#ECFAEC"),
    ]
    for x, item in zip(xs, top):
        clean_box(x, y_top, w, h, *item)
    for x, item in zip(xs, bot):
        clean_box(x, y_bot, w, h, *item)

    # Clear arrows located between boxes only.
    for i in range(3):
        arr(xs[i] + w + 0.010, y_top + h/2, xs[i+1] - 0.010, y_top + h/2)
        arr(xs[i] + w + 0.010, y_bot + h/2, xs[i+1] - 0.010, y_bot + h/2)

    ax.text(0.520, 0.060,
            "Borax-induced alkalization generates UV–Vis-validated smartphone color signatures for explainable prediction",
            ha="center", va="center", fontsize=10.1, fontweight="bold", color="0.02")

    # ------------------------ Lower evidence panels ------------------------
    # Reduced wspace and better proportions: color scale + UV–Vis + compact prediction chain.
    ax1 = fig.add_axes([0.055, 0.120, 0.300, 0.325])
    ax2 = fig.add_axes([0.385, 0.120, 0.430, 0.325])
    ax3 = fig.add_axes([0.840, 0.120, 0.135, 0.325])

    ax1.text(-0.120, 1.080, "a", transform=ax1.transAxes, fontsize=13, fontweight="bold")
    ax1.set_title("Smartphone color scale", fontsize=11.2, pad=8)
    concs = [c for c in ALL_CONC if c in summary.index]
    if concs:
        arr_rgb = np.array([[_rgb01_from_series(summary.loc[c]) for c in concs]])
        ax1.imshow(arr_rgb, aspect="auto", extent=[min(concs), max(concs), 0, 1])
        for c in concs:
            ax1.axvline(c, color="white", lw=0.42, alpha=0.68)
        ax1.set_xlim(min(concs), max(concs))
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.tick_params(axis="x", labelsize=8.2)
    ax1.set_xlabel("Borax concentration (mM)", fontsize=8.8, labelpad=13)
    for sp in ax1.spines.values():
        sp.set_visible(False)

    ax2.text(-0.095, 1.080, "b", transform=ax2.transAxes, fontsize=13, fontweight="bold")
    ax2.set_title("Representative UV–Vis fingerprints", fontsize=11.2, pad=8)
    if uv_spectra is not None and not uv_spectra.empty:
        for c in [0, 5, 10, 15, 20]:
            sub = uv_spectra[np.isclose(uv_spectra["concentration_mM"], c)]
            if not sub.empty:
                g = sub.groupby("wavelength_nm", as_index=False)["absorbance"].mean()
                ax2.plot(g["wavelength_nm"], g["absorbance"], lw=1.35, label=f"{c} mM")
        ax2.legend(frameon=False, fontsize=7.3, loc="upper right")
    ax2.set_xlabel("Wavelength (nm)", fontsize=9.2)
    ax2.set_ylabel("Absorbance", fontsize=9.2)
    ax2.grid(True, alpha=0.16, lw=0.5)

    ax3.text(-0.230, 1.080, "c", transform=ax3.transAxes, fontsize=13, fontweight="bold")
    ax3.set_title("Prediction workflow", fontsize=11.2, pad=8)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    steps = [
        ("Color features", "RGB, HSV, CIELab"),
        ("Validation", "Set A ↔ Set B"),
        ("Attribution", "Permutation / SHAP"),
        ("Prediction", "mM output"),
    ]
    ys = [0.785, 0.565, 0.345, 0.125]
    bx, bw, bh = 0.035, 0.930, 0.125
    for (title, sub), y in zip(steps, ys):
        ax3.add_patch(FancyBboxPatch((bx, y), bw, bh,
                                     boxstyle="round,pad=0.014,rounding_size=0.017",
                                     fc="white", ec="0.25", lw=0.85))
        ax3.text(0.50, y + bh*0.66, title, ha="center", va="center",
                 fontsize=7.6, fontweight="bold", color="0.02")
        ax3.text(0.50, y + bh*0.30, sub, ha="center", va="center",
                 fontsize=6.5, fontweight="semibold", color="0.26")

    # arrows strictly between boxes, never crossing the text inside the boxes
    for i in range(3):
        y_from = ys[i] - 0.012
        y_to = ys[i+1] + bh + 0.012
        ax3.annotate("", xy=(0.50, y_to), xytext=(0.50, y_from),
                     arrowprops=dict(arrowstyle="-|>", lw=0.90, color="0.25", mutation_scale=7.0))

    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)




# =====================================================================
# ULTRA-CLEAN BALANCED OVERRIDES FOR FINAL Q1 FIGURE 1 AND FIGURE 2
# - Figure 1: compact-but-taller boxes, no text overlap, balanced lower panels.
# - Figure 2: symbolic species cards only; no chemical cartoons or pH arrows
#             colliding with cards; balanced A/B/C proportions.
# =====================================================================

def _draw_q1_box(ax, xy, wh, title, subtitle="", step=None, fc="#F7F7F7", ec="0.18"):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        fc=fc, ec=ec, lw=1.05, zorder=2
    ))
    if step is not None:
        ax.text(x + 0.014, y + h - 0.022, f"Step {step}",
                ha="left", va="top", fontsize=6.8,
                color="0.22", fontweight="bold", zorder=4)
    ax.text(x + w/2, y + h*0.59, title,
            ha="center", va="center", fontsize=8.9,
            fontweight="bold", color="0.02", linespacing=0.88, zorder=4)
    if subtitle:
        ax.text(x + w/2, y + h*0.245, subtitle,
                ha="center", va="center", fontsize=7.0,
                color="0.20", fontweight="semibold", linespacing=0.94, zorder=4)

def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Balanced Figure 1: taller boxes, compact width, clear text, proportional panels."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(17.5, 8.2), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Multimodal smartphone colorimetry for anthocyanin-assisted borax quantification",
                 fontsize=18, fontweight="bold", y=0.965)

    # ---------------- workflow ----------------
    ax = fig.add_axes([0.075, 0.615, 0.850, 0.285])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.020, 0.725, "Chemical\nvalidation", rotation=90,
            ha="center", va="center", fontsize=8.7, fontweight="bold", color="0.14", linespacing=0.88)
    ax.text(0.020, 0.295, "Digital\nquantification", rotation=90,
            ha="center", va="center", fontsize=8.7, fontweight="bold", color="0.14", linespacing=0.88)

    ax.plot([0.070, 0.970], [0.505, 0.505], ls=(0, (4, 4)), lw=0.75, color="0.78")
    ax.text(0.520, 0.522, "chemical validation supports smartphone-based quantification",
            ha="center", va="bottom", fontsize=7.6, color="0.30", fontweight="semibold")

    # narrower and taller boxes: more efficient than long horizontal boxes
    box_w, box_h = 0.150, 0.235
    xs = [0.105, 0.335, 0.565, 0.795]
    y_top, y_bot = 0.635, 0.205

    top = [
        ("Borax\nstandard", "alkalization", 1, "#EAF4FF"),
        ("Purple cabbage\nanthocyanin", "natural pH probe", 2, "#F2EAFB"),
        ("Anthocyanin\nconversion", "pH-responsive forms", 3, "#F8F8F8"),
        ("UV–Vis\nvalidation", "spectral fingerprint", 4, "#ECF8F1"),
    ]
    bottom = [
        ("Smartphone\nimaging", "controlled acquisition", 5, "#FFF4E2"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", 6, "#EAF6FF"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", 7, "#F8F8F8"),
        ("Borax\nprediction", "quantitative output", 8, "#ECFAEC"),
    ]
    for i, item in enumerate(top):
        _draw_q1_box(ax, (xs[i], y_top), (box_w, box_h), item[0], item[1], step=item[2], fc=item[3])
    for i, item in enumerate(bottom):
        _draw_q1_box(ax, (xs[i], y_bot), (box_w, box_h), item[0], item[1], step=item[2], fc=item[3])

    # arrows in the gaps only
    for i in range(3):
        ax.annotate("", xy=(xs[i+1]-0.012, y_top + box_h/2), xytext=(xs[i]+box_w+0.012, y_top + box_h/2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.15, color="0.25", mutation_scale=9))
        ax.annotate("", xy=(xs[i+1]-0.012, y_bot + box_h/2), xytext=(xs[i]+box_w+0.012, y_bot + box_h/2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.15, color="0.25", mutation_scale=9))

    ax.text(0.520, 0.045,
            "Borax-induced alkalization generates UV–Vis-validated smartphone color signatures for explainable prediction",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="0.02")

    # ---------------- lower panels: proportional; UV-Vis no longer dominates ----------------
    ax1 = fig.add_axes([0.070, 0.130, 0.300, 0.310])
    ax2 = fig.add_axes([0.410, 0.130, 0.335, 0.310])
    ax3 = fig.add_axes([0.790, 0.130, 0.185, 0.310])

    ax1.text(-0.115, 1.08, "a", transform=ax1.transAxes, fontsize=13, fontweight="bold")
    ax1.set_title("Smartphone color scale", fontsize=11.2, pad=8)
    concs = [c for c in ALL_CONC if c in summary.index]
    if concs:
        arr_rgb = np.array([[_rgb01_from_series(summary.loc[c]) for c in concs]])
        ax1.imshow(arr_rgb, aspect="auto", extent=[min(concs), max(concs), 0, 1])
        for c in concs:
            ax1.axvline(c, color="white", lw=0.42, alpha=0.68)
        ax1.set_xlim(min(concs), max(concs))
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.tick_params(axis="x", labelsize=8.2)
    ax1.set_xlabel("Borax concentration (mM)", fontsize=8.8, labelpad=13)
    for sp in ax1.spines.values():
        sp.set_visible(False)

    ax2.text(-0.095, 1.08, "b", transform=ax2.transAxes, fontsize=13, fontweight="bold")
    ax2.set_title("Representative UV–Vis fingerprints", fontsize=11.2, pad=8)
    if uv_spectra is not None and not uv_spectra.empty:
        for c in [0, 5, 10, 15, 20]:
            sub = uv_spectra[np.isclose(uv_spectra["concentration_mM"], c)]
            if not sub.empty:
                g = sub.groupby("wavelength_nm", as_index=False)["absorbance"].mean()
                ax2.plot(g["wavelength_nm"], g["absorbance"], lw=1.30, label=f"{c} mM")
        ax2.legend(frameon=False, fontsize=7.1, loc="upper right")
    ax2.set_xlabel("Wavelength (nm)", fontsize=9.0)
    ax2.set_ylabel("Absorbance", fontsize=9.0)
    ax2.grid(True, alpha=0.16, lw=0.5)

    ax3.text(-0.16, 1.08, "c", transform=ax3.transAxes, fontsize=13, fontweight="bold")
    ax3.set_title("Prediction workflow", fontsize=11.2, pad=8)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    steps = [("Color features", "RGB, HSV, CIELab"), ("Validation", "Set A ↔ Set B"),
             ("Attribution", "Permutation / SHAP"), ("Prediction", "mM output")]
    ys = [0.78, 0.56, 0.34, 0.12]
    bx, bw, bh = 0.10, 0.80, 0.125
    for (ttl, sub), y in zip(steps, ys):
        ax3.add_patch(FancyBboxPatch((bx, y), bw, bh, boxstyle="round,pad=0.014,rounding_size=0.017",
                                     fc="white", ec="0.25", lw=0.85))
        ax3.text(0.50, y + bh*0.65, ttl, ha="center", va="center", fontsize=7.6, fontweight="bold", color="0.02")
        ax3.text(0.50, y + bh*0.30, sub, ha="center", va="center", fontsize=6.5, fontweight="semibold", color="0.26")
    for i in range(3):
        y_from = ys[i] - 0.010
        y_to = ys[i+1] + bh + 0.012
        ax3.annotate("", xy=(0.50, y_to), xytext=(0.50, y_from),
                     arrowprops=dict(arrowstyle="-|>", lw=0.85, color="0.25", mutation_scale=6.5))

    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Balanced Figure 2 with symbolic species cards, no pH-arrow collisions, and balanced panel proportions."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(18.0, 7.1), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 fontsize=18, fontweight="bold", y=0.970)

    # More balanced A/B/C proportions
    axA = fig.add_axes([0.055, 0.150, 0.355, 0.690])
    axB = fig.add_axes([0.465, 0.140, 0.315, 0.705])
    axC = fig.add_axes([0.825, 0.190, 0.130, 0.600])

    # Panel A
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    axA.text(-0.055, 1.04, "a", transform=axA.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axA.set_title("Borax-induced anthocyanin species shift", fontsize=12.6, fontweight="bold", pad=10)
    axA.text(0.50, 0.82, "Increasing borax concentration increases alkalinity",
             ha="center", va="center", fontsize=9.4, fontweight="bold", color="0.12")

    card_w, card_h = 0.245, 0.400
    y_card = 0.310
    xs = [0.075, 0.378, 0.681]
    cards = [
        ("#8A59A5", "AH$^+$", "Flavylium", "cationic / purple", "low-pH dominant"),
        ("#326EB6", "A", "Quinoidal", "neutral / blue", "deprotonated form"),
        ("#2B8A78", "A$^-$ / C", "Chalcone-like", "open-ring / green-blue", "alkaline form"),
    ]
    for x0, (col, sym, ttl, st, note) in zip(xs, cards):
        axA.add_patch(FancyBboxPatch((x0, y_card), card_w, card_h,
                                     boxstyle="round,pad=0.012,rounding_size=0.018",
                                     fc="white", ec="0.65", lw=0.85))
        axA.add_patch(Rectangle((x0 + 0.040, y_card + card_h - 0.075), card_w - 0.080, 0.045,
                                fc=col, ec="0.25", lw=0.45))
        axA.text(x0 + card_w/2, y_card + card_h*0.55, sym,
                 ha="center", va="center", fontsize=20, fontweight="bold", color="0.03")
        axA.text(x0 + card_w/2, y_card + 0.105, ttl,
                 ha="center", va="center", fontsize=8.5, fontweight="bold", color="0.02")
        axA.text(x0 + card_w/2, y_card + 0.068, st,
                 ha="center", va="center", fontsize=7.0, color="0.28")
        axA.text(x0 + card_w/2, y_card + 0.032, note,
                 ha="center", va="center", fontsize=6.6, color="0.44")
    # small equilibrium symbols between cards; no pH arrows colliding with cards
    axA.text(xs[0] + card_w + 0.030, y_card + card_h*0.53, "⇌", ha="center", va="center", fontsize=13, color="0.25")
    axA.text(xs[1] + card_w + 0.030, y_card + card_h*0.53, "⇌", ha="center", va="center", fontsize=13, color="0.25")
    axA.text(0.50, 0.100, "Simplified species notation for natural anthocyanin extract; detailed substituents may vary.",
             ha="center", va="center", fontsize=7.2, color="0.40")

    # Panel B
    axB.text(-0.13, 1.04, "b", transform=axB.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axB.set_title("pH–hue coupling", fontsize=12.6, fontweight="bold", pad=10)
    valid = summary.dropna(subset=["pH", "H_mean"])
    if not valid.empty and len(valid) >= 4:
        x = valid["pH"].astype(float).values
        y = valid["H_mean"].astype(float).values
        c = valid.index.astype(float).values
        order = 2 if len(x) >= 6 else 1
        coef = np.polyfit(x, y, order)
        xs_fit = np.linspace(np.nanmin(x), np.nanmax(x), 220)
        ys_fit = np.polyval(coef, xs_fit)
        resid = y - np.polyval(coef, x)
        band = 0.92 * np.nanstd(resid) if np.isfinite(np.nanstd(resid)) else 1.0
        axB.fill_between(xs_fit, ys_fit-band, ys_fit+band, color="0.82", alpha=0.42, lw=0, zorder=0)
        axB.plot(xs_fit, ys_fit, color="0.15", lw=2.0, zorder=1)
        sc = axB.scatter(x, y, c=c, cmap="viridis", s=42, edgecolor="black", lw=0.50, zorder=3)
        for conc in [0, 5, 10, 15, 20]:
            if conc in valid.index:
                axB.text(valid.loc[conc, "pH"] + 0.018, valid.loc[conc, "H_mean"] + 0.55, str(conc), fontsize=7.4)
        try:
            from scipy.stats import pearsonr, spearmanr
            rr, pp = pearsonr(x, y); rho, _ = spearmanr(x, y)
            ptxt = "p < 0.001" if pp < 0.001 else f"p = {pp:.2g}"
            txt = f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\n{ptxt}"
        except Exception:
            rr = np.corrcoef(x, y)[0, 1]
            txt = f"r = {rr:.3f}\nR² = {rr**2:.3f}"
        axB.text(0.95, 0.955, txt, transform=axB.transAxes, ha="right", va="top", fontsize=8.0,
                 bbox=dict(boxstyle="round,pad=0.23", fc="white", ec="0.60", alpha=0.93))
        cb = fig.colorbar(sc, ax=axB, fraction=0.026, pad=0.010)
        cb.set_label("mM", fontsize=7.8)
        cb.ax.tick_params(labelsize=6.8)
    axB.set_xlabel("Measured pH", fontsize=10)
    axB.set_ylabel("HSV hue angle (°)", fontsize=10)
    axB.grid(True, alpha=0.18, lw=0.5)

    # Panel C: horizontal trajectory, no arrow/text collision
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    axC.text(-0.22, 1.04, "c", transform=axC.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axC.set_title("Color trajectory", fontsize=12.6, fontweight="bold", pad=10)
    means = summary.reindex([c for c in ALL_CONC if c in summary.index]).dropna(subset=["R_mean", "G_mean", "B_mean"])
    if not means.empty:
        concs = means.index.values.astype(float)
        arr_rgb = np.array([[_rgb01_from_series(means.loc[c]) for c in means.index]])
        x0, y0, w, h = 0.12, 0.47, 0.78, 0.15
        axC.imshow(arr_rgb, aspect="auto", extent=[x0, x0+w, y0, y0+h])
        n = len(means)
        for i in range(n+1):
            xx = x0 + w*i/n
            axC.plot([xx, xx], [y0, y0+h], color="white", lw=0.30, alpha=0.65)
        axC.annotate("", xy=(x0+w, y0+h+0.10), xytext=(x0, y0+h+0.10),
                     arrowprops=dict(arrowstyle="-|>", lw=1.05, color="0.20"))
        axC.text(x0, y0+h+0.135, "0 mM", ha="left", va="bottom", fontsize=8.5, fontweight="bold")
        axC.text(x0+w, y0+h+0.135, "20 mM", ha="right", va="bottom", fontsize=8.5, fontweight="bold")
        axC.text(x0+w/2, y0-0.085, "mean ROI color from smartphone images", ha="center", fontsize=7.8, color="0.35")
        h0 = means.loc[0, "H_mean"] if 0 in means.index and "H_mean" in means.columns else np.nan
        h20 = means.loc[20, "H_mean"] if 20 in means.index and "H_mean" in means.columns else np.nan
        if np.isfinite(h0) and np.isfinite(h20):
            axC.text(x0, y0-0.22, f"Hue {h0:.1f}°", ha="left", fontsize=8.0, color="0.25")
            axC.text(x0+w, y0-0.22, f"{h20:.1f}°", ha="right", fontsize=8.0, color="0.25")

    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)



# =============================================================================
# FINAL PUBLICATION-QUALITY OVERRIDES FOR FIGURE 1 AND FIGURE 2
# These definitions intentionally override earlier versions above.
# =============================================================================

def _draw_publication_workflow_box(ax, x, y, w, h, title, subtitle, fc):
    """Clean journal-style workflow box without step badges."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        fc=fc, ec="0.16", lw=1.00
    ))
    # Split title into controlled lines and keep enough vertical distance from subtitle
    title_lines = title.split("\n")
    if len(title_lines) == 1:
        ax.text(x+w/2, y+h*0.64, title, ha="center", va="center",
                fontsize=8.9, fontweight="bold", color="0.02")
    else:
        ax.text(x+w/2, y+h*0.70, title_lines[0], ha="center", va="center",
                fontsize=8.8, fontweight="bold", color="0.02")
        ax.text(x+w/2, y+h*0.52, "\n".join(title_lines[1:]), ha="center", va="center",
                fontsize=8.8, fontweight="bold", color="0.02", linespacing=0.90)
    ax.text(x+w/2, y+h*0.23, subtitle, ha="center", va="center",
            fontsize=7.2, fontweight="semibold", color="0.18")


def figure_1_workflow_graphical(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Final Figure 1: compact two-layer workflow with balanced data panels."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(17.2, 8.0), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Multimodal smartphone colorimetry for anthocyanin-assisted borax quantification",
                 fontsize=17.2, fontweight="bold", y=0.965)

    # Workflow block
    ax = fig.add_axes([0.075, 0.600, 0.850, 0.300])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Layer labels placed outside the process boxes with clear margin
    ax.text(0.023, 0.725, "Chemical\nvalidation", rotation=90,
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="0.12", linespacing=0.90)
    ax.text(0.023, 0.285, "Digital\nquantification", rotation=90,
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="0.12", linespacing=0.90)

    # Slim separator with readable statement
    ax.plot([0.070, 0.970], [0.505, 0.505], ls=(0, (4, 4)), lw=0.70, color="0.78")
    ax.text(0.520, 0.523, "chemical validation supports smartphone-based quantification",
            ha="center", va="bottom", fontsize=8.2, color="0.26", fontweight="bold")

    # Boxes: not too wide, taller; enough room for 3 clear text levels
    box_w, box_h = 0.165, 0.220
    xs = [0.105, 0.345, 0.585, 0.825]
    y_top, y_bot = 0.635, 0.205
    top = [
        ("Borax\nstandard", "alkalization", "#EAF4FF"),
        ("Purple cabbage\nanthocyanin", "natural pH probe", "#F2EAFB"),
        ("Anthocyanin\nconversion", "pH-responsive forms", "#F8F8F8"),
        ("UV–Vis\nvalidation", "spectral fingerprint", "#ECF8F1"),
    ]
    bottom = [
        ("Smartphone\nimaging", "controlled acquisition", "#FFF4E2"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", "#EAF6FF"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", "#F8F8F8"),
        ("Borax\nprediction", "quantitative output", "#ECFAEC"),
    ]
    for i, (ttl, sub, fc) in enumerate(top):
        _draw_publication_workflow_box(ax, xs[i], y_top, box_w, box_h, ttl, sub, fc)
    for i, (ttl, sub, fc) in enumerate(bottom):
        _draw_publication_workflow_box(ax, xs[i], y_bot, box_w, box_h, ttl, sub, fc)

    # Arrows only in inter-box gaps, not touching text
    for i in range(3):
        ax.annotate("", xy=(xs[i+1]-0.010, y_top+box_h/2), xytext=(xs[i]+box_w+0.010, y_top+box_h/2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.10, color="0.22", mutation_scale=8.5))
        ax.annotate("", xy=(xs[i+1]-0.010, y_bot+box_h/2), xytext=(xs[i]+box_w+0.010, y_bot+box_h/2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.10, color="0.22", mutation_scale=8.5))

    ax.text(0.520, 0.045,
            "Borax-induced alkalization generates UV–Vis-validated smartphone color signatures for explainable prediction",
            ha="center", va="center", fontsize=9.7, fontweight="bold", color="0.02")

    # Data panels: balanced proportions; UV-Vis is important but not visually overwhelming
    ax1 = fig.add_axes([0.075, 0.125, 0.300, 0.315])
    ax2 = fig.add_axes([0.420, 0.125, 0.340, 0.315])
    ax3 = fig.add_axes([0.805, 0.125, 0.165, 0.315])

    ax1.text(-0.105, 1.08, "a", transform=ax1.transAxes, fontsize=13, fontweight="bold")
    ax1.set_title("Smartphone color scale", fontsize=11.2, pad=8)
    concs = [c for c in ALL_CONC if c in summary.index]
    if concs:
        arr_rgb = np.array([[_rgb01_from_series(summary.loc[c]) for c in concs]])
        ax1.imshow(arr_rgb, aspect="auto", extent=[min(concs), max(concs), 0, 1])
        for c in concs:
            ax1.axvline(c, color="white", lw=0.42, alpha=0.65)
        ax1.set_xlim(min(concs), max(concs))
    ax1.set_ylim(0, 1); ax1.set_yticks([])
    ax1.set_xticks([0, 5, 10, 15, 20]); ax1.tick_params(axis="x", labelsize=8.0)
    ax1.set_xlabel("Borax concentration (mM)", fontsize=8.7, labelpad=12)
    for sp in ax1.spines.values(): sp.set_visible(False)

    ax2.text(-0.088, 1.08, "b", transform=ax2.transAxes, fontsize=13, fontweight="bold")
    ax2.set_title("Representative UV–Vis fingerprints", fontsize=11.2, pad=8)
    if uv_spectra is not None and not uv_spectra.empty:
        for c in [0, 5, 10, 15, 20]:
            sub = uv_spectra[np.isclose(uv_spectra["concentration_mM"], c)]
            if not sub.empty:
                g = sub.groupby("wavelength_nm", as_index=False)["absorbance"].mean()
                ax2.plot(g["wavelength_nm"], g["absorbance"], lw=1.22, label=f"{c} mM")
        ax2.legend(frameon=False, fontsize=7.0, loc="upper right")
    ax2.set_xlabel("Wavelength (nm)", fontsize=8.7)
    ax2.set_ylabel("Absorbance", fontsize=8.7)
    ax2.tick_params(labelsize=8.0)
    ax2.grid(True, alpha=0.14, lw=0.5)

    ax3.text(-0.14, 1.08, "c", transform=ax3.transAxes, fontsize=13, fontweight="bold")
    ax3.set_title("Prediction workflow", fontsize=11.2, pad=8)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    steps = [("Color features", "RGB, HSV, CIELab"), ("Validation", "Set A ↔ Set B"),
             ("Attribution", "Permutation / SHAP"), ("Prediction", "mM output")]
    ys = [0.77, 0.55, 0.33, 0.11]
    bx, bw, bh = 0.10, 0.80, 0.130
    for (ttl, sub), y in zip(steps, ys):
        ax3.add_patch(FancyBboxPatch((bx, y), bw, bh,
                                     boxstyle="round,pad=0.012,rounding_size=0.017",
                                     fc="white", ec="0.25", lw=0.85))
        ax3.text(0.50, y+bh*0.63, ttl, ha="center", va="center", fontsize=7.3, fontweight="bold")
        ax3.text(0.50, y+bh*0.30, sub, ha="center", va="center", fontsize=6.3, fontweight="semibold", color="0.28")
    for i in range(3):
        ax3.annotate("", xy=(0.50, ys[i+1]+bh+0.012), xytext=(0.50, ys[i]-0.008),
                     arrowprops=dict(arrowstyle="-|>", lw=0.82, color="0.25", mutation_scale=6.0))

    savefig(FIG_MAIN / "Fig01_workflow_schematic", tight=False)


def _species_card_final(ax, x, y, w, h, color, symbol, name, state, note):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.020",
                                fc="white", ec="0.70", lw=0.95))
    ax.add_patch(Rectangle((x+w*0.14, y+h*0.78), w*0.72, h*0.105,
                           fc=color, ec="0.35", lw=0.45))
    ax.text(x+w/2, y+h*0.55, symbol, ha="center", va="center",
            fontsize=24, fontweight="bold", color="0.03")
    ax.text(x+w/2, y+h*0.31, name, ha="center", va="center",
            fontsize=9.0, fontweight="bold", color="0.03")
    ax.text(x+w/2, y+h*0.205, state, ha="center", va="center",
            fontsize=7.5, color="0.23")
    ax.text(x+w/2, y+h*0.110, note, ha="center", va="center",
            fontsize=6.8, color="0.43")


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Final Figure 2: balanced mechanism, pH-hue coupling and color trajectory."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)
    fig = plt.figure(figsize=(17.2, 6.7), dpi=300)
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 fontsize=18.0, fontweight="bold", y=0.965)

    # More balanced panel areas
    axA = fig.add_axes([0.060, 0.160, 0.365, 0.690])
    axB = fig.add_axes([0.485, 0.160, 0.330, 0.690])
    axC = fig.add_axes([0.855, 0.160, 0.120, 0.690])

    # Panel A: clean symbolic cards
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    axA.text(-0.075, 1.045, "a", transform=axA.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axA.set_title("Borax-induced anthocyanin species shift", fontsize=13.2, fontweight="bold", pad=12)
    axA.text(0.50, 0.810, "Increasing borax concentration increases alkalinity",
             ha="center", va="center", fontsize=10.5, fontweight="bold", color="0.10")

    card_w, card_h = 0.205, 0.415
    xs = [0.095, 0.397, 0.699]
    y = 0.310
    cards = [
        ("#8A59A5", r"AH$^+$", "Flavylium", "cationic / purple", "low-pH dominant"),
        ("#3571B8", "A", "Quinoidal", "neutral / blue", "deprotonated form"),
        ("#2F8F78", r"A$^-$ / C", "Chalcone-like", "open-ring / green-blue", "alkaline form"),
    ]
    for x, cdat in zip(xs, cards):
        _species_card_final(axA, x, y, card_w, card_h, *cdat)
    # compact equilibrium symbols between cards, away from card text
    axA.text(xs[0]+card_w+0.048, y+card_h*0.53, "⇌", ha="center", va="center", fontsize=12.5, color="0.28")
    axA.text(xs[1]+card_w+0.048, y+card_h*0.53, "⇌", ha="center", va="center", fontsize=12.5, color="0.28")
    axA.text(0.50, 0.105,
             "Simplified species notation for natural anthocyanin extract; detailed substituents may vary.",
             ha="center", va="center", fontsize=7.5, color="0.42")

    # Panel B: same science, lighter confidence band and smaller colorbar
    axB.text(-0.125, 1.045, "b", transform=axB.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axB.set_title("pH–hue coupling", fontsize=13.2, fontweight="bold", pad=12)
    valid = summary.dropna(subset=["pH", "H_mean"])
    if not valid.empty and len(valid) >= 4:
        x = valid["pH"].astype(float).values
        yv = valid["H_mean"].astype(float).values
        cval = valid.index.astype(float).values
        order = 2 if len(x) >= 6 else 1
        coef = np.polyfit(x, yv, order)
        xs_fit = np.linspace(np.nanmin(x), np.nanmax(x), 240)
        ys_fit = np.polyval(coef, xs_fit)
        resid = yv - np.polyval(coef, x)
        band = 0.58 * np.nanstd(resid) if np.isfinite(np.nanstd(resid)) else 0.8
        axB.fill_between(xs_fit, ys_fit-band, ys_fit+band, color="0.75", alpha=0.18, lw=0, zorder=0)
        axB.plot(xs_fit, ys_fit, color="0.12", lw=2.0, zorder=1)
        sc = axB.scatter(x, yv, c=cval, cmap="viridis", s=42, edgecolor="black", lw=0.45, zorder=3)
        for conc in [0, 5, 10, 15, 20]:
            if conc in valid.index:
                axB.text(valid.loc[conc, "pH"] + 0.014, valid.loc[conc, "H_mean"] + 0.45,
                         str(conc), fontsize=7.2)
        try:
            from scipy.stats import pearsonr, spearmanr
            rr, pp = pearsonr(x, yv); rho, _ = spearmanr(x, yv)
            ptxt = "p < 0.001" if pp < 0.001 else f"p = {pp:.2g}"
            txt = f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\n{ptxt}"
        except Exception:
            rr = np.corrcoef(x, yv)[0, 1]
            txt = f"r = {rr:.3f}\nR² = {rr**2:.3f}"
        axB.text(0.94, 0.96, txt, transform=axB.transAxes, ha="right", va="top", fontsize=7.9,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.60", alpha=0.93))
        cb = fig.colorbar(sc, ax=axB, fraction=0.024, pad=0.008)
        cb.set_label("mM", fontsize=7.5)
        cb.ax.tick_params(labelsize=6.7)
    axB.set_xlabel("Measured pH", fontsize=10)
    axB.set_ylabel("HSV hue angle (°)", fontsize=10)
    axB.tick_params(labelsize=8.3)
    axB.grid(True, alpha=0.15, lw=0.5)

    # Panel C: trajectory strip only, not redundant manual H table
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    axC.text(-0.26, 1.045, "c", transform=axC.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axC.set_title("Color trajectory", fontsize=13.2, fontweight="bold", pad=12)
    means = summary.reindex([c for c in ALL_CONC if c in summary.index]).dropna(subset=["R_mean", "G_mean", "B_mean"])
    if not means.empty:
        arr_rgb = np.array([[_rgb01_from_series(means.loc[c]) for c in means.index]])
        x0, y0, w, h = 0.08, 0.50, 0.84, 0.135
        axC.text(x0, y0+h+0.095, "0 mM", ha="left", va="bottom", fontsize=8.3, fontweight="bold")
        axC.text(x0+w, y0+h+0.095, "20 mM", ha="right", va="bottom", fontsize=8.3, fontweight="bold")
        axC.annotate("", xy=(x0+w, y0+h+0.070), xytext=(x0, y0+h+0.070),
                     arrowprops=dict(arrowstyle="-|>", lw=0.95, color="0.25", mutation_scale=7))
        axC.imshow(arr_rgb, aspect="auto", extent=[x0, x0+w, y0, y0+h])
        n = len(means)
        for i in range(n+1):
            xx = x0 + w*i/n
            axC.plot([xx, xx], [y0, y0+h], color="white", lw=0.30, alpha=0.70)
        axC.text(x0+w/2, y0-0.075, "mean ROI color from smartphone images", ha="center", fontsize=7.3, color="0.35")
        h0 = means.loc[0, "H_mean"] if 0 in means.index and "H_mean" in means.columns else np.nan
        h20 = means.loc[20, "H_mean"] if 20 in means.index and "H_mean" in means.columns else np.nan
        if np.isfinite(h0) and np.isfinite(h20):
            axC.text(x0, y0-0.185, f"Hue {h0:.1f}°", ha="left", fontsize=7.6, color="0.25")
            axC.text(x0+w, y0-0.185, f"{h20:.1f}°", ha="right", fontsize=7.6, color="0.25")

    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)



# =============================================================================
# FINAL Q1 TOP-TIER POLISH OVERRIDES
# - Figure 2A converted into a compact species-transition schematic.
# - Figure 2C made less empty and more data-calibration-like.
# - Figure 4 and Figure 5 merged conceptually: Fig04 now contains spectra,
#   heatmap, difference spectra, and robust descriptors in one 2x3 panel.
# - Fig05 is saved as a supplementary descriptor-only figure to avoid main-text
#   fragmentation.
# - Figure 9 fallback no longer contains software/install placeholder text.
# =============================================================================


def _species_compact_card(ax, x, y, w, h, color, symbol, name, state, note):
    """Small species card for journal figure; no pseudo-molecular drawing."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        fc="white", ec="0.62", lw=0.90
    ))
    ax.add_patch(Rectangle((x+w*0.16, y+h*0.75), w*0.68, h*0.105,
                           fc=color, ec="0.35", lw=0.40))
    ax.text(x+w/2, y+h*0.535, symbol, ha="center", va="center",
            fontsize=23, fontweight="bold", color="0.03")
    ax.text(x+w/2, y+h*0.315, name, ha="center", va="center",
            fontsize=8.7, fontweight="bold", color="0.03")
    ax.text(x+w/2, y+h*0.205, state, ha="center", va="center",
            fontsize=7.2, color="0.26")
    ax.text(x+w/2, y+h*0.105, note, ha="center", va="center",
            fontsize=6.6, color="0.45")


def figure_0_molecular_mechanism(color_df: pd.DataFrame) -> None:
    """Final polished Figure 2: compact mechanism, pH-hue coupling, and color trajectory."""
    set_premium_style()
    summary = _safe_color_summary_for_visual(color_df)

    fig = plt.figure(figsize=(17.2, 6.55), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle("Mechanistic basis of borax-induced anthocyanin colorimetry",
                 fontsize=17.5, fontweight="bold", y=0.962)

    # Better visual balance: A moderately large, B central, C compact but not empty.
    axA = fig.add_axes([0.060, 0.165, 0.365, 0.670])
    axB = fig.add_axes([0.485, 0.165, 0.330, 0.670])
    axC = fig.add_axes([0.855, 0.165, 0.125, 0.670])

    # Panel A — symbolic chemical-species transition.
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    axA.text(-0.075, 1.045, "a", transform=axA.transAxes,
             fontsize=13.5, fontweight="bold", va="bottom")
    axA.set_title("Borax-induced anthocyanin species shift",
                  fontsize=12.8, fontweight="bold", pad=10)
    axA.text(0.50, 0.800, "Increasing borax concentration increases alkalinity",
             ha="center", va="center", fontsize=9.8, fontweight="bold", color="0.12")

    card_w, card_h = 0.205, 0.405
    xs = [0.095, 0.397, 0.699]
    y = 0.325
    cards = [
        ("#8A59A5", r"AH$^+$", "Flavylium", "cationic / purple", "low-pH dominant"),
        ("#3571B8", "A", "Quinoidal", "neutral / blue", "deprotonated form"),
        ("#2F8F78", r"A$^-$ / C", "Chalcone-like", "open-ring / green-blue", "alkaline form"),
    ]
    for x0, cdat in zip(xs, cards):
        _species_compact_card(axA, x0, y, card_w, card_h, *cdat)
    # Clean equilibrium symbols between cards; no pH-arrow text collision.
    axA.text(xs[0] + card_w + 0.047, y + card_h*0.53, "⇌",
             ha="center", va="center", fontsize=12.4, color="0.25")
    axA.text(xs[1] + card_w + 0.047, y + card_h*0.53, "⇌",
             ha="center", va="center", fontsize=12.4, color="0.25")
    axA.text(0.50, 0.115,
             "Simplified species notation for natural anthocyanin extract; detailed substituents may vary.",
             ha="center", va="center", fontsize=7.3, color="0.42")

    # Panel B — pH-hue quantitative coupling.
    axB.text(-0.125, 1.045, "b", transform=axB.transAxes,
             fontsize=13.5, fontweight="bold", va="bottom")
    axB.set_title("pH–hue coupling", fontsize=12.8, fontweight="bold", pad=10)
    valid = summary.dropna(subset=["pH", "H_mean"])
    if not valid.empty and len(valid) >= 4:
        x = valid["pH"].astype(float).values
        yv = valid["H_mean"].astype(float).values
        cval = valid.index.astype(float).values
        order = 2 if len(x) >= 6 else 1
        coef = np.polyfit(x, yv, order)
        xs_fit = np.linspace(np.nanmin(x), np.nanmax(x), 260)
        ys_fit = np.polyval(coef, xs_fit)
        resid = yv - np.polyval(coef, x)
        band = 0.50 * np.nanstd(resid) if np.isfinite(np.nanstd(resid)) else 0.8
        axB.fill_between(xs_fit, ys_fit-band, ys_fit+band,
                         color="0.75", alpha=0.13, lw=0, zorder=0)
        axB.plot(xs_fit, ys_fit, color="0.13", lw=2.0, zorder=1)
        sc = axB.scatter(x, yv, c=cval, cmap="viridis", s=42,
                         edgecolor="black", lw=0.45, zorder=3)
        for conc in [0, 5, 10, 15, 20]:
            if conc in valid.index:
                axB.text(valid.loc[conc, "pH"] + 0.014,
                         valid.loc[conc, "H_mean"] + 0.45,
                         str(conc), fontsize=7.1)
        try:
            from scipy.stats import pearsonr, spearmanr
            rr, pp = pearsonr(x, yv); rho, _ = spearmanr(x, yv)
            ptxt = "p < 0.001" if pp < 0.001 else f"p = {pp:.2g}"
            txt = f"r = {rr:.3f}\nρ = {rho:.3f}\nR² = {rr**2:.3f}\n{ptxt}"
        except Exception:
            rr = np.corrcoef(x, yv)[0, 1]
            txt = f"r = {rr:.3f}\nR² = {rr**2:.3f}"
        axB.text(0.94, 0.96, txt, transform=axB.transAxes,
                 ha="right", va="top", fontsize=7.8,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.60", alpha=0.94))
        cb = fig.colorbar(sc, ax=axB, fraction=0.024, pad=0.008)
        cb.set_label("mM", fontsize=7.4)
        cb.ax.tick_params(labelsize=6.6)
    axB.set_xlabel("Measured pH", fontsize=10)
    axB.set_ylabel("HSV hue angle (°)", fontsize=10)
    axB.tick_params(labelsize=8.2)
    axB.grid(True, alpha=0.13, lw=0.5)

    # Panel C — compact calibration-like trajectory, reduced white-space.
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    axC.text(-0.28, 1.045, "c", transform=axC.transAxes,
             fontsize=13.5, fontweight="bold", va="bottom")
    axC.set_title("Color trajectory", fontsize=12.8, fontweight="bold", pad=10)
    means = summary.reindex([c for c in ALL_CONC if c in summary.index]).dropna(subset=["R_mean", "G_mean", "B_mean"])
    if not means.empty:
        arr_rgb = np.array([[_rgb01_from_series(means.loc[c]) for c in means.index]])
        x0, y0, w, h = 0.08, 0.50, 0.86, 0.155
        axC.text(x0, y0+h+0.095, "0 mM", ha="left", va="bottom", fontsize=8.2, fontweight="bold")
        axC.text(x0+w, y0+h+0.095, "20 mM", ha="right", va="bottom", fontsize=8.2, fontweight="bold")
        axC.annotate("", xy=(x0+w, y0+h+0.070), xytext=(x0, y0+h+0.070),
                     arrowprops=dict(arrowstyle="-|>", lw=0.95, color="0.25", mutation_scale=7))
        axC.imshow(arr_rgb, aspect="auto", extent=[x0, x0+w, y0, y0+h])
        n = len(means)
        for i in range(n+1):
            xx = x0 + w*i/n
            axC.plot([xx, xx], [y0, y0+h], color="white", lw=0.30, alpha=0.72)
        # Add five unobtrusive ticks to make it look like a mini-calibration scale.
        for tick in [0, 5, 10, 15, 20]:
            if tick in means.index:
                frac = (tick - means.index.min()) / max(1e-9, (means.index.max() - means.index.min()))
                xx = x0 + w*frac
                axC.plot([xx, xx], [y0-0.010, y0], color="0.25", lw=0.55)
                axC.text(xx, y0-0.035, f"{tick:g}", ha="center", va="top", fontsize=6.6, color="0.25")
        axC.text(x0+w/2, y0-0.090, "mean ROI color", ha="center", fontsize=7.4, color="0.35")
        h0 = means.loc[0, "H_mean"] if 0 in means.index and "H_mean" in means.columns else np.nan
        h20 = means.loc[20, "H_mean"] if 20 in means.index and "H_mean" in means.columns else np.nan
        if np.isfinite(h0) and np.isfinite(h20):
            axC.text(x0, y0-0.190, f"Hue {h0:.1f}°", ha="left", fontsize=7.5, color="0.25")
            axC.text(x0+w, y0-0.190, f"{h20:.1f}°", ha="right", fontsize=7.5, color="0.25")

    savefig(FIG_MAIN / "Fig02_molecular_mechanism_color_scale", tight=False)


def figure_4_uvvis_top_tier(uv_spectra: pd.DataFrame) -> None:
    """Main UV-Vis figure revised as a compact 2x3 panel including descriptors."""
    if uv_spectra.empty:
        return
    set_premium_style()
    avg = uv_spectra.groupby(["concentration_mM", "wavelength_nm"])["absorbance"].mean().reset_index()
    concs = sorted(avg["concentration_mM"].unique())
    selected = [c for c in [0, 5, 10, 15, 20] if c in concs]
    if len(selected) < 3:
        selected = concs[::max(1, len(concs)//5)]
    cmap = plt.cm.viridis

    # Spectral features computed from averaged spectra for panels d-f.
    feats = []
    for conc in concs:
        g = avg[avg["concentration_mM"] == conc].sort_values("wavelength_nm")
        wl = g["wavelength_nm"].to_numpy(float)
        ab = g["absorbance"].to_numpy(float)
        if len(wl) < 3:
            continue
        win = (wl >= 350) & (wl <= 760)
        if not np.any(win): win = np.ones_like(wl, dtype=bool)
        wlw, abw = wl[win], ab[win]
        idx = int(np.nanargmax(abw))
        auc = float(np.trapezoid(abw, wlw))
        feats.append({"concentration_mM": conc,
                      "lambda_max_nm": float(wlw[idx]),
                      "Amax": float(abw[idx]),
                      "AUC": auc})
    fdf = pd.DataFrame(feats)

    fig = plt.figure(figsize=(15.5, 7.2), dpi=300)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.05, 1.05], height_ratios=[1.15, 0.90],
                  wspace=0.34, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    for i, conc in enumerate(selected):
        g = avg[avg["concentration_mM"] == conc]
        ax.plot(g["wavelength_nm"], g["absorbance"], lw=1.85,
                label=f"{conc:g} mM", color=cmap(i/max(1, len(selected)-1)))
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance")
    ax.set_title("Selected UV–Vis spectra"); ax.legend(frameon=False, fontsize=7.6)
    ax.grid(alpha=0.13); add_panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    pivot = avg.pivot_table(index="concentration_mM", columns="wavelength_nm", values="absorbance")
    im = ax2.imshow(pivot.values, aspect="auto", origin="lower",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()], cmap="magma")
    ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Borax concentration (mM)")
    ax2.set_title("Absorbance heatmap")
    cb = fig.colorbar(im, ax=ax2, pad=0.018, fraction=0.040); cb.set_label("Absorbance")
    add_panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[0, 2])
    blank = avg[avg["concentration_mM"] == 0][["wavelength_nm", "absorbance"]].rename(columns={"absorbance":"blank"})
    for i, conc in enumerate(selected):
        if conc == 0: continue
        g = avg[avg["concentration_mM"] == conc].merge(blank, on="wavelength_nm", how="inner")
        ax3.plot(g["wavelength_nm"], g["absorbance"] - g["blank"], lw=1.65,
                 label=f"{conc:g} mM", color=cmap(i/max(1, len(selected)-1)))
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("Wavelength (nm)"); ax3.set_ylabel("ΔAbsorbance vs blank")
    ax3.set_title("Difference spectra"); ax3.legend(frameon=False, fontsize=7.6)
    ax3.grid(alpha=0.13); add_panel_label(ax3, "c")

    for axd, col, ttl, ylab, lab in zip(
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
        ["lambda_max_nm", "Amax", "AUC"],
        ["λmax shift", "Peak absorbance", "Integrated absorbance"],
        ["λmax (nm)", "Amax", "AUC"],
        ["d", "e", "f"]):
        if not fdf.empty and col in fdf.columns:
            axd.plot(fdf["concentration_mM"], fdf[col], marker="o", lw=1.85, ms=4.5)
        axd.set_xlabel("Borax concentration (mM)"); axd.set_ylabel(ylab); axd.set_title(ttl)
        axd.grid(alpha=0.13); add_panel_label(axd, lab)

    fig.suptitle("Spectroscopic validation and robust UV–Vis descriptors of borax-induced anthocyanin transformation", y=1.015)
    savefig(FIG_MAIN / "Fig04_UVVis_validation_and_descriptors")
    # Also overwrite the legacy Fig04 name so the figure map still points to an updated file.
    savefig(FIG_MAIN / "Fig04_UVVis_selected_heatmap_difference")


def figure_5_uvvis_features(uv_features: pd.DataFrame) -> None:
    """Descriptor-only figure is now supplementary, because descriptors are merged into Fig. 4."""
    if uv_features.empty:
        return
    g = uv_features.groupby("concentration_mM").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    for ax, col, title, ylabel in zip(
        axes,
        ["lambda_max_nm", "Amax", "AUC"],
        ["λmax shift", "Peak absorbance", "Integrated absorbance"],
        ["λmax (nm)", "Amax", "AUC"]):
        if col not in g.columns:
            ax.axis("off"); continue
        ax.plot(g["concentration_mM"], g[col], marker="o", lw=1.8)
        ax.set_xlabel("Borax concentration (mM)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.13)
    fig.suptitle("Supplementary UV–Vis spectral descriptors", y=1.03)
    savefig(FIG_SUPP / "FigS_UVVis_robust_spectral_descriptors")


def figure_9_explainability(color_df: pd.DataFrame) -> pd.DataFrame:
    """Main XAI figure without placeholder software-install text."""
    if color_df.empty:
        return pd.DataFrame()
    cols = feature_columns(color_df, include_ph=False)
    if not cols:
        return pd.DataFrame()
    Xdf = color_df[cols].fillna(color_df[cols].median())
    X = Xdf.values
    y = color_df["concentration_mM"].values.astype(float)

    try:
        import shap
        model = ExtraTreesRegressor(n_estimators=500, random_state=42, min_samples_leaf=1)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        shap_df = pd.DataFrame({"feature": cols, "mean_abs_SHAP": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_SHAP", ascending=False)
        shap_df.to_csv(TAB_SUPP / "TableS_SHAP_feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(8.0, 5.6))
        top_feats = shap_df.head(12)["feature"].tolist()[::-1]
        rng = np.random.default_rng(42)
        for yi, feat in enumerate(top_feats):
            j = cols.index(feat)
            vals = sv[:, j]
            raw = Xdf[feat].values
            norm = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw) + 1e-9)
            jitter = rng.normal(0, 0.055, size=len(vals))
            ax.scatter(vals, yi + jitter, c=norm, cmap="viridis", s=16, edgecolor="none", alpha=0.84)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.set_yticks(range(len(top_feats))); ax.set_yticklabels(top_feats)
        ax.set_xlabel("SHAP value impact on predicted concentration (mM)")
        ax.set_title("SHAP summary of colorimetric feature contributions")
        sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0, vmax=1))
        cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label("Feature value (low → high)")
        ax.grid(axis="x", alpha=0.13)
        savefig(FIG_MAIN / "Fig09_explainable_feature_importance")
        return shap_df
    except Exception:
        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        model.fit(X, y)
        result = permutation_importance(model, X, y, n_repeats=80,
                                        random_state=42, scoring="neg_root_mean_squared_error")
        imp = pd.DataFrame({
            "feature": cols,
            "importance_mean": result.importances_mean,
            "importance_sd": result.importances_std
        }).sort_values("importance_mean", ascending=False)
        top = imp.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        ax.barh(top["feature"], top["importance_mean"],
                xerr=top["importance_sd"], capsize=3,
                edgecolor="black", linewidth=0.45)
        ax.set_xlabel("Permutation importance (increase in RMSE score)")
        ax.set_title("Model interpretability by permutation feature importance")
        ax.grid(axis="x", alpha=0.13)
        # No SHAP-install text inside a publication figure.
        savefig(FIG_MAIN / "Fig09_explainable_feature_importance")
        return imp




# =============================================================================
# FINAL PUBLICATION PACKAGE OVERRIDES
# Adds: output cleanup, graphical abstract, corrected Fig. 4 saving, premium
# table figures, literature-comparison template, and a corrected final figure map.
# These functions intentionally override previous definitions without changing
# the validated ROI, UV-Vis, chemometrics, and ML logic above.
# =============================================================================


def _delete_previous_generated_figures() -> None:
    """Remove stale generated figure files so numbering never contains blanks/duplicates."""
    for folder in [FIG_MAIN, FIG_SUPP]:
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ["*.png", "*.pdf", "*.tiff", "*.tif", "*.svg"]:
            for f in folder.glob(ext):
                try:
                    f.unlink()
                except Exception:
                    pass


def _savefig_multi(fig: plt.Figure, base_paths: List[Path], tight: bool = True) -> None:
    """Save the same open figure to several base paths, then close once."""
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    for base in base_paths:
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".png", dpi=FIG_DPI, bbox_inches="tight")
        if SAVE_PDF:
            fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        if SAVE_TIFF:
            fig.savefig(str(base) + ".tiff", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def figure_4_uvvis_top_tier(uv_spectra: pd.DataFrame) -> None:
    """Corrected main UV-Vis figure.

    This replaces the previous version that attempted to call savefig twice on
    the same closed figure, which could create blank or inconsistent Fig. 4
    outputs. The descriptor panels remain integrated into one six-panel Fig. 4.
    """
    if uv_spectra.empty:
        return
    set_premium_style()
    avg = uv_spectra.groupby(["concentration_mM", "wavelength_nm"])["absorbance"].mean().reset_index()
    concs = sorted(avg["concentration_mM"].unique())
    selected = [c for c in [0, 5, 10, 15, 20] if c in concs]
    if len(selected) < 3:
        selected = concs[::max(1, len(concs)//5)]
    cmap = plt.cm.viridis

    feats = []
    for conc in concs:
        g = avg[avg["concentration_mM"] == conc].sort_values("wavelength_nm")
        wl = g["wavelength_nm"].to_numpy(float)
        ab = g["absorbance"].to_numpy(float)
        if len(wl) < 3:
            continue
        win = (wl >= 350) & (wl <= 760)
        if not np.any(win):
            win = np.ones_like(wl, dtype=bool)
        wlw, abw = wl[win], ab[win]
        idx = int(np.nanargmax(abw))
        feats.append({
            "concentration_mM": conc,
            "lambda_max_nm": float(wlw[idx]),
            "Amax": float(abw[idx]),
            "AUC": float(np.trapezoid(abw, wlw)),
        })
    fdf = pd.DataFrame(feats)

    fig = plt.figure(figsize=(15.5, 7.2), dpi=300)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.05, 1.05], height_ratios=[1.15, 0.90],
                  wspace=0.34, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    for i, conc in enumerate(selected):
        g = avg[avg["concentration_mM"] == conc]
        ax.plot(g["wavelength_nm"], g["absorbance"], lw=1.85,
                label=f"{conc:g} mM", color=cmap(i/max(1, len(selected)-1)))
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance")
    ax.set_title("Selected UV–Vis spectra"); ax.legend(frameon=False, fontsize=7.6)
    ax.grid(alpha=0.13); add_panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    pivot = avg.pivot_table(index="concentration_mM", columns="wavelength_nm", values="absorbance")
    im = ax2.imshow(pivot.values, aspect="auto", origin="lower",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()], cmap="magma")
    ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Borax concentration (mM)")
    ax2.set_title("Absorbance heatmap")
    cb = fig.colorbar(im, ax=ax2, pad=0.018, fraction=0.040); cb.set_label("Absorbance")
    add_panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[0, 2])
    blank = avg[avg["concentration_mM"] == 0][["wavelength_nm", "absorbance"]].rename(columns={"absorbance":"blank"})
    for i, conc in enumerate(selected):
        if conc == 0 or blank.empty:
            continue
        g = avg[avg["concentration_mM"] == conc].merge(blank, on="wavelength_nm", how="inner")
        ax3.plot(g["wavelength_nm"], g["absorbance"] - g["blank"], lw=1.65,
                 label=f"{conc:g} mM", color=cmap(i/max(1, len(selected)-1)))
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("Wavelength (nm)"); ax3.set_ylabel("ΔAbsorbance vs blank")
    ax3.set_title("Difference spectra"); ax3.legend(frameon=False, fontsize=7.6)
    ax3.grid(alpha=0.13); add_panel_label(ax3, "c")

    for axd, col, ttl, ylab, lab in zip(
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
        ["lambda_max_nm", "Amax", "AUC"],
        ["λmax shift", "Peak absorbance", "Integrated absorbance"],
        ["λmax (nm)", "Amax", "AUC"],
        ["d", "e", "f"]):
        if not fdf.empty and col in fdf.columns:
            axd.plot(fdf["concentration_mM"], fdf[col], marker="o", lw=1.85, ms=4.5)
        axd.set_xlabel("Borax concentration (mM)"); axd.set_ylabel(ylab); axd.set_title(ttl)
        axd.grid(alpha=0.13); add_panel_label(axd, lab)

    fig.suptitle("Spectroscopic validation and robust UV–Vis descriptors of borax-induced anthocyanin transformation", y=1.015)
    _savefig_multi(fig, [FIG_MAIN / "Fig04_UVVis_validation_and_descriptors",
                         FIG_MAIN / "Fig04_UVVis_selected_heatmap_difference"])


def create_graphical_abstract(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Create a submission-ready graphical abstract from the same data objects."""
    set_premium_style()
    fig, ax = plt.subplots(figsize=(15.8, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.suptitle("Multimodal Smartphone Colorimetry for Quantitative Borax Determination", y=0.98,
                 fontsize=15.5, fontweight="bold")

    nodes = [
        ("Purple cabbage\nanthocyanin", "natural probe", "#F2E8FA"),
        ("Borax-induced\nalkalization", "pH increase", "#EAF3FF"),
        ("Color transition", "purple → blue-green", "#F7F7F7"),
        ("Smartphone\nROI extraction", "RGB–HSV–CIELab", "#FFF2DF"),
        ("UV–Vis\nvalidation", "spectral fingerprints", "#EAF7F0"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", "#F3F3F3"),
        ("Borax prediction", "quantitative mM", "#EAF8EC"),
    ]
    xs = np.linspace(0.08, 0.92, len(nodes))
    y = 0.58
    for i, (title, subtitle, fc) in enumerate(nodes):
        ax.add_patch(FancyBboxPatch((xs[i]-0.055, y-0.125), 0.110, 0.22,
                                    boxstyle="round,pad=0.018,rounding_size=0.022",
                                    fc=fc, ec="0.18", lw=1.0))
        ax.text(xs[i], y+0.035, title, ha="center", va="center", fontsize=9.4, fontweight="bold", linespacing=0.86)
        ax.text(xs[i], y-0.070, subtitle, ha="center", va="center", fontsize=7.2, color="0.30")
        if i < len(nodes)-1:
            ax.add_patch(FancyArrowPatch((xs[i]+0.060, y-0.010), (xs[i+1]-0.060, y-0.010),
                                         arrowstyle="-|>", mutation_scale=13, lw=1.1, color="0.22"))

    # Smartphone color strip
    if not color_df.empty:
        means = color_df.groupby("concentration_mM")[["R_mean", "G_mean", "B_mean"]].mean().reindex(ALL_CONC).dropna()
        x0, y0, w, h = 0.18, 0.18, 0.35, 0.075
        n = len(means)
        for i, (conc, row) in enumerate(means.iterrows()):
            ax.add_patch(Rectangle((x0+w*i/n, y0), w/n, h, fc=np.clip(row.values/255, 0, 1), ec="white", lw=0.2))
        ax.text(x0, y0-0.035, "0 mM", ha="left", va="top", fontsize=8.3)
        ax.text(x0+w, y0-0.035, "20 mM", ha="right", va="top", fontsize=8.3)
        ax.text(x0+w/2, y0+h+0.025, "data-derived smartphone color trajectory", ha="center", fontsize=8.2, color="0.33")

    # Mini spectrum inset
    if not uv_spectra.empty:
        inset = ax.inset_axes([0.60, 0.12, 0.25, 0.23])
        avg = _get_avg_uvvis(uv_spectra)
        for c in [0, 10, 20]:
            if c in set(avg["concentration_mM"].astype(float)):
                g = avg[np.isclose(avg["concentration_mM"].astype(float), float(c))]
                inset.plot(g["wavelength_nm"], g["absorbance"], lw=1.0, label=f"{c} mM")
        inset.set_xlabel("nm", fontsize=7); inset.set_ylabel("Abs", fontsize=7)
        inset.tick_params(labelsize=6); inset.legend(frameon=False, fontsize=6, loc="upper right")
        inset.set_title("UV–Vis validation", fontsize=7.6)
        inset.grid(alpha=0.12)

    ax.text(0.50, 0.045,
            "Natural anthocyanin chemistry + smartphone imaging + spectroscopic validation + explainable machine learning",
            ha="center", va="center", fontsize=10.2, fontweight="bold")
    _savefig_multi(fig, [OUTPUT / "Graphical_Abstract", FIG_MAIN / "Graphical_Abstract"], tight=False)


def create_premium_publication_tables(color_df: pd.DataFrame, uv_features: pd.DataFrame,
                                      analytical: pd.DataFrame, perf_ext: pd.DataFrame,
                                      perf_int: pd.DataFrame, imp: pd.DataFrame) -> None:
    """Create publication-ready PNG/PDF table figures and complete CSV tables."""
    # Main Table 1: standards and pH
    table1 = pd.DataFrame({"Borax concentration (mM)": list(PH_MAP.keys()), "Measured pH": list(PH_MAP.values())})
    table1.to_csv(TAB_MAIN / "Table1_borax_standards_and_pH.csv", index=False)
    format_table_png(table1, "Table 1. Borax standards and measured pH", TAB_MAIN / "Table1_borax_standards_and_pH", max_rows=25)

    # Main Table 2: analytical figures of merit
    if analytical is not None and not analytical.empty:
        analytical.to_csv(TAB_MAIN / "Table2_analytical_figures_of_merit.csv", index=False)
        format_table_png(analytical, "Table 2. Analytical figures of merit", TAB_MAIN / "Table2_analytical_figures_of_merit", max_rows=20, font_size=7.2, col_width=1.15)

    # Main Table 3: external ML validation, best models per direction
    if perf_ext is not None and not perf_ext.empty:
        perf_clean = perf_ext.copy()
        perf_clean = perf_clean.dropna(subset=["RMSE_mM"])
        if not perf_clean.empty:
            table3 = perf_clean.sort_values(["validation", "RMSE_mM", "MAE_mM"]).groupby("validation", as_index=False).head(5)
            table3.to_csv(TAB_MAIN / "Table3_external_validation_performance.csv", index=False)
            format_table_png(table3, "Table 3. Independent external validation performance", TAB_MAIN / "Table3_external_validation_performance", max_rows=12, font_size=7.3, col_width=1.10)

    # Main Table 4: comparison template; no fabricated literature values.
    comparison = pd.DataFrame([
        {"Method": "This work", "Probe/Reagent": "Purple cabbage anthocyanin", "Platform": "Smartphone + UV–Vis validation", "Linear range": "Auto-filled from Table 2", "LOD": "Auto-filled from Table 2", "Notes": "Natural probe; XAI-supported"},
        {"Method": "Literature study 1", "Probe/Reagent": "Fill from reference", "Platform": "Fill from reference", "Linear range": "Fill", "LOD": "Fill", "Notes": "Fill"},
        {"Method": "Literature study 2", "Probe/Reagent": "Fill from reference", "Platform": "Fill from reference", "Linear range": "Fill", "LOD": "Fill", "Notes": "Fill"},
        {"Method": "Literature study 3", "Probe/Reagent": "Fill from reference", "Platform": "Fill from reference", "Linear range": "Fill", "LOD": "Fill", "Notes": "Fill"},
    ])
    if analytical is not None and not analytical.empty:
        row0 = analytical.iloc[0]
        range_col = next((c for c in analytical.columns if "range" in c.lower()), None)
        lod_col = next((c for c in analytical.columns if "lod" in c.lower()), None)
        comparison.loc[0, "Linear range"] = str(row0[range_col]) if range_col else "See Table 2"
        comparison.loc[0, "LOD"] = str(row0[lod_col]) if lod_col else "See Table 2"
    comparison.to_csv(TAB_MAIN / "Table4_comparison_with_reported_methods_TEMPLATE.csv", index=False)
    format_table_png(comparison, "Table 4. Comparison with reported borax sensing methods (template)", TAB_MAIN / "Table4_comparison_with_reported_methods_TEMPLATE", max_rows=8, font_size=7.0, col_width=1.35)

    # Supplementary tables as figure images for SI.
    if color_df is not None and not color_df.empty:
        color_df.to_csv(TAB_SUPP / "TableS1_extracted_color_features_complete.csv", index=False)
        color_summary = color_df.groupby("concentration_mM").mean(numeric_only=True).reset_index()
        color_summary.to_csv(TAB_SUPP / "TableS2_concentration_level_color_summary.csv", index=False)
        keep_cols = [c for c in ["concentration_mM", "pH", "H_mean", "R_mean", "G_mean", "B_mean", "Lab_a_mean", "Lab_b_mean", "Excess_Blue_Index"] if c in color_summary.columns]
        format_table_png(color_summary[keep_cols], "Table S2. Concentration-level color feature summary", TAB_SUPP / "TableS2_concentration_level_color_summary", max_rows=25, font_size=7.0, col_width=1.05)
    if uv_features is not None and not uv_features.empty:
        uv_features.to_csv(TAB_SUPP / "TableS3_UVVis_spectral_features_complete.csv", index=False)
        uv_mean = uv_features.groupby("concentration_mM").mean(numeric_only=True).reset_index()
        keep = [c for c in ["concentration_mM", "lambda_max_nm", "Amax", "AUC", "A420", "A520", "A620"] if c in uv_mean.columns]
        format_table_png(uv_mean[keep], "Table S3. UV–Vis spectral descriptors", TAB_SUPP / "TableS3_UVVis_spectral_descriptors", max_rows=25, font_size=7.0, col_width=1.05)
    if perf_int is not None and not perf_int.empty:
        perf_int.to_csv(TAB_SUPP / "TableS4_internal_LOIO_model_performance.csv", index=False)
        format_table_png(perf_int.head(12), "Table S4. Internal leave-one-image-out model performance", TAB_SUPP / "TableS4_internal_LOIO_model_performance", max_rows=12, font_size=7.2, col_width=1.05)
    if imp is not None and not imp.empty:
        imp.to_csv(TAB_SUPP / "TableS5_feature_importance_values.csv", index=False)
        format_table_png(imp.head(15), "Table S5. Feature importance values", TAB_SUPP / "TableS5_feature_importance_values", max_rows=15, font_size=7.0, col_width=1.20)


def _write_final_maps() -> None:
    figure_map = pd.DataFrame([
        ["Graphical Abstract", "End-to-end multimodal sensing concept", "Graphical_Abstract"],
        ["Fig. 1", "Multimodal smartphone colorimetry workflow", "figures_main/Fig01_workflow_schematic"],
        ["Fig. 2", "Mechanistic basis of borax-induced anthocyanin colorimetry", "figures_main/Fig02_molecular_mechanism_color_scale"],
        ["Fig. 3", "Representative ROI validation and color evolution", "figures_main/Fig03_representative_ROI_and_color_gallery"],
        ["Fig. 4", "pH-mediated color response and CIELab trajectory", "figures_main/Fig03_pH_hue_Lab_mechanism"],
        ["Fig. 5", "UV–Vis validation: spectra, heatmap, difference spectra, descriptors", "figures_main/Fig04_UVVis_validation_and_descriptors"],
        ["Fig. 6", "Analytical calibration and LOD/LOQ", "figures_main/Fig06_analytical_calibration_dual_linear_nonlinear"],
        ["Fig. 7", "Chemometric PCA and multimodal correlation", "figures_main/Fig07_PCA_and_multimodal_correlation"],
        ["Fig. 8", "Independent ganjil–genap ML validation", "figures_main/Fig08_ML_external_validation"],
        ["Fig. 9", "Explainable feature relevance", "figures_main/Fig09_explainable_feature_importance"],
        ["Fig. 10", "Bland–Altman agreement and permutation test", "figures_main/Fig10_BlandAltman_and_permutation_test"],
        ["Fig. 11", "Cross-modal smartphone–UV–Vis correlation", "figures_main/Fig11_color_UVVis_correlation"],
    ], columns=["Item", "Description", "File base"])
    figure_map.to_csv(OUTPUT / "recommended_main_figure_map_FINAL.csv", index=False)

    table_map = pd.DataFrame([
        ["Table 1", "Borax standards and measured pH", "tables_main/Table1_borax_standards_and_pH"],
        ["Table 2", "Analytical figures of merit", "tables_main/Table2_analytical_figures_of_merit"],
        ["Table 3", "External validation performance", "tables_main/Table3_external_validation_performance"],
        ["Table 4", "Comparison with reported methods template", "tables_main/Table4_comparison_with_reported_methods_TEMPLATE"],
        ["Table S1", "Complete extracted color features", "tables_supplementary/TableS1_extracted_color_features_complete.csv"],
        ["Table S2", "Concentration-level color feature summary", "tables_supplementary/TableS2_concentration_level_color_summary"],
        ["Table S3", "UV–Vis spectral descriptors", "tables_supplementary/TableS3_UVVis_spectral_descriptors"],
        ["Table S4", "Internal LOIO model performance", "tables_supplementary/TableS4_internal_LOIO_model_performance"],
        ["Table S5", "Feature importance values", "tables_supplementary/TableS5_feature_importance_values"],
    ], columns=["Item", "Description", "File base"])
    table_map.to_csv(OUTPUT / "recommended_table_map_FINAL.csv", index=False)


def main() -> None:
    set_premium_style()
    _delete_previous_generated_figures()
    print("============================================================")
    print("BORAX Q1 PREMIUM COMPLETE FIGURE + TABLE GENERATOR")
    print("============================================================")
    ganjil_folder = first_existing_folder(PHOTO_GANJIL_CANDIDATES)
    genap_folder = first_existing_folder(PHOTO_GENAP_CANDIDATES)
    uv_ganjil_folder = first_existing_folder(UVVIS_GANJIL_CANDIDATES)
    uv_genap_folder = first_existing_folder(UVVIS_GENAP_CANDIDATES)
    print(f"Photo ganjil folder : {ganjil_folder}")
    print(f"Photo genap folder  : {genap_folder}")
    print(f"UV-Vis ganjil folder: {uv_ganjil_folder}")
    print(f"UV-Vis genap folder : {uv_genap_folder}")

    manual = load_manual_roi_override(MANUAL_ROI_OVERRIDE_CSV)
    if manual:
        print(f"Manual ROI override loaded: {len(manual)} ROI entries")

    print("\n[1/9] Extracting smartphone color features...")
    out_g = extract_color_features_from_set("ganjil", ganjil_folder, GANJIL_CONC, manual)
    out_e = extract_color_features_from_set("genap", genap_folder, GENAP_CONC, manual)
    dfs = []; rois = []
    for out in [out_g, out_e]:
        if isinstance(out, tuple):
            df, rq = out
            if not df.empty: dfs.append(df)
            if not rq.empty: rois.append(rq)
    color_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    color_df = ensure_pH_column(color_df)
    roi_qc = pd.concat(rois, ignore_index=True) if rois else pd.DataFrame()
    if not color_df.empty:
        color_df.to_csv(PROC_DIR / "processed_color_features.csv", index=False)
        aggregate_color(color_df).to_csv(PROC_DIR / "concentration_level_color_summary.csv", index=False)
        print(f"Extracted {len(color_df)} ROI records from {color_df['image_file'].nunique()} images.")
    else:
        print("[WARNING] No smartphone color records were extracted.")
    if not roi_qc.empty:
        roi_qc.to_csv(TAB_SUPP / "TableS_ROI_coordinates_and_QC.csv", index=False)

    print("\n[2/9] Loading and processing UV-Vis data...")
    uv_raw = pd.concat([load_uvvis_dataset(uv_ganjil_folder, "ganjil"), load_uvvis_dataset(uv_genap_folder, "genap")], ignore_index=True)
    uv_spectra, uv_features = process_uvvis(uv_raw) if not uv_raw.empty else (pd.DataFrame(), pd.DataFrame())
    if not uv_spectra.empty:
        uv_spectra.to_csv(PROC_DIR / "uvvis_spectra_interpolated.csv", index=False)
        uv_features.to_csv(PROC_DIR / "uvvis_features.csv", index=False)
        print(f"Processed UV-Vis spectra: {uv_features.shape[0]} files/features.")
    else:
        print("UV-Vis data not found/readable. UV-Vis figures will be skipped.")

    print("\n[3/9] Generating graphical abstract...")
    create_graphical_abstract(color_df, uv_spectra)

    print("\n[4/9] Generating main figures...")
    figure_1_workflow_graphical(color_df, uv_spectra)
    figure_0_molecular_mechanism(color_df)
    figure_2_roi_and_color_gallery(color_df)
    figure_3_ph_hue_lab(color_df)
    figure_4_uvvis_top_tier(uv_spectra)
    # Descriptor-only UV-Vis figure is supplementary; Fig. 5 main is integrated into Fig. 4 output.
    figure_5_uvvis_features(uv_features)
    analytical = figure_6_analytical_calibration(color_df)
    pca_loadings, uv_corr = figure_7_chemometrics(color_df, uv_features)
    perf_ext, pred_ext, perf_int, pred_int = figure_8_ml_external(color_df)
    imp = figure_9_explainability(color_df)
    uv_bridge = figure_11_color_uvvis_correlation(color_df, uv_features)
    perm = figure_10_bland_altman_and_permutation(color_df, pred_ext, perf_ext)

    print("\n[5/9] Generating supplementary figures...")
    supplementary_figures(color_df, uv_features, perf_int, pred_int, pca_loadings, pred_ext, perf_ext)

    print("\n[6/9] Saving prediction and model tables...")
    if not perf_ext.empty:
        perf_ext.to_csv(PROC_DIR / "external_validation_performance_all_models.csv", index=False)
    if not pred_ext.empty:
        pred_ext.to_csv(PROC_DIR / "external_validation_predictions_all_models.csv", index=False)
    if not perf_int.empty:
        perf_int.to_csv(PROC_DIR / "internal_LOIO_performance_all_models.csv", index=False)
    if not pred_int.empty:
        pred_int.to_csv(PROC_DIR / "internal_LOIO_predictions_all_models.csv", index=False)
    if not imp.empty:
        imp.to_csv(TAB_SUPP / "TableS_feature_importance_values.csv", index=False)
    if not perm.empty:
        perm.to_csv(TAB_SUPP / "TableS_permutation_test_R2_distribution.csv", index=False)

    print("\n[7/9] Generating publication-style tables and table images...")
    create_main_and_supplementary_tables(color_df, roi_qc, uv_features, analytical, perf_ext, perf_int, pca_loadings, uv_corr)
    create_premium_publication_tables(color_df, uv_features, analytical, perf_ext, perf_int, imp)

    print("\n[8/9] Writing final figure/table maps...")
    _write_final_maps()

    print("\n[9/9] Writing README...")
    readme = f"""
BORAX Q1 PREMIUM COMPLETE OUTPUT PACKAGE
=======================================

Output folder:
{OUTPUT}

Main figure folder:
{FIG_MAIN}

Supplementary figure folder:
{FIG_SUPP}

Main table folder:
{TAB_MAIN}

Supplementary table folder:
{TAB_SUPP}

New in this finalized version:
1. A Graphical_Abstract figure is generated automatically.
2. Old generated figures are deleted at the start to prevent blank/duplicate numbering.
3. Fig. 4 is saved correctly from one open figure object; no closed-figure duplicate saving.
4. UV-Vis descriptors are integrated into the main UV-Vis figure and also exported as Supplementary.
5. Premium PNG/PDF/TIFF table figures are generated for main and supplementary tables.
6. Table 4 is provided as a literature-comparison template; fill values only after manual reference verification.
7. Final maps are written to recommended_main_figure_map_FINAL.csv and recommended_table_map_FINAL.csv.

Recommended manuscript mapping:
Graphical Abstract: Graphical_Abstract
Fig. 1: Workflow
Fig. 2: Mechanistic basis
Fig. 3: ROI and color evolution
Fig. 4: pH, hue, and CIELab response
Fig. 5: UV-Vis validation and descriptors
Fig. 6: Analytical calibration
Fig. 7: Chemometrics
Fig. 8: External ML validation
Fig. 9: Explainable feature relevance
Fig. 10: Bland-Altman and permutation test
Fig. 11: Smartphone-UV-Vis cross-modal agreement

ROI correction:
Inspect roi_overlays/. If any ROI is off-center, edit tables_supplementary/TableS_ROI_coordinates_and_QC.csv,
save it as manual_roi_override.csv beside this script, and rerun.
"""
    (OUTPUT / "README_Q1_premium_complete_outputs.txt").write_text(readme.strip(), encoding="utf-8")
    print("\nDone.")
    print(f"Output folder: {OUTPUT}")


# ============================================================
# FINAL VISUAL PATCH: journal-style graphical abstract and tables
# ============================================================
# This block intentionally redefines selected functions above main().
# Goals:
# 1) Graphical abstract follows the cleaner two-row Fig. 1 workflow style.
# 2) Table figures use journal style: no vertical grid lines, wrapped text,
#    compact columns, controlled decimals, and no text collision.
# 3) Large supplementary data are saved as CSV; image tables show readable excerpts.

import textwrap as _textwrap


def _fmt_cell_value(v):
    """Readable table values for publication figures."""
    if pd.isna(v):
        return ""
    if isinstance(v, (float, np.floating)):
        av = abs(float(v))
        if av == 0:
            return "0"
        if av >= 100:
            return f"{v:.1f}"
        if av >= 10:
            return f"{v:.2f}".rstrip("0").rstrip(".")
        if av >= 1:
            return f"{v:.3f}".rstrip("0").rstrip(".")
        return f"{v:.3g}"
    s = str(v)
    return s


def _pretty_header(s: str) -> str:
    """Convert technical column names into compact journal table headers."""
    mapping = {
        "concentration_mM": "Concentration\n(mM)",
        "Borax concentration (mM)": "Borax\n(mM)",
        "Measured pH": "Measured\npH",
        "selected_linear_feature": "Linear\nfeature",
        "linear_range_mM": "Linear\nrange (mM)",
        "slope": "Slope",
        "intercept": "Intercept",
        "R2": "R²",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "blank_sd": "Blank\nSD",
        "LOD_mM": "LOD\n(mM)",
        "LOQ_mM": "LOQ\n(mM)",
        "validation": "Validation",
        "model": "Model",
        "RMSE_mM": "RMSE\n(mM)",
        "MAE_mM": "MAE\n(mM)",
        "importance_mean": "Importance\nmean",
        "importance_sd": "Importance\nSD",
        "lambda_max_nm": "λmax\n(nm)",
        "Amax": "Amax",
        "AUC": "AUC",
        "Lab_a_mean": "Lab a*",
        "Lab_b_mean": "Lab b*",
        "H_mean": "Hue\n(°)",
        "R_mean": "R",
        "G_mean": "G",
        "B_mean": "B",
        "Excess_Blue_Index": "Excess blue\nindex",
        "Probe/Reagent": "Probe /\nreagent",
        "Linear range": "Linear\nrange",
    }
    return mapping.get(str(s), str(s).replace("_", "\n" if len(str(s)) > 13 else "_"))


def _wrap_text(s, width=14):
    s = _fmt_cell_value(s)
    if "\n" in s:
        return s
    if len(s) <= width:
        return s
    return "\n".join(_textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def _prepare_table_for_figure(df: pd.DataFrame, max_rows: int, wrap_width: int = 15) -> pd.DataFrame:
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows).copy()
    # Drop very long free-text columns from QC image excerpts; CSV keeps the complete data.
    long_text_cols = [c for c in d.columns if str(c).lower() in {"note", "comment", "remarks"}]
    if long_text_cols and len(d.columns) > 6:
        d = d.drop(columns=long_text_cols)
    # Shorten common validation labels.
    for c in d.columns:
        if c == "validation":
            d[c] = d[c].astype(str).str.replace("ganjil_to_genap", "A → B", regex=False).str.replace("genap_to_ganjil", "B → A", regex=False)
    d = d.apply(lambda col: col.map(_fmt_cell_value))
    d.columns = [_pretty_header(c) for c in d.columns]
    for c in d.columns:
        d[c] = d[c].map(lambda x: _wrap_text(x, width=wrap_width))
    return d


def format_table_png(df: pd.DataFrame, title: str, path: Path, max_rows: int = 25,
                     font_size: int = 8, col_width: float = 1.25) -> None:
    """Premium journal-like table figure: horizontal rules only, wrapped text."""
    if df is None or df.empty:
        return
    set_premium_style()
    d = _prepare_table_for_figure(df, max_rows=max_rows, wrap_width=16)
    nrows, ncols = d.shape

    # Keep large data tables readable; avoid huge blank canvases.
    fig_w = max(7.0, min(16.0, 1.15 * ncols + 2.4))
    row_h = 0.34 if nrows > 18 else 0.44
    fig_h = max(2.3, min(12.0, 1.05 + row_h * (nrows + 1)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)

    # Column width heuristics: distribute evenly but give text-heavy columns more space.
    weights = []
    for c in d.columns:
        cname = str(c).replace("\n", " ").lower()
        if any(k in cname for k in ["probe", "platform", "notes", "method", "feature"]):
            weights.append(1.45)
        else:
            weights.append(1.0)
    sw = sum(weights)
    col_widths = [w/sw for w in weights]

    tbl = ax.table(
        cellText=d.values,
        colLabels=d.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.36 if nrows <= 12 else 1.16)

    # Horizontal-rule style, no vertical lines.
    for (r, c), cell in tbl.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor("0.30")
        if r == 0:
            cell.set_facecolor("0.94")
            cell.set_text_props(fontweight="bold", color="black")
            cell.set_linewidth(0.85)
        else:
            cell.set_linewidth(0.38)
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F8F8F8")
        cell.PAD = 0.030

    fig.tight_layout(pad=0.8)
    for ext in ["png", "pdf", "tiff"]:
        fig.savefig(f"{path}.{ext}", dpi=600 if ext != "pdf" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_graphical_abstract(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """Clean two-row graphical abstract mirroring the main workflow figure."""
    set_premium_style()
    fig = plt.figure(figsize=(15.5, 7.4), dpi=300)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.65, 0.12, 1.15],
                  width_ratios=[1.15, 1.35, 1.05], hspace=0.35, wspace=0.30)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.suptitle("Multimodal Smartphone Colorimetry for Quantitative Borax Determination",
                 y=0.985, fontsize=16.0, fontweight="bold")

    row1 = [
        ("Purple cabbage\nanthocyanin", "natural pH probe", "#F2E8FA"),
        ("Borax-induced\nalkalization", "pH increase", "#EAF3FF"),
        ("Anthocyanin\nconversion", "color transition", "#F7F7F7"),
        ("UV–Vis\nvalidation", "spectral fingerprint", "#EAF7F0"),
    ]
    row2 = [
        ("Smartphone\nimaging", "controlled acquisition", "#FFF2DF"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab", "#EAF6FF"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP", "#F3F3F3"),
        ("Borax\nprediction", "quantitative mM", "#EAF8EC"),
    ]

    def draw_row(nodes, y, label):
        x0, x1 = 0.12, 0.92
        gap = 0.050
        w = (x1 - x0 - gap * 3) / 4
        h = 0.24
        ax.text(0.065, y, label, rotation=90, ha="center", va="center", fontsize=9.0,
                fontweight="bold", color="0.15")
        for i, (t, st, fc) in enumerate(nodes):
            x = x0 + i * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y - h/2), w, h,
                                        boxstyle="round,pad=0.012,rounding_size=0.018",
                                        fc=fc, ec="0.18", lw=1.0))
            ax.text(x + w/2, y + 0.035, t, ha="center", va="center",
                    fontsize=8.7, fontweight="bold", linespacing=0.88)
            ax.text(x + w/2, y - 0.068, st, ha="center", va="center",
                    fontsize=7.0, fontweight="bold", color="0.28")
            if i < 3:
                ax.add_patch(FancyArrowPatch((x + w + 0.010, y), (x + w + gap - 0.010, y),
                                             arrowstyle="-|>", mutation_scale=12, lw=1.0, color="0.22"))
    draw_row(row1, 0.70, "Chemical\nvalidation")
    draw_row(row2, 0.35, "Digital\nquantification")
    ax.plot([0.10, 0.94], [0.525, 0.525], ls=(0, (3, 3)), lw=0.75, color="0.72")
    ax.text(0.52, 0.548, "UV–Vis validation supports smartphone-based quantification",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="0.38")
    ax.text(0.52, 0.075,
            "Natural anthocyanin probe → smartphone color signatures → independently validated, explainable borax prediction",
            ha="center", va="center", fontsize=10.3, fontweight="bold")

    # Bottom-left: data-derived color strip.
    ax1 = fig.add_subplot(gs[2, 0])
    ax1.set_title("Smartphone color trajectory", fontsize=9.0, pad=6)
    if not color_df.empty:
        means = color_df.groupby("concentration_mM")[["R_mean", "G_mean", "B_mean"]].mean().reindex(ALL_CONC).dropna()
        arr = np.array([np.clip(row.values/255, 0, 1) for _, row in means.iterrows()])[None, :, :]
        ax1.imshow(arr, aspect="auto", extent=[float(means.index.min()), float(means.index.max()), 0, 1])
        for x in means.index:
            ax1.axvline(float(x), color="white", lw=0.25, alpha=0.6)
    ax1.set_yticks([]); ax1.set_xlabel("Borax concentration (mM)")
    ax1.set_xlim(0, 20); ax1.set_xticks([0, 5, 10, 15, 20])
    for s in ax1.spines.values(): s.set_visible(False)

    # Bottom-middle: compact UV-Vis validation.
    ax2 = fig.add_subplot(gs[2, 1])
    ax2.set_title("Representative UV–Vis fingerprints", fontsize=9.0, pad=6)
    if not uv_spectra.empty:
        avg = _get_avg_uvvis(uv_spectra)
        for conc in [0, 10, 20]:
            g = avg[np.isclose(avg["concentration_mM"].astype(float), float(conc))]
            if not g.empty:
                ax2.plot(g["wavelength_nm"], g["absorbance"], lw=1.2, label=f"{conc} mM")
    ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Absorbance")
    ax2.tick_params(labelsize=7); ax2.legend(frameon=False, fontsize=7, loc="upper right")
    ax2.grid(alpha=0.12)

    # Bottom-right: clean prediction stack without long arrows.
    ax3 = fig.add_subplot(gs[2, 2])
    ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_title("Prediction layer", fontsize=9.0, pad=6)
    steps = [("Color features", "RGB, HSV, CIELab"), ("Validation", "Set A ↔ Set B"),
             ("Attribution", "Permutation / SHAP"), ("Prediction", "mM output")]
    yy = [0.78, 0.56, 0.34, 0.12]
    for i, (main, sub) in enumerate(steps):
        ax3.add_patch(FancyBboxPatch((0.18, yy[i]-0.065), 0.64, 0.13,
                                     boxstyle="round,pad=0.012,rounding_size=0.012",
                                     fc="white", ec="0.25", lw=0.8))
        ax3.text(0.50, yy[i]+0.018, main, ha="center", va="center", fontsize=7.4, fontweight="bold")
        ax3.text(0.50, yy[i]-0.027, sub, ha="center", va="center", fontsize=6.4, fontweight="bold", color="0.35")
        if i < len(steps)-1:
            ax3.annotate("", xy=(0.50, yy[i+1]+0.077), xytext=(0.50, yy[i]-0.078),
                         arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.25"))

    fig.savefig(OUTPUT / "Graphical_Abstract.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_MAIN / "Graphical_Abstract.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "Graphical_Abstract.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_MAIN / "Graphical_Abstract.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "Graphical_Abstract.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_MAIN / "Graphical_Abstract.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_premium_publication_tables(color_df: pd.DataFrame, uv_features: pd.DataFrame,
                                      analytical: pd.DataFrame, perf_ext: pd.DataFrame,
                                      perf_int: pd.DataFrame, imp: pd.DataFrame) -> None:
    """Publication-ready main and SI tables with readable figure versions."""
    table1 = pd.DataFrame({"Borax concentration (mM)": list(PH_MAP.keys()), "Measured pH": list(PH_MAP.values())})
    table1.to_csv(TAB_MAIN / "Table1_borax_standards_and_pH.csv", index=False)
    format_table_png(table1, "Table 1. Borax standards and measured pH", TAB_MAIN / "Table1_borax_standards_and_pH", max_rows=25, font_size=8.2)

    if analytical is not None and not analytical.empty:
        analytical2 = analytical.copy()
        # Keep the visual table compact; CSV remains complete.
        analytical2.to_csv(TAB_MAIN / "Table2_analytical_figures_of_merit.csv", index=False)
        format_table_png(analytical2, "Table 2. Analytical figures of merit", TAB_MAIN / "Table2_analytical_figures_of_merit", max_rows=12, font_size=7.4)

    if perf_ext is not None and not perf_ext.empty:
        perf_clean = perf_ext.copy()
        if "validation" in perf_clean.columns and "RMSE_mM" in perf_clean.columns:
            table3 = perf_clean.sort_values(["validation", "RMSE_mM", "MAE_mM"]).groupby("validation", as_index=False).head(3)
        else:
            table3 = perf_clean.head(8)
        table3.to_csv(TAB_MAIN / "Table3_external_validation_performance.csv", index=False)
        format_table_png(table3, "Table 3. Independent external validation performance", TAB_MAIN / "Table3_external_validation_performance", max_rows=8, font_size=7.8)

    # A compact template only; the CSV is the editable source for actual literature values.
    if analytical is not None and not analytical.empty and "LOD_mM" in analytical.columns:
        lod_val = analytical["LOD_mM"].iloc[0]
        rng = analytical["linear_range_mM"].iloc[0] if "linear_range_mM" in analytical.columns else "0–20"
    else:
        lod_val, rng = "", "0–20"
    comparison = pd.DataFrame([
        ["This work", "Purple cabbage\nanthocyanin", "Smartphone colorimetry\n+ UV–Vis validation", rng, lod_val, "Natural probe;\nXAI-supported"],
        ["Literature 1", "Fill from\nreference", "Fill from\nreference", "Fill", "Fill", "Fill"],
        ["Literature 2", "Fill from\nreference", "Fill from\nreference", "Fill", "Fill", "Fill"],
        ["Literature 3", "Fill from\nreference", "Fill from\nreference", "Fill", "Fill", "Fill"],
    ], columns=["Method", "Probe/Reagent", "Platform", "Linear range", "LOD", "Notes"])
    comparison.to_csv(TAB_MAIN / "Table4_comparison_with_reported_methods_TEMPLATE.csv", index=False)
    format_table_png(comparison, "Table 4. Comparison with reported borax sensing methods", TAB_MAIN / "Table4_comparison_with_reported_methods_TEMPLATE", max_rows=8, font_size=7.6)

    # Supplementary tables: save full CSVs but make readable visual excerpts.
    if not color_df.empty:
        color_summary = aggregate_color(color_df)
        color_summary.to_csv(TAB_SUPP / "TableS2_concentration_level_color_summary.csv", index=False)
        keep_cols = [c for c in ["concentration_mM", "pH", "H_mean", "R_mean", "G_mean", "B_mean", "Lab_a_mean", "Lab_b_mean", "Excess_Blue_Index"] if c in color_summary.columns]
        format_table_png(color_summary[keep_cols], "Table S2. Concentration-level color feature summary", TAB_SUPP / "TableS2_concentration_level_color_summary", max_rows=22, font_size=6.4)

    if uv_features is not None and not uv_features.empty:
        uv_mean = uv_features.groupby("concentration_mM", as_index=False).mean(numeric_only=True)
        uv_mean.to_csv(TAB_SUPP / "TableS3_UVVis_spectral_descriptors.csv", index=False)
        keep = [c for c in ["concentration_mM", "lambda_max_nm", "Amax", "AUC", "A420", "A520", "A620"] if c in uv_mean.columns]
        format_table_png(uv_mean[keep], "Table S3. UV–Vis spectral descriptors", TAB_SUPP / "TableS3_UVVis_spectral_descriptors", max_rows=22, font_size=6.7)

    if perf_int is not None and not perf_int.empty:
        perf_int.to_csv(TAB_SUPP / "TableS4_internal_LOIO_model_performance.csv", index=False)
        format_table_png(perf_int.head(10), "Table S4. Internal leave-one-image-out model performance", TAB_SUPP / "TableS4_internal_LOIO_model_performance", max_rows=10, font_size=7.8)

    if imp is not None and not imp.empty:
        imp.to_csv(TAB_SUPP / "TableS5_feature_importance_values.csv", index=False)
        format_table_png(imp.head(15), "Table S5. Feature importance values", TAB_SUPP / "TableS5_feature_importance_values", max_rows=15, font_size=7.4)



# =============================================================================
# FINAL INTEGRATED PATCH: GRAPHICAL ABSTRACT + BOOKTABS TABLES
# =============================================================================
# -*- coding: utf-8 -*-
"""
PATCH FINAL Q1 TOP-TIER
Graphical Abstract + Booktabs Tables for borax smartphone colorimetry program.

Cara pakai:
1) Simpan file ini di folder yang sama dengan program dasar:
   borax_Q1_TOP_TIER_FINAL_Q1_PREMIUM_ALL_FIGURES_REVISED_FINAL.py
2) Pada bagian PALING BAWAH program dasar, sebelum:
      if __name__ == "__main__":
          main()
   tambahkan:
      from borax_Q1_PATCH_GRAPHICAL_TABLES_BOOKTABS import *
3) Jalankan ulang program dasar.

Catatan:
Patch ini tidak mengubah ROI extraction, UV-Vis processing, ML validation, dan figure utama lain.
Patch hanya mengganti:
- graphical abstract menjadi layout dua-baris yang bersih
- format_table_png menjadi booktabs style tanpa garis vertikal
- create_main_and_supplementary_tables agar tabel tidak bertabrakan
"""


import math
import textwrap
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec


# ============================================================
# GENERAL HELPERS
# ============================================================

def _q1_fmt_number(x):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
        if abs(x) >= 100:
            return f"{x:.1f}"
        if abs(x) >= 10:
            return f"{x:.2f}".rstrip("0").rstrip(".")
        if abs(x) >= 1:
            return f"{x:.3f}".rstrip("0").rstrip(".")
        return f"{x:.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def _wrap_text(x, width=18):
    s = "" if pd.isna(x) else str(x)
    s = s.replace("_", " ")
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def _clean_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "concentration_mM": "Borax\n(mM)",
        "Borax concentration (mM)": "Borax\n(mM)",
        "pH": "pH",
        "selected_linear_feature": "Feature",
        "linear_range_mM": "Linear\nrange",
        "slope": "Slope",
        "intercept": "Intercept",
        "R2": "R²",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "blank_sd": "Blank SD",
        "LOD_mM": "LOD\n(mM)",
        "LOQ_mM": "LOQ\n(mM)",
        "validation": "Validation",
        "model": "Model",
        "RMSE_mM": "RMSE\n(mM)",
        "MAE_mM": "MAE\n(mM)",
        "importance_mean": "Importance\nmean",
        "importance_sd": "Importance\nSD",
        "lambda_max_nm": "λmax\n(nm)",
    }
    out = df.copy()
    out = out.rename(columns={c: rename.get(c, c.replace("_", "\n")) for c in out.columns})
    return out


def _prepare_table_for_figure(df: pd.DataFrame, max_rows=25, wrap_width=18) -> pd.DataFrame:
    d = df.copy().head(max_rows)

    # Hindari konflik concentration_mM ketika groupby menghasilkan kolom sama.
    d = d.loc[:, ~d.columns.duplicated()].copy()

    # Format angka.
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(_q1_fmt_number)
        else:
            d[c] = d[c].map(lambda x: _wrap_text(x, wrap_width))

    d = _clean_table_columns(d)
    return d


def format_table_png(
    df: pd.DataFrame,
    title: str,
    path: Path,
    max_rows: int = 25,
    font_size: int = 8,
    wrap_width: int = 18,
    row_height: float = 0.38,
    min_width: float = 7.0,
    max_width: float = 15.5,
) -> None:
    """
    Booktabs-style table figure:
    - no vertical grid lines
    - only top/mid/bottom horizontal rules
    - wrapped text
    - controlled row height
    """
    if df is None or df.empty:
        return

    d = _prepare_table_for_figure(df, max_rows=max_rows, wrap_width=wrap_width)
    nrows, ncols = d.shape

    # Kolom panjang diberi ruang lebih besar.
    fig_w = float(np.clip(1.35 * ncols + 1.2, min_width, max_width))
    fig_h = float(np.clip(row_height * (nrows + 2.2), 2.4, 16.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, weight="bold", pad=10)

    table = ax.table(
        cellText=d.values,
        colLabels=d.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.35)

    # Booktabs style.
    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_linewidth(0.0)
        cell.PAD = 0.075

        if r == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
            cell.set_facecolor("#F4F4F4")
            cell.set_text_props(weight="bold", color="black")
        else:
            cell.visible_edges = "B"
            cell.set_linewidth(0.28)
            cell.set_edgecolor("0.55")
            if r % 2 == 0:
                cell.set_facecolor("#FAFAFA")
            else:
                cell.set_facecolor("white")

    # Bottom rule lebih tebal.
    last_row = nrows
    for c in range(ncols):
        cell = table[(last_row, c)]
        cell.visible_edges = "B"
        cell.set_linewidth(0.8)
        cell.set_edgecolor("0.20")

    fig.savefig(str(path) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# GRAPHICAL ABSTRACT REPLACEMENT
# ============================================================

def _mean_rgb_table(color_df: pd.DataFrame) -> pd.DataFrame:
    if color_df is None or color_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["R_mean", "G_mean", "B_mean"] if c in color_df.columns]
    if len(cols) != 3:
        return pd.DataFrame()
    return (
        color_df.groupby("concentration_mM")[cols]
        .mean()
        .sort_index()
    )


def _draw_color_strip(ax, color_df: pd.DataFrame):
    means = _mean_rgb_table(color_df)
    ax.set_axis_off()
    if means.empty:
        ax.text(0.5, 0.5, "Smartphone color trajectory", ha="center", va="center", fontsize=10)
        return

    concs = [int(c) for c in means.index if 0 <= int(c) <= 20]
    for i, c in enumerate(concs):
        rgb = np.clip(means.loc[c].values / 255.0, 0, 1)
        ax.add_patch(Rectangle((i, 0.30), 1, 0.34, fc=rgb, ec="white", lw=0.35))

    ax.set_xlim(0, len(concs))
    ax.set_ylim(0, 1)
    ax.text(0, 0.12, "0 mM", ha="left", va="center", fontsize=9, weight="bold")
    ax.text(len(concs), 0.12, "20 mM", ha="right", va="center", fontsize=9, weight="bold")
    ax.text(len(concs) / 2, 0.85, "data-derived smartphone color trajectory",
            ha="center", va="center", fontsize=9.5)


def _draw_uvvis_inset(ax, uv_spectra: pd.DataFrame):
    if uv_spectra is None or uv_spectra.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "UV-Vis validation", ha="center", va="center", fontsize=10)
        return

    avg = (
        uv_spectra.groupby(["concentration_mM", "wavelength_nm"])["absorbance"]
        .mean()
        .reset_index()
    )
    for c in [0, 10, 20]:
        sub = avg[np.isclose(avg["concentration_mM"], c)]
        if not sub.empty:
            ax.plot(sub["wavelength_nm"], sub["absorbance"], lw=1.4, label=f"{c} mM")

    ax.set_title("UV-Vis validation", fontsize=9.5, pad=3)
    ax.set_xlabel("Wavelength (nm)", fontsize=8)
    ax.set_ylabel("Abs.", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.12)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def _draw_workflow_box(ax, x, y, w, h, title, subtitle, fc):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        fc=fc, ec="0.15", lw=1.15
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.60, title,
            ha="center", va="center", fontsize=9.2, weight="bold", linespacing=0.90)
    ax.text(x + w / 2, y + h * 0.30, subtitle,
            ha="center", va="center", fontsize=7.8, color="0.25")
    return box


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=9.5,
        lw=1.05,
        color="0.20",
        shrinkA=3,
        shrinkB=3
    ))


def graphical_abstract_q1(color_df: pd.DataFrame, uv_spectra: pd.DataFrame) -> None:
    """
    Graphical abstract baru: dua baris workflow + color trajectory + UV-Vis inset.
    Layout mengikuti Figure 1, tidak ada kotak saling bertabrakan.
    """
    global FIG_MAIN
    try:
        outdir = FIG_MAIN
    except NameError:
        outdir = Path("figures_main")
        outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.8, 7.2))
    gs = GridSpec(
        3, 3,
        height_ratios=[1.35, 0.75, 0.95],
        width_ratios=[1.05, 1.05, 0.90],
        hspace=0.28,
        wspace=0.34,
        figure=fig
    )

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.set_title(
        "Multimodal smartphone colorimetry for quantitative borax determination",
        fontsize=18, weight="bold", pad=12
    )

    left = 0.04
    right = 0.96
    gap = 0.035
    n = 4
    w = (right - left - gap * (n - 1)) / n
    h = 0.22
    y_top = 0.55
    y_bottom = 0.20

    colors_top = ["#EAF3FB", "#F1E8F7", "#F7F7F7", "#EAF7F0"]
    colors_bottom = ["#FFF3DE", "#EAF5FB", "#F7F7F7", "#EAF8EA"]

    top = [
        ("Purple cabbage\nanthocyanin", "natural pH probe"),
        ("Borax-induced\nalkalization", "pH increase"),
        ("Color\ntransition", "purple → blue-green"),
        ("UV-Vis\nvalidation", "spectral fingerprint"),
    ]
    bottom = [
        ("Smartphone\nimaging", "controlled acquisition"),
        ("ROI + color\nfeatures", "RGB–HSV–CIELab"),
        ("Chemometrics\n+ XAI", "PCA, ML, SHAP"),
        ("Borax\nprediction", "quantitative mM"),
    ]

    top_centers = []
    bottom_centers = []

    for i, (txt, sub) in enumerate(top):
        x = left + i * (w + gap)
        _draw_workflow_box(ax, x, y_top, w, h, txt, sub, colors_top[i])
        top_centers.append((x + w / 2, y_top + h / 2))
        if i < n - 1:
            _arrow(ax, x + w, y_top + h / 2, x + w + gap, y_top + h / 2)

    for i, (txt, sub) in enumerate(bottom):
        x = left + i * (w + gap)
        _draw_workflow_box(ax, x, y_bottom, w, h, txt, sub, colors_bottom[i])
        bottom_centers.append((x + w / 2, y_bottom + h / 2))
        if i < n - 1:
            _arrow(ax, x + w, y_bottom + h / 2, x + w + gap, y_bottom + h / 2)

    ax.plot([left, right], [0.47, 0.47], ls=(0, (3, 3)), lw=0.8, color="0.70")
    ax.text(0.5, 0.485, "spectroscopic validation supports smartphone-based quantification",
            ha="center", va="bottom", fontsize=8.4, color="0.35")
    ax.text(0.5, 0.065,
            "Natural anthocyanin chemistry + smartphone imaging + UV-Vis validation + explainable machine learning",
            ha="center", va="center", fontsize=11.2, weight="bold")

    ax_strip = fig.add_subplot(gs[1, 0:2])
    _draw_color_strip(ax_strip, color_df)

    ax_uv = fig.add_subplot(gs[1:, 2])
    _draw_uvvis_inset(ax_uv, uv_spectra)

    ax_note = fig.add_subplot(gs[2, 0:2])
    ax_note.set_axis_off()
    ax_note.text(
        0.5, 0.55,
        "Borax increases alkalinity, transforming anthocyanin color signatures that are captured by smartphone ROIs and validated by UV-Vis spectra.",
        ha="center", va="center", fontsize=11.0, weight="bold", wrap=True
    )

    fig.savefig(str(outdir / "Graphical_Abstract_Q1_premium.png"), dpi=600, bbox_inches="tight")
    fig.savefig(str(outdir / "Graphical_Abstract_Q1_premium.pdf"), bbox_inches="tight")
    fig.savefig(str(outdir / "Graphical_Abstract_Q1_premium.tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# TABLE GENERATOR REPLACEMENT
# ============================================================

def aggregate_color(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "concentration_mM" not in work.columns:
        return pd.DataFrame()
    num_cols = [
        c for c in work.columns
        if pd.api.types.is_numeric_dtype(work[c]) and c != "concentration_mM"
    ]
    agg = (
        work.groupby("concentration_mM", as_index=False)[num_cols]
        .mean()
        .sort_values("concentration_mM")
    )
    return agg


def create_main_and_supplementary_tables(
    color_df: pd.DataFrame,
    roi_qc: pd.DataFrame,
    uv_features: pd.DataFrame,
    analytical: pd.DataFrame,
    perf_ext: pd.DataFrame,
    perf_int: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    uv_corr: pd.DataFrame,
) -> None:
    """
    Final Q1 table output:
    - CSV full data tetap disimpan
    - PNG/PDF table figure dibuat ringkas, tanpa garis vertikal, text wrapped
    """
    global TAB_MAIN, TAB_SUPP
    try:
        TAB_MAIN.mkdir(parents=True, exist_ok=True)
        TAB_SUPP.mkdir(parents=True, exist_ok=True)
    except NameError:
        TAB_MAIN = Path("tables_main"); TAB_SUPP = Path("tables_supplementary")
        TAB_MAIN.mkdir(parents=True, exist_ok=True); TAB_SUPP.mkdir(parents=True, exist_ok=True)

    # Table 1
    try:
        ph_map = PH_MAP
    except NameError:
        ph_map = {i: np.nan for i in range(21)}
    tbl1 = pd.DataFrame({
        "Borax concentration (mM)": list(ph_map.keys()),
        "Measured pH": list(ph_map.values()),
    })
    tbl1.to_csv(TAB_MAIN / "Table1_borax_standards_and_pH.csv", index=False)
    format_table_png(tbl1, "Table 1. Borax standards and measured pH",
                     TAB_MAIN / "Table1_borax_standards_and_pH",
                     max_rows=25, font_size=8.2, wrap_width=16, row_height=0.32)

    # Table 2
    if analytical is not None and not analytical.empty:
        ana = analytical.copy()
        keep = [c for c in [
            "selected_linear_feature", "analytical_feature", "linear_range_mM",
            "slope", "slope_deg_per_mM", "intercept", "R2", "RMSE", "MAE",
            "blank_sd", "LOD_mM", "LOQ_mM"
        ] if c in ana.columns]
        if keep:
            ana = ana[keep]
        ana.to_csv(TAB_MAIN / "Table2_analytical_figures_of_merit.csv", index=False)
        format_table_png(ana, "Table 2. Analytical figures of merit",
                         TAB_MAIN / "Table2_analytical_figures_of_merit",
                         max_rows=8, font_size=8.0, wrap_width=14, row_height=0.45)

    # Table 3
    if perf_ext is not None and not perf_ext.empty:
        ext = perf_ext.copy()
        if "validation" in ext.columns and "RMSE_mM" in ext.columns:
            ext = ext.sort_values(["validation", "RMSE_mM"]).groupby("validation", as_index=False).head(5)
        keep = [c for c in ["validation", "model", "R2", "RMSE_mM", "MAE_mM"] if c in ext.columns]
        if keep:
            ext = ext[keep]
        ext.to_csv(TAB_MAIN / "Table3_external_validation_performance.csv", index=False)
        format_table_png(ext, "Table 3. Independent external validation performance",
                         TAB_MAIN / "Table3_external_validation_performance",
                         max_rows=12, font_size=8.0, wrap_width=18, row_height=0.40)

    # Table 4 template: do not overcrowd PNG
    if analytical is not None and not analytical.empty:
        lod = analytical["LOD_mM"].iloc[0] if "LOD_mM" in analytical.columns else ""
        rng = analytical["linear_range_mM"].iloc[0] if "linear_range_mM" in analytical.columns else ""
    else:
        lod, rng = "", ""
    tbl4 = pd.DataFrame([
        ["This work", "Purple cabbage\nanthocyanin", "Smartphone colorimetry\n+ UV-Vis validation", rng, lod, "Natural probe;\nXAI-supported"],
        ["Literature 1", "Fill from reference", "Fill from reference", "Fill", "Fill", "Fill"],
        ["Literature 2", "Fill from reference", "Fill from reference", "Fill", "Fill", "Fill"],
        ["Literature 3", "Fill from reference", "Fill from reference", "Fill", "Fill", "Fill"],
    ], columns=["Method", "Probe/Reagent", "Platform", "Linear range", "LOD", "Notes"])
    tbl4.to_csv(TAB_MAIN / "Table4_comparison_template.csv", index=False)
    format_table_png(tbl4, "Table 4. Comparison with reported borax sensing methods",
                     TAB_MAIN / "Table4_comparison_template",
                     max_rows=8, font_size=8.0, wrap_width=16, row_height=0.55, max_width=13.5)

    # Supplementary CSV
    if color_df is not None and not color_df.empty:
        color_df.to_csv(TAB_SUPP / "TableS1_extracted_color_features_full.csv", index=False)
        agg = aggregate_color(color_df)
        agg.to_csv(TAB_SUPP / "TableS2_concentration_level_color_summary_full.csv", index=False)

        # Compact display version
        show_cols = [c for c in ["concentration_mM", "pH", "H_mean", "R_mean", "G_mean", "B_mean", "Lab_a_mean", "Lab_b_mean", "Excess_Blue_Index"] if c in agg.columns]
        if show_cols:
            format_table_png(agg[show_cols], "Table S2. Concentration-level color feature summary",
                             TAB_SUPP / "TableS2_concentration_level_color_summary",
                             max_rows=25, font_size=6.8, wrap_width=10, row_height=0.31, max_width=14.0)

    if roi_qc is not None and not roi_qc.empty:
        roi_qc.to_csv(TAB_SUPP / "TableS3_ROI_coordinates_and_QC_full.csv", index=False)
        keep = [c for c in ["dataset", "image_file", "replicate", "concentration_mM", "roi_cx", "roi_cy", "roi_radius"] if c in roi_qc.columns]
        show = roi_qc[keep].head(25) if keep else roi_qc.head(25)
        format_table_png(show, "Table S3. ROI coordinates and QC excerpt",
                         TAB_SUPP / "TableS3_ROI_coordinates_and_QC_excerpt",
                         max_rows=25, font_size=5.9, wrap_width=12, row_height=0.32, max_width=15.0)

    if uv_features is not None and not uv_features.empty:
        uv_features.to_csv(TAB_SUPP / "TableS4_UVVis_spectral_descriptors_full.csv", index=False)
        keep = [c for c in ["concentration_mM", "lambda_max_nm", "Amax", "AUC", "A420", "A520", "A620"] if c in uv_features.columns]
        show = uv_features[keep].copy() if keep else uv_features.copy()
        if "dataset" in uv_features.columns:
            # average by concentration for display
            num = [c for c in show.columns if c != "concentration_mM" and pd.api.types.is_numeric_dtype(show[c])]
            show = show.groupby("concentration_mM", as_index=False)[num].mean()
        format_table_png(show, "Table S4. UV-Vis spectral descriptors",
                         TAB_SUPP / "TableS4_UVVis_spectral_descriptors",
                         max_rows=25, font_size=7.0, wrap_width=11, row_height=0.32, max_width=13.5)

    if perf_int is not None and not perf_int.empty:
        perf_int.to_csv(TAB_SUPP / "TableS5_internal_LOIO_model_performance.csv", index=False)
        keep = [c for c in ["model", "R2", "RMSE_mM", "MAE_mM"] if c in perf_int.columns]
        show = perf_int[keep] if keep else perf_int
        format_table_png(show, "Table S5. Internal leave-one-image-out model performance",
                         TAB_SUPP / "TableS5_internal_LOIO_model_performance",
                         max_rows=12, font_size=8.0, wrap_width=16, row_height=0.38)

    if pca_loadings is not None and not pca_loadings.empty:
        pca_loadings.to_csv(TAB_SUPP / "TableS6_PCA_loadings.csv", index=False)

    if uv_corr is not None and not uv_corr.empty:
        uv_corr.to_csv(TAB_SUPP / "TableS7_color_UVVis_correlations.csv", index=False)


# ============================================================
# OPTIONAL HOOK
# ============================================================

def run_final_graphical_abstract_if_possible(color_df=None, uv_spectra=None):
    try:
        graphical_abstract_q1(color_df, uv_spectra)
        print("Graphical abstract Q1 premium regenerated.")
    except Exception as e:
        print(f"[WARNING] Graphical abstract was not generated: {e}")

if __name__ == "__main__":
    main()
