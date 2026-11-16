"""
Step 3 — Random Forest susceptibility model with spatial cross-validation.

Reads:
  data/training_points_matched_wgs84.geojson — 76 labelled points (38 + 38)
  data/rudraprayag_features.tif              — 6-band feature raster (optional,
                                               for scoring the whole domain)

Writes:
  data/auc_across_draws.csv                  — per-draw AUC table
  data/rudraprayag_susceptibility.tif        — averaged susceptibility (20 draws)
  data/rudraprayag_susceptibility_sd.tif     — model disagreement (std dev)
  outputs/fig4_model_evaluation.png          — evaluation figure
  outputs/fig5_susceptibility_map.png        — susceptibility map

Key design choices (from original notebook 03_model.ipynb):
  - Terrain features only (elevation, slope, aspect_sin, aspect_cos, dist_river)
    to avoid circularity with NDVI / land cover, which can describe the scar itself.
  - Matched stable-point sampling: one stable point within 3 km of each landslide.
  - Spatial cross-validation: 5-fold Leave-One-Group-Out on k-means blocks.
  - 20 draws of stable points (seeds 100–119); results averaged across draws.
  - Final map = mean of 20 models' predictions over the whole domain.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "outputs"

TERRAIN_FEATURES = ["elevation", "slope", "aspect_sin", "aspect_cos", "dist_river"]
BAND_NAMES = ["elevation", "slope", "aspect", "ndvi", "landcover", "dist_river"]
N_DRAWS = 20
N_BLOCKS = 5
RF_KWARGS = dict(n_estimators=100, min_samples_leaf=2, n_jobs=-1)


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_points(data_dir: Path) -> pd.DataFrame:
    """Load training points and add engineered features."""
    gdf = gpd.read_file(data_dir / "training_points_matched_wgs84.geojson").to_crs("EPSG:32644")
    df = pd.DataFrame()
    df["label"] = gdf["label"].astype(int)
    df["x"] = gdf.geometry.x
    df["y"] = gdf.geometry.y
    for b in BAND_NAMES:
        if b in gdf.columns:
            df[b] = gdf[b].astype(float)
        else:
            df[b] = np.nan
    # aspect → circular encoding
    df["aspect_sin"] = np.sin(np.deg2rad(df["aspect"].fillna(0)))
    df["aspect_cos"] = np.cos(np.deg2rad(df["aspect"].fillna(0)))
    return df


def _make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(**RF_KWARGS, random_state=seed)


def _spatial_cv_auc(df: pd.DataFrame, features: list[str], seed: int) -> float:
    """Leave-one-block-out AUC with a fresh k-means block layout per draw."""
    blocks = KMeans(n_clusters=N_BLOCKS, random_state=seed, n_init=10).fit_predict(
        df[["x", "y"]]
    )
    labels = df["label"].values
    prob = cross_val_predict(
        _make_rf(seed), df[features].fillna(0), labels,
        groups=blocks, cv=LeaveOneGroupOut(), method="predict_proba",
    )[:, 1]
    return roc_auc_score(labels, prob)


def _run_20_draws(df: pd.DataFrame) -> pd.DataFrame:
    """20-draw AUC table: 'original matched' design, seeds 100–119."""
    rows = []
    for draw in range(N_DRAWS):
        seed = 100 + draw
        auc = _spatial_cv_auc(df, TERRAIN_FEATURES, seed)
        rows.append({"draw": draw, "seed": seed, "auc": auc,
                     "design": "original matched"})
        print(f"  Draw {draw+1:2d}/20  seed={seed}  AUC={auc:.3f}")
    return pd.DataFrame(rows)


def _score_domain(df: pd.DataFrame, tif_path: Path) -> tuple[np.ndarray, dict, np.ndarray]:
    """Score every valid pixel using 20 models (one per draw), return mean + sd."""
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        arrays = src.read()          # (6, H, W)
        H, W = arrays.shape[1], arrays.shape[2]

    elev = arrays[0]
    valid = (elev > 0) & (elev < 3500) & (arrays[4] != 70) & (arrays[4] != 80)
    aspect = arrays[2]
    Xs = pd.DataFrame({
        "elevation":  arrays[0].ravel(),
        "slope":      arrays[1].ravel(),
        "aspect_sin": np.sin(np.deg2rad(aspect.ravel())),
        "aspect_cos": np.cos(np.deg2rad(aspect.ravel())),
        "dist_river": arrays[5].ravel(),
    }).fillna(0)

    sums = np.zeros(H * W)
    sq_sums = np.zeros(H * W)
    for draw in range(N_DRAWS):
        seed = 100 + draw
        rf = _make_rf(seed)
        rf.fit(df[TERRAIN_FEATURES].fillna(0), df["label"])
        prob = rf.predict_proba(Xs)[:, 1]
        sums += prob
        sq_sums += prob ** 2
        print(f"  Scored draw {draw+1:2d}/20")

    mean_prob = (sums / N_DRAWS).reshape(H, W)
    sd_prob = np.sqrt(np.maximum(sq_sums / N_DRAWS - (sums / N_DRAWS) ** 2, 0)).reshape(H, W)
    mean_prob[~valid] = np.nan
    sd_prob[~valid] = np.nan

    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan)
    return mean_prob, out_profile, sd_prob


# ── plotting ─────────────────────────────────────────────────────────────────

def _plot_evaluation(auc_df: pd.DataFrame, df: pd.DataFrame, out_path: Path) -> None:
    """Figure 4 — AUC box plot (20 draws) and feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) AUC distribution across draws
    ax = axes[0]
    ax.boxplot(auc_df["auc"], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor="#4a90d9", alpha=0.7))
    ax.set_ylabel("Within-block AUC")
    ax.set_title("(a) Model skill — 20 draws\n(spatial cross-validation, terrain model)")
    ax.set_xlim(0.3, 1.8)
    ax.set_ylim(0.2, 1.0)
    ax.set_xticks([])
    mean_auc = auc_df["auc"].mean()
    ax.axhline(mean_auc, color="red", linestyle="--", linewidth=1,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="Chance (0.5)")
    ax.legend(fontsize=9)

    # (b) feature importances (single model for illustration)
    ax2 = axes[1]
    rf_single = RandomForestClassifier(**RF_KWARGS, random_state=42)
    rf_single.fit(df[TERRAIN_FEATURES].fillna(0), df["label"])
    feat_imp = pd.Series(rf_single.feature_importances_, index=TERRAIN_FEATURES).sort_values()
    colors = ["#4a90d9" if imp == feat_imp.max() else "#a6c8e8" for imp in feat_imp]
    ax2.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="k", linewidth=0.5)
    ax2.set_xlabel("Mean decrease in impurity")
    ax2.set_title("(b) Feature importance (single model, seed 42)")
    ax2.axvline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4 saved → {out_path}")


def _plot_susceptibility(susc: np.ndarray, tif_path: Path, out_path: Path) -> None:
    """Figure 5 — susceptibility map with quintile class breaks."""
    from rasterio.plot import plotting_extent
    with rasterio.open(tif_path) as src:
        extent = plotting_extent(src)

    flat = susc[~np.isnan(susc)]
    breaks = np.quantile(flat, [0.2, 0.4, 0.6, 0.8])
    classes = np.digitize(susc, breaks)   # 0=very low ... 4=very high
    classes = classes.astype(float)
    classes[np.isnan(susc)] = np.nan

    cmap = plt.get_cmap("RdYlGn_r", 5)
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(classes, extent=extent, cmap=cmap, origin="upper",
                   vmin=-0.5, vmax=4.5)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["Very low", "Low", "Moderate", "High", "Very high"], fontsize=9)
    ax.set_title("Landslide susceptibility — Rudraprayag\n(average of 20 Random Forest models)", fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 5 saved → {out_path}")


# ── main ────────────────────────────────────────────────────────────────────

def run_model(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading training points ===")
    df = _load_points(data_dir)
    print(f"Points: {len(df)} ({df['label'].sum()} landslides, {(df['label']==0).sum()} stable)")
    print(f"Feature columns available: {[c for c in TERRAIN_FEATURES if c in df.columns]}")

    # check all terrain features are present
    missing = [f for f in TERRAIN_FEATURES if f not in df.columns or df[f].isna().all()]
    if missing:
        print(
            f"\nWarning: features {missing} are missing or all-NaN.\n"
            "They will be filled with 0. Run step 1 (features.py) and step 2 (inventory.py)\n"
            "to get proper raster-sampled values."
        )

    print("\n=== 20-draw spatial CV AUC ===")
    auc_df = _run_20_draws(df)
    auc_path = data_dir / "auc_across_draws.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"\nAUC summary: mean={auc_df['auc'].mean():.3f}  "
          f"95% range [{auc_df['auc'].quantile(0.025):.3f}–{auc_df['auc'].quantile(0.975):.3f}]")
    print(f"Saved AUC table → {auc_path}")

    print("\n=== Generating evaluation figure ===")
    _plot_evaluation(auc_df, df, out_dir / "fig4_model_evaluation.png")

    # score whole domain if the feature raster exists
    tif_path = data_dir / "rudraprayag_features.tif"
    if tif_path.exists():
        print("\n=== Scoring full domain (20 models) ===")
        mean_susc, profile, sd_susc = _score_domain(df, tif_path)

        susc_tif = data_dir / "rudraprayag_susceptibility.tif"
        with rasterio.open(susc_tif, "w", **profile) as dst:
            dst.write(mean_susc.astype("float32"), 1)
        print(f"Susceptibility raster → {susc_tif}")

        sd_tif = data_dir / "rudraprayag_susceptibility_sd.tif"
        with rasterio.open(sd_tif, "w", **profile) as dst:
            dst.write(sd_susc.astype("float32"), 1)
        print(f"Disagreement raster → {sd_tif}")

        print("\n=== Generating susceptibility map ===")
        _plot_susceptibility(mean_susc, tif_path, out_dir / "fig5_susceptibility_map.png")
    else:
        print(
            f"\nFeature raster not found at {tif_path}. Skipping domain scoring.\n"
            "To generate the susceptibility map, run step 1 (features.py) first."
        )

    print("\n=== Model run complete ===")


if __name__ == "__main__":
    run_model()

