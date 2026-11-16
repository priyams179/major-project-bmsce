"""
Step 2 — Build landslide inventory and sample stable (no-landslide) points.

Reads:
  data/landslide_review_rudraprayag (4).geojson  — 299 manually reviewed candidates
  data/rudraprayag_features.tif                  — feature raster from step 1

Writes:
  data/training_points_matched_wgs84.geojson     — 76 matched training points (38 + 38)
  outputs/fig3_landslide_inventory.png           — inventory map (Figure 3)

The original repo used Google Drive; this module reads from local data/.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "outputs"


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_review(data_dir: Path = DATA_DIR) -> gpd.GeoDataFrame:
    """Load the most recent review GeoJSON from data/."""
    patterns = [
        str(data_dir / "landslide_review_rudraprayag*.geojson"),
        str(data_dir / "landslide_review_rudraprayag *.geojson"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(
            f"No landslide_review_rudraprayag*.geojson in {data_dir}.\n"
            "Copy it from uttarakhand_risk_mapping/data/ or run the GEE review script."
        )
    latest = max(files, key=os.path.getmtime)
    print(f"Review file: {Path(latest).name}")
    return gpd.read_file(latest)


def _deduplicate_landslides(review: gpd.GeoDataFrame, min_dist_m: float = 200) -> gpd.GeoDataFrame:
    """Keep one 'yes' point per cluster (deduplicate within min_dist_m)."""
    review_utm = review.to_crs("EPSG:32644")
    yes = review_utm[review_utm["decision"] == "yes"].reset_index(drop=True)
    kept: list[int] = []
    for i, geom in enumerate(yes.geometry):
        if all(geom.distance(yes.geometry[k]) > min_dist_m for k in kept):
            kept.append(i)
    deduped = yes.iloc[kept].reset_index(drop=True)
    print(f"Decisions — yes: {(review['decision']=='yes').sum()}, "
          f"no: {(review['decision']=='no').sum()}, "
          f"unsure: {(review['decision']=='unsure').sum()}")
    print(f"After deduplication (>{min_dist_m} m): {len(deduped)} landslides")
    return deduped


def _sample_stable_points(
    tif_path: Path,
    landslides_utm: gpd.GeoDataFrame,
    n: int = 38,
    seed: int = 42,
    max_elev: float = 3500,
    min_dist_m: float = 500,
    match_radius_m: float = 3000,
) -> gpd.GeoDataFrame:
    """Sample n stable points matched within match_radius_m of a landslide."""
    rng = np.random.default_rng(seed)
    band_names = ["elevation", "slope", "aspect", "ndvi", "landcover", "dist_river"]

    with rasterio.open(tif_path) as src:
        elev = src.read(1)
        lc = src.read(5)
        tr = src.transform

    # valid pixel mask
    valid = (elev > 0) & (elev < max_elev) & (lc != 70) & (lc != 80)
    rows, cols = np.where(valid)
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e

    ls_xy = [(p.x, p.y) for p in landslides_utm.geometry]

    # filter out pixels within min_dist_m of any landslide
    d_any = np.full(len(xs), np.inf)
    for lx, ly in ls_xy:
        d_any = np.minimum(d_any, np.hypot(xs - lx, ys - ly))
    pool_xs = xs[d_any > min_dist_m]
    pool_ys = ys[d_any > min_dist_m]

    # matched sampling: one stable point per landslide within match_radius_m
    used = np.zeros(len(pool_xs), dtype=bool)
    chosen: list[int] = []
    for lx, ly in ls_xy:
        d = np.hypot(pool_xs - lx, pool_ys - ly)
        for radius in [match_radius_m, 5000, 10_000]:
            cand = np.where(~used & (d <= radius))[0]
            if len(cand):
                break
        if not len(cand):
            print(f"  Warning: no stable candidate found for landslide at ({lx:.0f}, {ly:.0f})")
            continue
        j = int(rng.choice(cand))
        used[j] = True
        chosen.append(j)

    with rasterio.open(tif_path) as src:
        vals = np.array(list(src.sample(list(zip(pool_xs[chosen], pool_ys[chosen])))))

    stable = gpd.GeoDataFrame(
        vals, columns=band_names,
        geometry=gpd.points_from_xy(pool_xs[chosen], pool_ys[chosen]),
        crs="EPSG:32644",
    )
    stable["label"] = 0
    print(f"Stable (no-landslide) points sampled: {len(stable)}")
    return stable


# ── main ────────────────────────────────────────────────────────────────────

def build_inventory(
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
    tif_name: str = "rudraprayag_features.tif",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = data_dir / tif_name

    # 1. load review and deduplicate
    review = _load_review(data_dir)
    landslides_utm = _deduplicate_landslides(review)

    # 2. add feature values for landslide points
    band_names = ["elevation", "slope", "aspect", "ndvi", "landcover", "dist_river"]
    if tif_path.exists():
        with rasterio.open(tif_path) as src:
            ls_utm = landslides_utm.to_crs(src.crs)
            coords = [(p.x, p.y) for p in ls_utm.geometry]
            vals = np.array(list(src.sample(coords)))
        landslides_feat = ls_utm.copy()
        for j, b in enumerate(band_names):
            landslides_feat[b] = vals[:, j]
        landslides_feat["label"] = 1

        # 3. sample stable points
        stable_feat = _sample_stable_points(tif_path, ls_utm)

        # 4. combine and save
        combined = gpd.GeoDataFrame(
            pd.concat([
                landslides_feat[band_names + ["label", "geometry"]],
                stable_feat[band_names + ["label", "geometry"]],
            ], ignore_index=True),
            crs="EPSG:32644",
        ).to_crs("EPSG:4326")  # WGS-84 for GeoJSON

        out_geojson = data_dir / "training_points_matched_wgs84.geojson"
        combined.to_file(out_geojson, driver="GeoJSON")
        print(f"Saved {len(combined)} training points → {out_geojson}")
    else:
        # Feature raster not yet available; use the pre-built training data
        print(
            f"Feature raster not found at {tif_path}.\n"
            "Using pre-built training_points_matched_wgs84.geojson from data/."
        )
        combined = gpd.read_file(data_dir / "training_points_matched_wgs84.geojson")
        landslides_utm = combined[combined["label"] == 1].to_crs("EPSG:32644")

    # 5. Figure 3 — inventory map
    _plot_inventory(landslides_utm, review.to_crs("EPSG:32644"),
                    tif_path if tif_path.exists() else None,
                    out_dir / "fig3_landslide_inventory.png")


def _plot_inventory(
    landslides_utm: gpd.GeoDataFrame,
    review_utm: gpd.GeoDataFrame,
    tif_path: Path | None,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 9))

    if tif_path and tif_path.exists():
        from rasterio.plot import plotting_extent
        with rasterio.open(tif_path) as src:
            elev = src.read(1, out_shape=(1, src.height // 4, src.width // 4)).squeeze().astype(float)
            extent = plotting_extent(src)
        elev[elev <= 0] = np.nan
        ax.imshow(elev, extent=extent, cmap="Greys_r", origin="upper", alpha=0.6,
                  vmin=np.nanpercentile(elev, 2), vmax=np.nanpercentile(elev, 98))

    no_ls = review_utm[review_utm["decision"] == "no"]
    ax.scatter(
        [p.x for p in no_ls.geometry], [p.y for p in no_ls.geometry],
        s=12, c="silver", marker=".", alpha=0.5, label="Reviewed — not landslide",
    )
    ax.scatter(
        [p.x for p in landslides_utm.geometry], [p.y for p in landslides_utm.geometry],
        s=60, c="red", marker="^", edgecolors="darkred", linewidths=0.5,
        zorder=5, label=f"Landslides (n={len(landslides_utm)})",
    )
    ax.set_title("Landslide inventory — Rudraprayag district", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("Easting (m, UTM 44N)")
    ax.set_ylabel("Northing (m, UTM 44N)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 3 saved → {out_path}")


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd  # noqa: F401 (needed for concat inside build_inventory)
    build_inventory()

