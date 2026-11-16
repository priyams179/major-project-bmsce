# Uttarakhand Landslide Risk Mapping

> **Sub-project within `landslide-management-landsat-col-2`**  
> Source: adapted from [makkergauri/uttarakhand-landslide-risk](https://github.com/makkergauri/uttarakhand-landslide-risk) (Colab notebooks → standalone Python pipeline)

---

## Overview

This sub-project maps **landslide susceptibility in Rudraprayag district, Uttarakhand** using free satellite data and a Random Forest model with rigorous spatial cross-validation. It is entirely independent of the main Siamese U-Net pipeline and lives in its own directory (`uttarakhand_risk_mapping/`).

**Study area:** Rudraprayag is one of Uttarakhand's 13 districts, located in the Garhwal Himalaya. It sits on the pilgimage route to Kedarnath and rises from ~700 m (southern valleys) to ~7,000 m (north). The district is drained by the Mandakini and Alaknanda rivers.

---

## Directory Structure

```
uttarakhand_risk_mapping/
├── src/
│   ├── __init__.py            # Package docstring
│   ├── features.py            # Step 1 — GEE feature raster export
│   ├── inventory.py           # Step 2 — inventory deduplication & stable-point sampling
│   ├── model.py               # Step 3 — Random Forest + spatial CV (20-draw)
│   ├── rainfall.py            # Step 4 — GPM IMERG monsoon rainfall analysis
│   └── interactive_map.py     # Step 5 — Folium interactive HTML map
├── pipeline.py                # CLI that runs all steps end-to-end
├── data/
│   ├── landslide_review_rudraprayag (4).geojson  — 299 manually reviewed candidates
│   ├── training_points_wgs84.geojson             — 76 points, naive sampling
│   ├── training_points_matched_wgs84.geojson     — 76 points, matched sampling ✓ used
│   ├── auc_across_draws.csv                       — 20-draw AUC results [generated]
│   ├── rainfall_monthly_rudraprayag.csv           — monthly IMERG rainfall 1998–2025
│   └── rainfall_monsoon_totals.csv                — JJAS totals by year
├── outputs/                   # All generated figures and HTML map
│   ├── fig3_landslide_inventory.png
│   ├── fig4_model_evaluation.png
│   ├── fig6_rainfall.png
│   └── interactive_map.html
├── figures/                   # Original figures from the upstream repo
│   ├── fig1_study_area.png
│   ├── fig2_conditioning_factors.png
│   ├── fig3_landslide_inventory.png
│   ├── fig4_model_evaluation.jpg
│   ├── fig5_susceptibility_map.png
│   └── fig6_rainfall.png
├── docs/index.html            # Interactive map (original — GitHub Pages)
├── 01_build_features.ipynb    # Original Colab notebook (upstream)
├── 02_landslide_inventory.ipynb
├── 03_model.ipynb
├── 04_rainfall.ipynb
├── 05_interactive_map.ipynb
├── 06_study_area.ipynb
├── requirements.txt           # Upstream requirements (Colab)
├── notes.md                   # Detailed working log from upstream author
└── README.md                  # This file
```

---

## Methodology

### Step 1 — Conditioning Factors (feature raster)

A 30-m resolution 6-band GeoTIFF covering Rudraprayag district is built via Google Earth Engine:

| Band | Factor | Source |
|------|--------|--------|
| 1 | Elevation (m) | NASA SRTM (USGS/SRTMGL1_003) |
| 2 | Slope (°) | Derived from SRTM |
| 3 | Aspect (°) | Derived from SRTM |
| 4 | NDVI | Sentinel-2 SR median, Oct 2024–Mar 2025, <20% cloud |
| 5 | Land cover | ESA WorldCover v200 (10 m, resampled) |
| 6 | Distance to river (m) | WWF HydroSHEDS free-flowing rivers |

> **Note:** The raster requires a GEE account (non-commercial). Run `python pipeline.py --step 1 --gee-project <your_id>` to export it. Without it, steps 2–5 use the pre-built training point GeoJSONs.

### Step 2 — Landslide Inventory

No existing inventory (GSI Bhukosh, NASA COOLR) was reachable for Rudraprayag, so the original author built one semi-automatically:

1. **Candidates:** Sentinel-2 (Oct–Dec 2025) pixels with NDVI < 0.2 AND slope > 25° AND not snow/ice/water AND elevation < 3,500 m → 330 candidate patches of 0.2–20 ha.
2. **Manual review:** 300 candidates reviewed on high-resolution satellite imagery → 42 yes, 190 no, 68 unsure.
3. **Deduplication:** points within 200 m merged → **38 landslides**.
4. **Stable points:** 38 random points, matched within 3 km of a landslide, >500 m from any landslide, same terrain constraints.

The `inventory.py` module re-runs the deduplication and stable-point sampling locally.

### Step 3 — Random Forest Model

**Features used (terrain only, to avoid circularity):**
- Elevation, slope, aspect (sin/cos encoded), distance to river

**Key design decisions:**
| Decision | Rationale |
|----------|-----------|
| Terrain only (no NDVI/land cover) | NDVI < 0.2 is used in the candidate filter — including it would be circular |
| Matched stable-point sampling | Avoids model learning only *where* landslides cluster geographically |
| Spatial cross-validation (Leave-One-Block-Out) | Prevents overfitting to spatial autocorrelation; 5 k-means blocks |
| 20 draws of stable points | With only 38 stable points, one draw can swing AUC by ±0.15 |
| Final map = average of 20 models | Reduces dependence on any single draw |

**Reported performance (from upstream paper, with the full feature raster):**

| Design | Mean AUC | 95% range |
|--------|----------|-----------|
| Original matched | 0.744 | 0.515–0.850 |
| Slope-matched | 0.694 | 0.467–0.862 |
| Coordinates only | 0.585 | 0.485–0.685 |

> Terrain beats coordinates in 90% of draws (+0.16 average gap) — the most robust finding.

**Held-out check (blocked CV, 20 models per fold):**
- 87% of held-out landslides in High/Very High class (95% range: 73–94%)
- 49% of stable points in High/Very High class
- Chance level: 40%

### Step 4 — Rainfall Analysis

Monthly district-average rainfall from **NASA GPM IMERG V07** (1998–2025, ~11 km pixels).

| Statistic | Value |
|-----------|-------|
| Average monsoon (JJAS) total | 1,336 mm |
| Wettest year | 2010 (1,924 mm, +44%) |
| Driest year | 2009 (813 mm, −39%) |
| Monsoon share of annual | ~74% |

The rainfall step generates Figure 6: seasonality bar chart, annual monsoon totals by year, and summary statistics.

### Step 5 — Interactive Map

A Folium HTML map with:
- **Base layers:** OpenStreetMap, Esri World Imagery, OpenTopoMap
- **Layers:** Landslide inventory (red triangles), stable points (blue dots), all 299 review candidates, susceptibility raster overlay (when feature raster + model run are available)
- **Popups:** Click any marker for coordinates and decision label

---

## Quick Start

### Prerequisites

All required packages are already in the project venv:

```bash
# From the landslide-management-landsat-col-2/ root:
uv sync   # installs folium, matplotlib-scalebar, etc.
```

### Run steps 2–5 (no GEE needed)

```bash
cd /path/to/landslide-management-landsat-col-2

# Run the full pipeline (steps 2-5) using pre-built training data:
.venv/bin/python uttarakhand_risk_mapping/pipeline.py

# Or run individual steps:
.venv/bin/python uttarakhand_risk_mapping/pipeline.py --step 2   # inventory
.venv/bin/python uttarakhand_risk_mapping/pipeline.py --step 3   # model
.venv/bin/python uttarakhand_risk_mapping/pipeline.py --step 4   # rainfall
.venv/bin/python uttarakhand_risk_mapping/pipeline.py --step 5   # interactive map
```

Outputs are written to `uttarakhand_risk_mapping/outputs/`.

### Full run with GEE (to get susceptibility map)

```bash
# Step 1: Export feature raster from GEE (~10–20 min background task)
.venv/bin/python uttarakhand_risk_mapping/pipeline.py \
    --step 1 --gee-project <your_ee_project_id>

# Download rudraprayag_features.tif from Google Drive → data/

# Then run steps 2–5 (scoring the full domain will now work):
.venv/bin/python uttarakhand_risk_mapping/pipeline.py \
    --gee-project <your_ee_project_id>
```

### CLI options

```
python pipeline.py [--step STEP] [--gee-project ID] [--data-dir DIR] [--out-dir DIR]

  --step          1 | 2 | 3 | 4 | 5 | all   (default: all = steps 2-5)
  --gee-project   GEE project ID (required for step 1 and rainfall download)
  --data-dir      path to data/ (default: uttarakhand_risk_mapping/data/)
  --out-dir       path for outputs (default: uttarakhand_risk_mapping/outputs/)
```

---

## Data Sources

| Factor | Dataset | Citation |
|--------|---------|----------|
| Elevation, slope, aspect | NASA SRTM 30 m | Farr et al. (2007), *Rev. Geophysics* 45 |
| NDVI | Sentinel-2 SR (Copernicus) | ESA Copernicus |
| Land cover | ESA WorldCover v200 10 m | Zanaga et al. (2022), doi:10.5281/zenodo.7254221 |
| Rivers | WWF HydroSHEDS | Grill et al. (2019), *Nature* 569, 215–221 |
| District boundary | FAO GAUL 2025 | FAO (2025), CC-BY-4.0 |
| Landslide inventory | Upstream author's own | Semi-auto, Sentinel-2, manually reviewed |
| Rainfall | NASA GPM IMERG V07 monthly | Huffman et al. (2019), doi:10.5067/GPM/IMERG/3B-MONTH/07 |

---

## Relationship to Main Project

This sub-project is **completely independent** of the main Siamese U-Net pipeline:

| Aspect | Main pipeline | This sub-project |
|--------|--------------|------------------|
| Sensor | Landsat 8/9 (30 m) | Sentinel-2 (10 m) + SRTM |
| Region | 40 global events | Rudraprayag district only |
| Model | Siamese U-Net (deep learning) | Random Forest (interpretable) |
| Task | Bi-temporal change detection | Susceptibility mapping (static) |
| Labels | Pseudo-labels (multi-index) | Manual inventory (38 points) |
| Accuracy metric | F1, IoU vs pseudo/L4S | AUC (spatial CV, 20 draws) |

Both share the same Python virtual environment (`.venv/`) and GEE credentials but write to separate `data/`, `checkpoints/`, and `results/` directories.

---

## Known Limitations

1. **Small inventory (38 landslides):** results vary substantially with stable-point sampling; 20-draw analysis quantifies this uncertainty.
2. **Single interpreter:** no field verification; only scars still bare in late 2025 and >0.2 ha detectable.
3. **Spatial clustering:** landslides cluster in the upper Mandakini valley (north); spatial CV partially controls for this.
4. **GEE dependency:** full feature raster and rainfall download require a GEE account; pre-built CSVs and GeoJSONs allow offline use.
5. **No velocity / trigger information:** the map shows *where*, not *when* — combine with rainfall (step 4) for temporal risk.

---

## References

- **Ghorbanzadeh et al. (2022).** Landslide4Sense: Reference benchmark dataset for remote sensing landslide detection. *IEEE TGRS*.
- **Farr et al. (2007).** The Shuttle Radar Topography Mission. *Reviews of Geophysics*, 45.
- **Grill et al. (2019).** Mapping the world's free-flowing rivers. *Nature*, 569, 215–221.
- **Huffman et al. (2019).** GPM IMERG Final Precipitation L3 1 month. GES DISC. doi:10.5067/GPM/IMERG/3B-MONTH/07
- **Zanaga et al. (2022).** ESA WorldCover 10 m 2021 v200. doi:10.5281/zenodo.7254221
- **Makkergauri (2026).** Where do slopes fail in Rudraprayag? EarthArXiv preprint. https://eartharxiv.org/repository/view/15237/
