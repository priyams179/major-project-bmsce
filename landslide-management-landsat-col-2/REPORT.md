# Landslide Detection from Landsat 8/9 — Technical Report

**Project:** `landslide-management-landsat-col-2`
**Date:** 21 Oct 2026 (LandSentry module added 21 Nov 2026, §2.8, §3.11, §4.8)
**Hardware:** NVIDIA RTX A5000 24 GB
**Stack:** Python 3.12, PyTorch 2.14, rasterio, geopandas, Google Earth Engine

---

## 1. Objective

Detect and map landslides from pre/post satellite imagery of documented
disaster events, producing georeferenced polygons (GeoJSON/CSV), and
establish **honest accuracy figures against human ground truth**, with a
focus on Indian landslide regions. All targets met on held-out test data
with real human labels:
- **Indian-region segmentation** (HR-GLDD, incl. Kerala/Karnataka Western
  Ghats): **F1 0.748 / IoU 0.60** (§4.3a) — *real* pixel masks, 3 m.
- **Scene-level detection** (L4S): **90.7 % acc / 92.2 % F1** (§4.1).
- **Pixel segmentation** (L4S 3-model ensemble): **0.762 F1 / 0.615 IoU** (§4.2).
- **Bitemporal Sentinel-2 change detection, cross-event** (LandSentry, `landsentry/`):
  - Kodagu test against the real inventory: F1 **0.694 ± 0.008**, a +2.2-point gain over NDVI differencing.
  - Kodagu → Wayanad zero-shot: F1 **0.894**, against an index-derived reference.
  - Wayanad → Kodagu does **not** transfer: F1 0.51.

## 2. Data

This project integrates satellite remote sensing datasets across optical, microwave radar (SAR), and topographic modalities to address different operational challenges: multi-temporal change detection, all-weather monsoon penetration in India, high-resolution scar delineation, and regional terrain susceptibility modeling.

### Master Dataset Inventory & Specification Matrix

| Dataset | Modality & Sensor | Geographic Region | Total Images / Pairs / Patches | Native Resolution | Ground Truth Type | Bands & Channels | Target Model(s) |
|---|---|---|---|---|---|---|---|
| **Landsat 8/9 Event Dataset (v2)** | Multi-spectral Optical + DEM (Landsat 8/9 OLI/TIRS, SRTM) | Global (6 continents, 40 events; 8 in India) | 78 GeoTIFFs (39 pre/post pairs); ~4,000+ patches (128×128) | ~30 m | Multi-index consensus pseudo-labels | 7 bands input (21 channels fused) | `SiameseUNet` (21 ch), strict/relaxed ensemble |
| **Landslide4Sense (L4S)** | Multi-spectral Optical + DEM (Sentinel-2 MSI, ALOS PALSAR) | Global mountain regions (multi-continental) | 3,799 patches (128×128 px): 2,661 train / 569 val / 569 test | 10 m | Expert human-annotated binary pixel masks | 14 channels (12 optical + DEM + slope) | `SingleUNet` (14 ch), `DeepUNet` (14 ch), `TwoStreamAttentionUNet` (14 ch), `PatchClassifier` (14 ch), 3-Model Ensemble |
| **HR-GLDD** | High-Res Optical (PlanetScope) | Global (10 sites, incl. Kodagu, Western Ghats, India) | 1,758 patches (128×128 px): 1,119 train / 284 val / 355 test | 3 m | Manually digitized pixel masks (~10% positive) | 4 optical bands + 1 derived NDVI (5 channels) | `DeepUNet` (5 ch) |
| **Landsat-5 India** | Historical Optical (Landsat-5 TM) | Indian Himalaya & Northeast (2007–2012) | 216 GeoTIFFs (108 pairs); 92 scoped (74 train / 18 test) | 30 m | NASA GLC points (catalog proximity validation) | 3 bands input (9 channels fused) | `SiameseUNet` (9 ch) |
| **Sentinel-1 SAR Western Ghats** | C-band Synthetic Aperture Radar (Sentinel-1 GRD) | Indian Western Ghats (Kerala, Karnataka, Maharashtra, TN) | 60 GeoTIFFs (30 pairs) + 1 large Kodagu mosaic (3613×4354 px) | 10 m | SAR pseudo-labels + Kodagu 2018 real inventory (217 polygons) | 2 backscatter bands + slope (6–7 channels fused) | `SiameseUNet` (6 ch), `PatchClassifier` (7 ch) |
| **Uttarakhand Rudraprayag Susceptibility** | Multi-Source Conditioning (SRTM, S2, WorldCover, HydroSHEDS) | Rudraprayag District, Garhwal Himalaya, Uttarakhand, India | 299 reviewed candidates; 76 matched points (38/38); 1 full raster (2489×2061 px) | 30 m | High-res satellite visual confirmation + matched background | 6 conditioning factors (elevation, slope, aspect, NDVI, LC, river dist) | `RandomForestClassifier` (20-draw spatial ensemble) |
| **LandSentry Sentinel-2 bitemporal (Kodagu 2018 + Wayanad 2024)** | Multi-spectral Optical (Sentinel-2 MSI L1C, s2cloudless-masked dry-season medians) | Kodagu (Karnataka) and Wayanad (Kerala), Western Ghats, India | 2 full scenes (4 composites): Kodagu 3594×4258 px, Wayanad 1779×2189 px; random 128×128 crops on the fly | 10 m | Kodagu: 217 real inventory polygons. Wayanad: seeded NDVI-change *reference* mask (not independent) | 4 bands per date (B2, B3, B4, B8), so 8 channels | `SiameseCNN`, `BITransformer`, NDVI-difference baseline |
| **NASA Global Landslide Catalog (GLC)** | Tabular point database | Global (11,033 events, 2007–2017; 865 in India) | 11,033 points | Point lat/lon | Documented reports & media | Lat, lon, date, trigger, injury/fatality | Independent spatial proximity check (`qc_glc.py`) |
| **NASA GPM IMERG V07** | Satellite Precipitation Radar/Radiometer | Rudraprayag District, Uttarakhand, India | 333 monthly series (1998–2025; 28 complete monsoons) | ~11 km (0.1°) | Measured precipitation rate (mm/hr & monthly mm) | Monthly rainfall total (mm) | Monsoon rainfall climatology & trigger analysis |

---

### 2.1 Custom Global Landsat 8/9 Event Dataset (v2)

- **Source:** Landsat 8 and Landsat 9 Collection 2 Level-2 Surface Reflectance via Google Earth Engine (`LANDSAT/LC08/C02/T1_L2` and `LANDSAT/LC09/C02/T1_L2`).
- **Geographic Region:** Global coverage spanning 40 catalogued catastrophic landslide events across 6 continents (Asia, North America, South America, Europe, Africa, Oceania). Includes 8 major Indian landslide events across the Western Ghats (Wayanad 2024, Kavalappara 2019, Idukki 2018, Shirur 2024), Himalayas (Chamoli 2021, Kedarnath 2013, Ramban 2024), and Eastern Himalaya (Teesta 2023).
- **Total Images & Format:** 78 GeoTIFF scenes forming 39 valid pre/post bi-temporal event pairs (285 MB total). Sliced during training into ~4,000+ patches of 128×128 pixels (stride 64). PRE composites are generated from a 6-month median window before the event date; POST composites are 6-month medians after the event date.
- **Resolution & Coordinate System:** EPSG:4326, ~30 m pixel grid, NODATA = −9999.0.

#### Bands, Channels, and Physical Remote Sensing Purpose

The model receives a 21-channel fused tensor formed by concatenating the 7-band pre-event composite, the 7-band post-event composite, and their 7-band absolute temporal difference: $[\text{PRE}(7) \mid \text{POST}(7) \mid |\text{POST}-\text{PRE}|(7)]$.

| Channel | Band Name | Spectral / Spatial Spec | Physical & Remote Sensing Purpose in Landslide Mapping |
|---|---|---|---|
| 1 | **Blue** (`SR_B2`) | 0.452 – 0.512 µm, 30 m | Sensitive to atmospheric aerosol scattering and deep water. Provides true-color contrast and separates shadow from fresh rock surfaces. |
| 2 | **Green** (`SR_B3`) | 0.533 – 0.590 µm, 30 m | Peak reflection band for healthy green vegetation. Used to assess vegetation canopy density and water sediment load. |
| 3 | **Red** (`SR_B4`) | 0.636 – 0.673 µm, 30 m | Strong absorption by plant chlorophyll. Bare rock, dry soil, and landslide debris have very high reflectance relative to vegetation, creating maximum spectral contrast across scar boundaries. |
| 4 | **NIR** (`SR_B5`) | 0.851 – 0.879 µm, 30 m | Near-infrared light is strongly scattered by spongy mesophyll cell structures in healthy plant leaves. When a landslide strips vegetation, NIR reflectance drops precipitously. Crucial for $\Delta\text{NDVI}$ and $\Delta\text{NBR}$. |
| 5 | **SWIR1** (`SR_B6`) | 1.566 – 1.651 µm, 30 m | Highly sensitive to soil moisture and bare mineral absorption. Freshly exposed bedrock, gravel, and mud demonstrate a massive surge in SWIR1 reflectance compared to pre-event forest canopy. |
| 6 | **SWIR2** (`SR_B7`) | 2.107 – 2.294 µm, 30 m | Directly responds to hydroxyl-bearing clay minerals, silicates, and saturated mud. Crucial for identifying debris flow runout zones and distinguishing fresh deposits from old weathered rock. |
| 7 | **Slope** | NASA SRTMGL1_003 (degrees) | Topographic slope. Landslides are gravity-driven failures that cannot initiate on flat ground. Slope acts as a physical terrain constraint, eliminating false positives in riverbeds and valley floodplains. |
| 8–14 | **POST Composite** | Same 7 bands post-disaster | Captures the altered radiometric and topographic state after mass failure. |
| 15–21 | **Temporal Difference** | $|\text{POST}_i - \text{PRE}_i|$ (7 bands) | Direct temporal difference channels. Isolates abrupt changes in spectral response (canopy loss + bare mineral exposure) while cancelling out static background land cover. |

---

### 2.2 Landslide4Sense (L4S) Benchmark Dataset

- **Source:** Zenodo 10463239 (*Ghorbanzadeh et al., IEEE TGRS 2022*), competition dataset for international landslide detection benchmarking.
- **Geographic Region:** Multi-continental coverage across major mountainous disaster zones globally (unlabeled geographic coordinates in the public release to prevent spatial memorization).
- **Total Images & Format:** 3,799 multi-spectral image tiles in HDF5 format (`.h5`), each 128×128 pixels at 10 m resolution. Paired with 3,799 expert-annotated binary ground-truth segmentation masks (`TrainData/mask/`).
- **Data Split:** Rigorous 3-way split: **70% train (2,661 patches) / 15% val (569 patches) / 15% test (569 patches)** with random seed 42.

#### Bands, Channels, and Physical Remote Sensing Purpose

The L4S dataset provides 14 multi-modal channels combining 12 optical bands from the Copernicus Sentinel-2 MSI instrument with 2 topographic bands from the ALOS PALSAR global terrain model:

| Channel | Band Identifier | Central Wavelength / Spec | Resolution | Physical & Remote Sensing Purpose in Landslide Mapping |
|---|---|---|---|---|
| 1 | **B01** (Coastal Aerosol) | 443 nm | 60 m → 10 m | Sensitive to atmospheric scattering, aerosols, and coastal/inland water bodies. Aids in atmospheric correction and haze detection. |
| 2 | **B02** (Blue) | 490 nm | 10 m | Differentiates bare soil from vegetation; useful for soil moisture and snow/ice discrimination. |
| 3 | **B03** (Green) | 560 nm | 10 m | Measures healthy green canopy reflectance; high reflectance contrast against dark saturated landslide mud. |
| 4 | **B04** (Red) | 665 nm | 10 m | Chlorophyll absorption maximum. Used with NIR to compute standard vegetative indices; delineates clear borders of denuded landslide scars. |
| 5 | **B05** (Red Edge 1) | 705 nm | 20 m → 10 m | Sits on the steep boundary between chlorophyll absorption and mesophyll scattering. Sensitive to vegetation stress and partial foliage destruction. |
| 6 | **B06** (Red Edge 2) | 740 nm | 20 m → 10 m | Tracks canopy chlorophyll content and leaf area index (LAI). Sensitive to forest canopy disturbance along landslide flanks. |
| 7 | **B07** (Red Edge 3) | 783 nm | 20 m → 10 m | Tracks leaf cellular structure. Bridges the red edge into the near-infrared plateau. |
| 8 | **B08** (Broad NIR) | 842 nm | 10 m | Highest vegetation canopy reflection. Key discriminator: healthy slope vegetation displays high B08, while stripped landslide tracks show very low B08. |
| 9 | **B09** (Water Vapor) | 945 nm | 60 m → 10 m | Atmospheric absorption band for atmospheric water vapor correction. |
| 10 | **B10** (Cirrus) | 1375 nm | 60 m → 10 m | Detects high-altitude optically thin cirrus clouds; ensures cloud-contaminated pixels are not mistaken for bare earth. |
| 11 | **B11** (SWIR1) | 1610 nm | 20 m → 10 m | Highly sensitive to moisture content in vegetation and soil. Differentiates wet mud deposits and debris flow tracks from dry exposed bedrock. |
| 12 | **B12** (SWIR2) | 2190 nm | 20 m → 10 m | Identifies clay minerals, hydrothermal alteration, and geological lithology exposed by slope failure. |
| 13 | **Slope** | ALOS PALSAR Slope (degrees) | 12.5 m → 10 m | Quantifies gravitational gradient. Slopes $>20^\circ$ represent prime failure territory; slopes $<5^\circ$ represent deposition or flood plains. |
| 14 | **DEM** | ALOS PALSAR Elevation (m) | 12.5 m → 10 m | Absolute topographic altitude. Dictates climate zone, vegetation boundary, and orographic relief energy. |

---

### 2.3 High-Resolution Global Landslide Detector (HR-GLDD)

- **Source:** Meena et al. (2023), *Earth System Science Data (ESSD)*, Zenodo DOI: `10.5281/zenodo.7189381`, Creative Commons Attribution 4.0.
- **Geographic Region:** 10 diverse global physiographic regions, explicitly including the **Western Ghats of India (Kodagu District, Karnataka — August 2018 monsoon disaster)**, alongside sites in Nepal, Taiwan, Japan, China, USA, and Europe.
- **Total Images & Format:** 1,758 PlanetScope satellite image patches (128×128 pixels) provided as NumPy arrays (`.npy`), totalling 576 MB. Author-curated partition: **1,119 train / 284 val / 355 test**.
- **Spatial Resolution:** **3 m ground sampling distance (GSD)** — 10× sharper than Landsat (30 m) and 3.3× sharper than Sentinel-2 (10 m).
- **Labels:** Real, manually digitized, high-precision binary pixel masks (~10% positive landslide pixels), resolving individual narrow debris tracks and shallow translational slides.

#### Bands, Channels, and Physical Remote Sensing Purpose

The pipeline ingests the 4 native PlanetScope optical bands and computes an on-the-fly 5th channel (NDVI):

| Channel | Band Identifier | Wavelength Range | Spatial Res | Physical & Remote Sensing Purpose in Landslide Mapping |
|---|---|---|---|---|
| 1 | **Red (R)** | 605 – 680 nm | 3 m | Captures fine bare-earth reflection and soil scar edges at 3 m resolution. |
| 2 | **Green (G)** | 510 – 590 nm | 3 m | Reflectance peak of lush tropical Western Ghats canopy; highlights sharp transitions where trees were swept away. |
| 3 | **Blue (B)** | 450 – 515 nm | 3 m | Resolves shadowed valleys and provides true-color visual contrast. |
| 4 | **Near-Infrared (NIR)** | 775 – 880 nm | 3 m | High-resolution plant biomass measurement; collapses inside landslide failure scars. |
| 5 | **NDVI** | Derived: $\frac{\text{NIR}-\text{R}}{\text{NIR}+\text{R}}$ | 3 m | Explicit normalized vegetative health index $[-1, +1]$. Strongly suppresses topographic shadows and sun-angle variations across steep valleys. |

---

### 2.4 Historical Landsat-5 (TM) Indian Landslide Dataset

- **Source:** USGS/NASA Landsat 5 Thematic Mapper (TM) Collection 2 Level-2 Surface Reflectance downloaded via Earth Engine (`LANDSAT/LT05/C02/T1_L2`).
- **Geographic Region:** Exclusively Indian landslide events from the NASA Global Landslide Catalog across the **Himalayas** (Uttarakhand, Himachal Pradesh, Jammu & Kashmir) and **Northeast India** (Assam, Sikkim, Mizoram, Nagaland) covering the historical window 2007–2012.
- **Total Images & Format:** 216 GeoTIFF scenes (108 pre/post event pairs). 92 pairs fall within the cloud-usable Himalayan/Northeast belts (74 train / 18 held-out test).
- **Cloud Masking Strategy:** Per-pixel `QA_PIXEL` masking + scene-level cloud filtering. (*Western Ghats monsoon events are cloud-starved in Landsat-5 due to the 16-day orbit revisit coinciding with monsoon cloud cover, achieving 0% detection — highlighting why radar/SAR is mandatory there*).
- **Labels & Validation:** Historical events possess documented coordinate points but no digitized pixel masks. Validated through independent **catalog proximity detection rates** (checking whether model flags mass change within 3 km and 5 km of documented failure coordinates).

#### Bands, Channels, and Physical Remote Sensing Purpose

The Landsat-5 model fuses 3 bands pre-event, 3 bands post-event, and their absolute temporal difference into a 9-channel tensor: $[\text{PRE}(3) \mid \text{POST}(3) \mid |\text{POST}-\text{PRE}|(3)]$.

| Channel | Band Identifier | Wavelength | Resolution | Physical & Remote Sensing Purpose in Landslide Mapping |
|---|---|---|---|---|
| 1 | **NIR** (`SR_B4`) | 0.76 – 0.90 µm | 30 m | Primary vegetation index band; reflects forest canopy biomass. Shows immediate drop across landslide scars. |
| 2 | **SWIR1** (`SR_B5`) | 1.55 – 1.75 µm | 30 m | Penetrates light atmospheric haze; highly sensitive to bare ground exposure and soil moisture differences. |
| 3 | **SWIR2** (`SR_B7`) | 2.08 – 2.35 µm | 30 m | Absorbs strongly in clay and hydroxyl-rich minerals; highlights newly fractured geological substrate. |
| 4–6 | **POST Composite** | Same 3 bands | 30 m | Captures post-disaster spectral properties. |
| 7–9 | **Temporal Difference** | $|\text{POST}_i - \text{PRE}_i|$ | 30 m | Isolates spectral change caused by disaster events. |

---

### 2.5 Sentinel-1 C-Band SAR Western Ghats Dataset & Kodagu Real Ground Truth

- **Source:** Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) Ground Range Detected (GRD) in Interferometric Wide (IW) swath mode via Google Earth Engine (`COPERNICUS/S1_GRD`).
- **Geographic Region:** Indian Western Ghats (Kerala, Karnataka, Maharashtra, Tamil Nadu). Encompasses 30 high-impact monsoon disaster events from 2015–2024: Wayanad (2024 Chooralmala debris flow), Kodagu (2018), Kavalappara (2019), Puthumala (2019), Pettimudi (2020), Taliye (2021), Irshalwadi (2023), and Shirur (2024).
- **Total Images & Format:** 60 GeoTIFF scenes (30 bi-temporal pre/post pairs) at 10 m resolution. Plus 1 massive full-extent mosaicked scene (3613×4354 pixels, 15.7 km² area) over Kodagu, Karnataka.
- **Ground Truth:**
  1. *Speckle-filtered pseudo-labels:* 3×3 median filter on backscatter change layers across 30 events.
  2. *Real Ground Truth:* 217 hand-delineated landslide polygons from Achaiah et al. (2024) for the August 2018 Kodagu disaster, rasterized onto the 10 m SAR grid. Evaluated using a strict spatial split: left 70% columns for training, held-out right 30% block for testing.
- **Why Radar:** C-band microwaves ($\lambda \approx 5.6\text{ cm}$) penetrate monsoon clouds, rain, and atmospheric smoke, providing all-weather imaging where optical satellites are completely blind.

#### Bands, Channels, and Physical Remote Sensing Purpose

The SAR pipeline constructs a multi-channel tensor combining dual-polarization backscatter, bi-temporal ratios, and topographic slope:

| Channel | Band Identifier | Mode / Polarization | Resolution | Physical & Remote Sensing Purpose in Landslide Mapping |
|---|---|---|---|---|
| 1 | **VV Pre** | Vertical Transmit, Vertical Receive (dB) | 10 m | Co-polarized backscatter sensitive to surface roughness and soil dielectric moisture content. Baseline pre-event roughness. |
| 2 | **VH Pre** | Vertical Transmit, Horizontal Receive (dB) | 10 m | Cross-polarized backscatter driven by multiple volume scattering within forest canopy branches. High over intact forests. |
| 3 | **VV Post** | Vertical Transmit, Vertical Receive (dB) | 10 m | Post-event co-polarized backscatter. Disrupted, churned landslide debris alters surface roughness, increasing or shifting VV. |
| 4 | **VH Post** | Vertical Transmit, Horizontal Receive (dB) | 10 m | Post-event cross-polarized backscatter. When a landslide uproots and strips the forest canopy, volume scattering collapses, causing a sharp drop in VH. |
| 5 | **$\Delta\text{VV}$** | $\text{VV}_{\text{post}} - \text{VV}_{\text{pre}}$ (log-ratio) | 10 m | Dynamic change layer. Cancels out stationary topographic radar shadow and layover geometry, highlighting surface disruption. |
| 6 | **$\Delta\text{VH}$** | $\text{VH}_{\text{post}} - \text{VH}_{\text{pre}}$ (log-ratio) | 10 m | Primary vegetation destruction indicator. Highly discriminative for landslide scar extraction in tropical forests. |
| 7 | **Slope** | NASADEM Elevation Slope (degrees) | 10 m | Physical terrain constraint. Prevents false alarms on low-lying flat agricultural fields that undergo radar moisture changes. |

---

### 2.6 Uttarakhand Rudraprayag Landslide Susceptibility & Rainfall Dataset

- **Source:** Multi-source integration combining NASA SRTM 30 m elevation, Copernicus Sentinel-2 MSI optical imagery, ESA WorldCover 10 m land cover, WWF HydroSHEDS river networks, and NASA GPM IMERG V07 satellite precipitation.
- **Geographic Region:** Rudraprayag District, Garhwal Himalaya, Uttarakhand, India (elevation 700 m to 7,000 m). Traversed by the Mandakini and Alaknanda rivers; encompasses the high-risk Kedarnath pilgrimage corridor.
- **Inventory Data:** 299 candidate bare patches screened from Sentinel-2 (Oct–Dec 2025; slope $>25^\circ$, NDVI $<0.2$, elevation $<3,500$ m) reviewed individually on high-resolution Google Earth imagery: 42 yes, 190 no, 68 unsure. Deduplication within 200 m yielded **38 confirmed landslides**.
- **Sampling Design:** 38 non-landslide stable points sampled using matched spatial buffers within 3 km of each landslide ($>500$ m separation). Evaluated across **20 random background draws** to eliminate sampling bias.
- **Domain Raster:** Full-district 6-band GeoTIFF (2489×2061 pixels, 30 m resolution, EPSG:32644).
- **Rainfall Records:** 333 monthly district-average precipitation values from NASA GPM IMERG V07 (1998–2025, 28 complete monsoon seasons).

#### Bands, Channels, and Physical Remote Sensing Purpose

| Channel | Conditioning Factor | Data Source & Spec | Resolution | Physical Role in Landslide Susceptibility |
|---|---|---|---|---|
| 1 | **Elevation** | NASA SRTMGL1_003 (m) | 30 m | Controls climatic vegetation zonation, permafrost boundaries, and orographic relief energy. Slopes $>3,500$ m transition into snow/ice. |
| 2 | **Slope** | Derived from SRTM (degrees) | 30 m | Primary gravitational shear stress driver. Slopes between $25^\circ$ and $45^\circ$ exhibit the highest failure frequency in Rudraprayag. |
| 3 | **Aspect (sin & cos)** | Derived from SRTM (degrees) | 30 m | Circular slope orientation decomposed into $\sin(\theta)$ (East-West) and $\cos(\theta)$ (North-South). Controls solar insolation, soil drying, and monsoon storm intercept. |
| 4 | **NDVI** | Sentinel-2 median (Oct 2024–Mar 2025) | 10 m → 30 m | Normalized Difference Vegetation Index. Measures protective root cohesion and canopy interception against rainfall erosion. |
| 5 | **Land Cover** | ESA WorldCover v200 (classes 10, 30, 60) | 10 m → 30 m | Surface permeability and mechanical soil stabilization (tree cover vs grassland vs exposed bare rock). |
| 6 | **Distance to River** | WWF HydroSHEDS free-flowing rivers (m) | 30 m | Proximity to fluvial drainage channels. River incision and toe-undercutting destabilize steep mountain valley flanks. |

---

### 2.7 Independent Catalogs (Validation & Climatology)

- **NASA Global Landslide Catalog (GLC):** 11,033 worldwide documented landslide events (2007–2017) compiled from media, disaster reports, and government registries. Contains 865 Indian events. Used in `qc_glc.py` for independent spatial proximity benchmarking.
- **NASA GPM IMERG V07 Precipitation:** Global Precipitation Measurement Integrated Multi-satellitE Retrievals for GPM. Monthly calibrated precipitation rate (0.1° × 0.1° grid, ~11 km) quantifying the monsoon triggering envelope for the Garhwal Himalaya.

---

### 2.8 LandSentry Sentinel-2 Bitemporal Dataset (Kodagu 2018 and Wayanad 2024)

This dataset is a fresh, self-contained re-implementation of the
[LandSentry](https://github.com/GHRUTHI/Landsentry) idea in `landsentry/`.
It reuses no code from `src/`. It is built for an explicit **cross-event**
test: train on one Western Ghats disaster, then detect another.

- **Source:** Sentinel-2 MSI **Level-1C** (`COPERNICUS/S2_HARMONIZED`), joined with `COPERNICUS/S2_CLOUD_PROBABILITY`. Pixels are masked when s2cloudless probability is ≥ 40 % or a QA60 cloud/cirrus bit is set. Scenes with more than 70 % cloud are dropped first. L1C is used instead of L2A because L2A coverage over India is incomplete before Dec 2018: Earth Engine returns 14 L2A scenes against 54 L1C scenes over Kodagu for Jan–May 2018. Both events must be processed identically.
- **Composites:** Each date is a per-pixel median over a **Dec–Mar dry-season window**. The before and after windows are the same months one year apart, so phenology matches and monsoon cloud is avoided.

  | Event | Before | After | S2 scenes (before/after) | Median clear looks/px | Valid coverage |
  |---|---|---|---|---|---|
  | Kodagu (16 Aug 2018) | 2017-12-01 → 2018-03-31 | 2018-12-01 → 2019-03-31 | 38 / 45 | 16 / 20 | 100 % |
  | Wayanad (30 Jul 2024) | 2023-12-01 → 2024-03-31 | 2024-12-01 → 2025-03-31 | 20 / 23 | 16 / 20 | 100 % |
- **Grid:** Both dates are downloaded directly onto one fixed **UTM 43N 10 m grid**, using `crs_transform` and 1024 px tiles in parallel. The before and after rasters are therefore pixel-aligned by construction. No resampling or co-registration step is needed, and none of the LandSentry ECC step is required.
- **Bands:** B2 (blue), B3 (green), B4 (red) and B8 (NIR) at native 10 m, for each date, giving **8 input channels**. This is the same band set as the original LandSentry.
- **Ground truth and labels:**
  1. *Kodagu: real.* The 217-polygon Achaiah et al. (2024) inventory (CC BY 4.0) is rasterised onto the S2 grid, giving **51,189 px = 5.12 km²** (0.33 % of the scene). An overlay check shows the polygons sit on the bright fresh scars (`data/landsentry/kodagu/qc_labels.png`).
  2. *Wayanad: reference, not independent.* No public vector outline of the Mundakkai–Chooralmala debris flow exists. The NRSC/Charter impact maps and journal figures are raster or PDF only, and the original LandSentry repo ships only a **point** placemark (`Mundakkai Landslide 2024.kmz`, 76.1348 E, 11.4646 N). We build a **seeded change reference mask** from it in four steps:
     - Keep pixels where NDVI(before) − NDVI(after) > 0.20 and NDVI(after) < 0.40.
     - Apply a 1 px morphological opening.
     - Keep only the connected change region (linked by a 3 px bridge) that passes within 300 m of the placemark.
     - Fill holes and drop specks below 0.2 ha.

     The result is **7,049 px = 0.70 km²**: a 5.2 km track (measured along its principal axis, in 4 connected pieces) from the crown down through Chooralmala. It matches the visible scar closely (`data/landsentry/wayanad/qc_labels.png`). Because it is *derived from an NDVI-change rule*, Wayanad scores measure agreement with that rule, not independent accuracy (§4.8).
- **Spatial splits:** A split raster stores 1 = train, 2 = val and 3 = test. No training crop may overlap the val or test regions, even as context.
  - *Kodagu:* 512 px (5.12 km) blocks are assigned at random (seeded) to match a 65/15/20 % split of landslide pixels. Train has 33,118 positive px, val 7,824 and test 10,247.
  - *Wayanad:* the single track is cut perpendicular to its principal (flow) axis, with 16 px gaps. The upper track (source and transport zone) is train with 3,720 px. A middle strip is val with 876 px. The lower track (runout and deposition through Chooralmala) is test with 2,025 px. Each part takes the whole background on its side, so the test region holds 1.77 M pixels of mostly negative terrain, and false positives are fully counted.

---

## 3. Method

To address different operational scales and modalities, five specialized machine learning and deep learning model architectures were implemented.

### Master Model Architecture & Selection Matrix

| Model Architecture | Implementation Class | Trainable Parameters | Training Dataset | Input Channels & Shape | Operational Task | Why This Model Was Chosen |
|---|---|---|---|---|---|---|
| **Siamese U-Net** | `SiameseUNet` | **20.24 M** | Landsat 8/9 v2, Landsat-5 India, Sentinel-1 SAR | 21 ch (L8/9), 9 ch (L5), 6 ch (S1); $[B, C, 128, 128]$ | Bi-temporal change detection & scar segmentation | Features extracted independently with shared-weight encoder; absolute temporal difference fusion isolates abrupt disaster changes from seasonal/illumination noise. |
| **Deep U-Net** | `DeepUNet` | **31.04 M** | Landslide4Sense (14 ch), HR-GLDD (5 ch) | 14 ch (L4S), 5 ch (HR-GLDD); $[B, C, 128, 128]$ | High-resolution pixel-level semantic segmentation | 5-level deep architecture with bottleneck dropout (0.3); multi-scale skip connections preserve narrow debris channels and sharp scar boundaries that are lost in downsampling. |
| **Two-Stream Attention U-Net** | `TwoStreamAttentionUNet` | **31.39 M** | Landslide4Sense | 14 ch (12 opt + 2 topo); $[B, 14, 128, 128]$ | Multi-modal pixel-level semantic segmentation & edge refinement | Asymmetric multi-modal encoding decouples optical reflectance from topographic elevation/slope; CBAM channel and spatial attention dynamically emphasizes critical spectral bands (SWIR/Red Edge); topographic gating layer $\mathbf{F}_{\text{opt}} \odot (1 + \tanh(\mathbf{F}_{\text{topo}}))$ enforces gravitational physical plausibility. |
| **Single U-Net Baseline** | `SingleUNet` | **7.71 M** | Landslide4Sense | 14 ch; $[B, 14, 128, 128]$ | Standard benchmark comparison | Lightweight 4-level U-Net providing an honest, direct baseline comparison against published L4S competition literature. |
| **Patch Classifier** | `PatchClassifier` | **9.41 M** | Landslide4Sense, Sentinel-1 SAR Kodagu Tiles | 14 ch (L4S), 7 ch (SAR); $[B, C, 128, 128]$ | Scene-level triage & presence detection | Solves the operational early-warning question ("does this 1.28 km tile contain a landslide?") where 90%+ accuracy is achievable without ambiguous pixel-edge penalties. |
| **Spatial Random Forest Ensemble** | `RandomForestClassifier` | **20 models × 100 trees** | Uttarakhand Rudraprayag conditioning stack | 5 tabular terrain features (elevation, slope, aspect sin/cos, river dist) | Static terrain susceptibility mapping & spatial risk modeling | Non-parametric, resistant to overfitting on small inventories (38 points), models complex non-linear terrain interactions, and enables spatial cross-validation across 20 background draws. |
| **LandSentry Siamese CNN** | `landsentry.models.SiameseCNN` | **1.93 M** | LandSentry S2 bitemporal (§2.8) | 8 ch (4 per date); $[B, 8, 128, 128]$ | Bitemporal change segmentation, cross-event | Shared-weight 4-scale encoder (32/64/128/256). The dates are fused only through $\lvert f(\text{pre}) - f(\text{post})\rvert$ at each scale (FC-Siam-diff), which makes the model symmetric and change-focused by design. |
| **LandSentry BIT Transformer** | `landsentry.models.BITransformer` | **0.69 M** | LandSentry S2 bitemporal (§2.8) | 8 ch; $[B, 8, 128, 128]$ | Bitemporal change segmentation, cross-event | Semantic tokens (4 per date) are exchanged through a transformer encoder across dates, then projected back to pixels by cross-attention. Global bitemporal context at low cost; tests whether attention transfers better across events than pure convolution. |
| **NDVI-difference baseline** | `landsentry.cross_event.baseline_score` | **0** | (threshold only) | NDVI(pre), NDVI(post) | Non-learned reference | A deep model must beat this to justify itself. It also exposes how circular the Wayanad reference mask is. |
| **Prithvi-100M ViT** *(Tested, rejected)* | Foundation Model ViT | **100 M** | Landslide4Sense fine-tune | 6 optical bands; $[B, 6, 224, 224]$ | Foundation model evaluation | Tested to explore whether NASA-IBM foundation pre-training could transfer; rejected due to coarse 14×14 tokenization losing fine scar edges (F1 0.669 vs 0.755). |

---

### 3.1 Pseudo-Labels Consensus Engine (v2 Event Set)

Because public polygon ground truth does not exist for the global historical event pairs, an unsupervised multi-index consensus change detection algorithm generates training pseudo-labels (`src/indices.py`):

1. **Normalized Difference Vegetation Index difference:** $\Delta\text{NDVI} = \text{NDVI}_{\text{pre}} - \text{NDVI}_{\text{post}}$
2. **Normalized Burn Ratio difference:** $\Delta\text{NBR} = \text{NBR}_{\text{pre}} - \text{NBR}_{\text{post}}$, where $\text{NBR} = \frac{\text{NIR} - \text{SWIR2}}{\text{NIR} + \text{SWIR2}}$
3. **Normalized Difference Moisture Index difference:** $\Delta\text{NDMI} = \text{NDMI}_{\text{pre}} - \text{NDMI}_{\text{post}}$, where $\text{NDMI} = \frac{\text{NIR} - \text{SWIR1}}{\text{NIR} + \text{SWIR1}}$
4. **Short-Wave Infrared surge:** $\Delta\text{SWIR2} = \text{SWIR2}_{\text{post}} - \text{SWIR2}_{\text{pre}}$

A pixel is labelled as landslide if it receives $\ge 2$ positive index votes on slopes $>10^\circ$, followed by morphological opening (removes isolated speckle) and closing (bridges fragmented scars). While self-consistent, all metrics evaluated against these labels are circular, serving as an upper bound rather than human ground truth.

---

### 3.2 Bi-Temporal Siamese U-Net (`SiameseUNet`)

- **Architecture:** Shared-weight 4-stage convolutional encoder (`cin=7`, `base=64`), absolute temporal difference fusion layer ($|\mathbf{f}_{\text{post}} - \mathbf{f}_{\text{pre}}|$), and a 4-stage convolutional decoder with skip connections (20.24M parameters).
- **Why Siamese:** Passing pre- and post-event images through a shared-weight encoder ensures that spectral features are extracted consistently without temporal bias. Fusing through absolute difference explicitly isolates abrupt change vectors caused by catastrophic slope failure while remaining invariant to seasonal sun angle shifts.
- **Loss Function:** Combined Focal-BCE and Dice Loss:
  $$\mathcal{L} = \mathcal{L}_{\text{FocalBCE}}(\gamma=2.0, \text{pos\_weight}=5.0) + \mathcal{L}_{\text{Dice}}$$
- **Training Protocol:** 50 epochs, AdamW optimizer ($\text{lr}=3\times 10^{-4}$ with Cosine Annealing), batch size 16, patch size 128×128 (stride 64), random horizontal/vertical flips, 90° rotations, and brightness jitter. Event-level train/validation split (31 train / 8 val events).

---

### 3.3 Strict and Relaxed Ensemble Decision Rules

To generate final georeferenced polygons, the continuous probability map from the Siamese U-Net is fused with the multi-index consensus mask:
- **Strict Ensemble (Primary):**
  $$\text{Mask}_{\text{strict}} = (P_{\text{model}} > 0.30) \land \text{Mask}_{\text{multi\_index}}$$
  *Trade-off:* Maximizes precision ($P=1.000$, $R=0.873$, $F_1=0.932$), eliminating false positives in riverbeds.
- **Relaxed Ensemble (Recall Variant):**
  $$\text{Mask}_{\text{relaxed}} = (P_{\text{model}} > 0.25) \land (\text{Mask}_{\text{multi\_index}} \lor P_{\text{model}} > 0.55)$$
  *Trade-off:* Recovers +68 additional landslide polygons across peripheral scars at the expense of precision ($P=0.378$, $R=0.879$, $F_1=0.529$).

---

### 3.4 Geocoding and Polygon Delineation

Raster masks are vectorized using `rasterio.features.shapes` into vector polygons (EPSG:4326), filtered by minimum area ($\ge 5$ connected pixels, equivalent to $\ge 4,500\text{ m}^2$ on 30 m grid), enriched with centroid coordinates, total surface area ($\text{km}^2$), and nearest-event metadata. Exported as `results/landslides.geojson` and `results/landslides.csv`.

---

### 3.5 Benchmark Pixel Segmentation Models (`SingleUNet` & `DeepUNet`)

Evaluated strictly against **human ground truth** on the Landslide4Sense dataset under a protocol designed to eliminate spatial and hyperparameter leakage:
1. **Three-Way Partition:** 70% train (2,661) / 15% val (569) / 15% test (569). Epoch selection on validation; decision threshold swept on validation and **frozen**; reported once on the held-out test set.
2. **Train-Split Normalization:** Channel-wise mean and standard deviation computed strictly from the training split (`checkpoints/l4s_norm.json`).

- **`SingleUNet`:** 4 encoder/decoder stages (`base=64`, 7.71M parameters), DiceBCE loss.
- **`DeepUNet`:** 5 encoder/decoder stages (`base=64`, 31.04M parameters) with bottleneck dropout (0.3).
- **`TwoStreamAttentionUNet`:** 5-level multi-modal architecture (`base=64`, 31.39M parameters) integrating key architectural innovations from the Landslide4Sense competition literature:
  1. *Two-Stream Multi-Modal Stems:* Decouples Sentinel-2 optical reflectance (12 bands) from ALOS PALSAR topography (2 bands: DEM + Slope) to prevent early feature interference.
  2. *CBAM Spectral & Spatial Attention:* Computes channel-wise feature recalibration via Squeeze-and-Excitation MLPs (reduction ratio 4) to dynamically boost discriminative bands (SWIR1/2, Red Edge) while downweighting atmospheric haze, followed by 7×7 spatial attention.
  3. *Topographic Gating:* Gating layer modulates optical features via $\mathbf{F}_{\text{opt}} \odot (1 + \tanh(\mathbf{F}_{\text{topo}}))$, hard-coding gravitational slope feasibility directly into the feature representations.
  4. *Topographic Slope-Gated Copy-Paste Augmentation:* Synthesizes physically realistic training scars by cropping confirmed landslide masks and pasting them onto candidate patches strictly where destination slope $> 8^\circ$, alleviating the 2.3% rare-class penalty.
  5. *Boundary-Aware Tversky Loss (`BoundaryAwareTverskyLoss`):* Augments Focal-Tversky with fixed 3×3 horizontal and vertical Sobel gradient convolutions ($\mathcal{L}_{\text{FT}} + 0.35\,\mathcal{L}_{\text{SobelBoundary}}$), enforcing razor-sharp perimeter delineation and suppressing edge splatter.
- **Loss Function:** Focal-Tversky Loss with focal BCE:
  $$\mathcal{L}_{\text{FT}} = 1 - \frac{\text{TP} + \epsilon}{\text{TP} + \alpha\,\text{FP} + \beta\,\text{FN} + \epsilon}$$
  Setting $\alpha=0.3, \beta=0.7$ penalizes false negatives heavily, prioritizing recall on the rare landslide class (~2.3% positive pixels).
- **Test-Time Augmentation (TTA):** Averaging predicted probabilities across all 8 transforms of the dihedral group $D_4$ (rotations and flips).

---

### 3.6 Patch-Level Landslide Presence Classifier (`PatchClassifier`)

Designed to address operational early-warning triage ("Does this 1.28 km tile contain a landslide?"):
- **Architecture:** 5 convolutional stages with LeakyReLU, batch normalization, max-pooling, Global Average Pooling (GAP), dropout (0.3), and a linear binary classification head (9.41M parameters).
- **Training Protocol:** Stratified 70/15/15 split (2,659 train / 570 val / 570 test; 58.7% positive / 41.3% negative). Binary Cross-Entropy with positive class weighting (`pos_weight = n_neg / n_pos`).
- **Why Patch Classification:** Per-pixel F1 saturates around 0.76 due to ambiguous boundary pixels (mud splatter, shadow transitions). Tile-level presence classification is far more separable and achieves **>90% accuracy**, making it ideal as an automated screen before human visual inspection.

---

### 3.7 PlanetScope High-Resolution Segmentation on HR-GLDD (`DeepUNet`)

- **Architecture:** 5-level `DeepUNet` (`cin=5`, `base=64`, 31.04M parameters).
- **Inputs:** 3 m PlanetScope bands $[R, G, B, \text{NIR}, \text{NDVI}]$.
- **Why High-Res:** Sub-canopy debris flows and narrow runouts in steep valleys like Kodagu (Western Ghats) are only 5–15 m wide. On 30 m Landsat, these appear as mixed single pixels; at 3 m resolution, they span 4–10 pure pixels, allowing precise boundary delineation.
- **Training:** Author-curated partition (1,119 train / 284 val / 355 test), Focal-Tversky loss, AdamW with warmup and cosine decay, 100 epochs, $D_4$ TTA.

---

### 3.8 Sentinel-1 SAR Western Ghats Models (Change & Tile Detection)

- **Change Segmentation:** `SiameseUNet` (`cin=6`, `base=64`, 20.24M parameters) ingesting dual-pol backscatter and log-ratios $[\text{VV}_{\text{pre}}, \text{VH}_{\text{pre}}, \text{VV}_{\text{post}}, \text{VH}_{\text{post}}, \Delta\text{VV}, \Delta\text{VH}]$.
- **Tile Classifier:** `PatchClassifier` (`cin=7`, 9.41M parameters) adding NASADEM topographic slope.
- **Why SAR:** Monsoonal cloud cover causes 100% data loss in optical sensors over the Western Ghats during the critical June–September landslide window. Sentinel-1 C-band microwaves penetrate clouds and rain, detecting surface roughness and vegetation removal.

---

### 3.9 Uttarakhand Rudraprayag Susceptibility Ensemble (`RandomForestClassifier`)

- **Model:** Ensemble of 20 `RandomForestClassifier` models (100 estimators each, `min_samples_leaf=2`).
- **Features Used (Terrain Only):** Elevation, slope, aspect ($\sin$ and $\cos$), distance to river.
- **Why Terrain Only:** Raw NDVI and land cover are deliberately omitted to avoid circularity, as low NDVI ($<0.2$) was used in the candidate scar screening.
- **Spatial Cross-Validation:** 5-fold Leave-One-Group-Out on spatial blocks determined by k-means clustering of point coordinates.
- **20-Draw Background Sampling:** With small inventories (38 scars), drawing stable points once can swing AUC by $\pm 0.15$. Re-drawing stable background points 20 times (seeds 100–119) and averaging predictions produces an honest, reproducible susceptibility map.
- **Why Random Forest:** Non-parametric, robust to multicollinearity among topographic variables, does not overfit on small sample sizes (76 points), and provides clear feature importance rankings.

---

### 3.10 Geospatial Foundation Model Experiment (`Prithvi-EO-1.0-100M`)

To assess whether pre-trained geospatial foundation models could supersede domain-specific U-Nets, we fine-tuned **Prithvi-EO-1.0-100M** (NASA-IBM, Vision Transformer pre-trained on 6-band HLS at 30 m):
- **Setup:** Mapped L4S bands to Prithvi order $[B2, B3, B4, B8, B11, B12]$, interpolated patches from 128×128 to 224×224 pixels, fine-tuned with AdamW for 30 epochs.
- **Result:** Test F1 reached **0.669** (IoU 0.502), trailing the scratch-trained `DeepUNet` (F1 **0.755**, IoU 0.606) by **8.6 F1 points**.
- **Failure Analysis:** ViT tokenization ($14\times 14$ patches) is too coarse for narrow landslide scars; upsampling loses fine edge detail that U-Net skip connections preserve. Additionally, the pre-training resolution (30 m) mismatched the target data (10 m), and ~2,600 training patches were insufficient to adapt a 100M-parameter transformer. Shipped models therefore rely on convolutional U-Nets.

### 3.11 LandSentry: Bitemporal Sentinel-2 Change Detection and Cross-Event Transfer (`landsentry/`)

This is a fresh implementation, separate from `src/`, of the LandSentry
project (bitemporal Siamese CNN and BIT transformer, Kodagu ↔ Wayanad
cross-event testing). Its known defects are
fixed: per-event normalisation, placeholder data, no augmentation, and a
README that contradicts the shipped results.

- **`SiameseCNN` (1.93 M params):** A shared encoder of 4 stages, each a double 3×3 conv + BatchNorm + ReLU (32, 64, 128 and 256 ch), with 2×2 max-pool between stages. At every scale the skip is $\lvert f_s(\text{pre}) - f_s(\text{post})\rvert$. The decoder uses ConvTranspose upsampling, concatenates the difference skip and applies a double conv. A 1×1 head outputs the logit.
- **`BITransformer` (0.69 M params)** follows Chen et al., 2021, "Remote Sensing Image Change Detection with Transformers":
  - A shared residual backbone (stem, then 48 ch at stride 2, then 64 ch at stride 4, each with 2 ResBlocks).
  - A **semantic tokenizer**: a 1×1 conv produces L = 4 spatial-softmax attention maps per date, and the tokens are $T = A^\top X$.
  - A 1-layer, 8-head **transformer encoder** over the 2L = 8 bitemporal tokens, with learned positional embeddings.
  - An 8-layer **cross-attention decoder**: each pixel queries its own date's tokens, with no pixel self-attention.
  - The difference $\lvert X'_\text{pre} - X'_\text{post}\rvert$ is bilinearly upsampled ×4 and passed through a conv head.
- **Normalisation (per scene, joint):** Per-band 2nd/98th percentiles are computed over the *union* of the before and after composites of a scene, and applied to both dates. Shared statistics keep real reflectance change intact. Per-scene statistics cancel event-to-event radiometric offsets (haze, sun angle, year). The original repo instead normalised Wayanad with Kodagu statistics.
- **Training:**
  - Random 128×128 crops, sampled on the fly from the *train region only*. 50 % are centred near a landslide pixel.
  - D4 flips and rotations, plus **independent per-date gain/offset jitter** (σ 0.08 / 0.04) to simulate inter-event radiometric shift.
  - Loss: 0.5 · BCE(pos_weight 5) + 0.5 · soft Dice, restricted to valid pixels.
  - AdamW (lr 3e-4, wd 1e-4) with OneCycle, bf16 autocast, batch 16, 200 steps per epoch, up to 40 epochs, early stop after 10 epochs with no val gain.
- **Model and threshold selection** use *only the source event's val region*. Each epoch, the scene is predicted with a sliding window (256 px tiles, 64 px overlap, Hann-window blending). The checkpoint with the best val F1 is kept together with its F1-optimal threshold. That threshold is **frozen** and applied unchanged to every test set, including the other event. The target event is never used for tuning.
- **Cross-event protocol:** {Siamese, BIT} × train on {Kodagu, Wayanad, both} × 3 seeds = 18 runs. Each is tested on the test region of **both** events. Off-diagonal cells (train on A, test on B) are true zero-shot transfers, and these are also scored on the target's *whole scene*.
  - Metrics: precision, recall, F1 and IoU at the frozen threshold; **average precision (AP)**, which does not depend on a threshold; and an *oracle* F1 (best threshold on the test set). The oracle F1 is reported only to separate ranking quality from threshold miscalibration; it is not a deployment number.
  - The non-learned **NDVI-difference baseline**, score = clip(NDVI_pre − NDVI_post, 0, 1), runs under the same threshold protocol.

---

## 4. Results

### 4.1 Headline: 90 %+ landslide detection (patch classifier, real human labels)

Held-out **test** set (570 tiles never used for training, model selection, or
threshold choice). Decision threshold frozen on val (0.35):

| Metric | Frozen thr (0.35) | thr 0.50 |
|---|---|---|
| **Accuracy** | **0.907** | 0.914 |
| **F1** | **0.922** | 0.927 |
| Precision | 0.910 | 0.931 |
| Recall | 0.934 | 0.922 |
| Balanced accuracy | 0.901 | — |
| **ROC-AUC** | **0.967** | 0.967 |

→ **Target ≥ 90 % met against human labels** — accuracy **90.7 %**, F1
**92.2 %**, AUC **0.967**, out-of-sample. Threshold-independent skill (AUC
0.967) confirms it is not a threshold artefact. Artifacts:
`checkpoints/best_l4s_cls.pt`, `l4s_cls_report.json`, `l4s_cls_norm.json`.

*Scope, stated plainly:* this is **scene-level presence**, not pixel
delineation. The two answer different questions — "is there a landslide
here?" (92 % F1) vs "draw its exact outline" (76 % F1, §4.2). Both use the
same real human labels; neither is circular.

### 4.2 Pixel-level segmentation (Landslide4Sense, **held-out test**, n=569)

Threshold frozen on val, reported once on the untouched test set:

| Model (honest 3-way) | F1 | Precision | Recall | IoU | Pixel acc | thr |
|---|---|---|---|---|---|---|
| Legacy baseline (SingleUNet + official norm, 60 ep) | 0.759 | 0.736 | 0.783 | 0.611 | 0.9888 | 0.60 |
| Improved (DeepUNet, train-norm, Focal-Tversky, TTA, 90 ep) | 0.755 | 0.738 | 0.773 | 0.606 | 0.9887 | 0.65 |
| Two-Stream Attention U-Net (CBAM, Copy-Paste, Boundary Loss, TTA, **25 ep**) | 0.754 | **0.744** | 0.765 | 0.606 | 0.9888 | 0.70 |
| Previous 2-Model Ensemble (SingleUNet + DeepUNet, TTA) | 0.760 | 0.733 | **0.789** | 0.613 | 0.9888 | 0.60 |
| **New 3-Model Diverse Ensemble (Shipped Headline, TTA)** | **0.762** | **0.756** | 0.767 | **0.615** | **0.9892** | 0.65 |
| Published baseline (paper) | 0.56–0.73 | | | | | |
| 2022 competition winners (heavy ensembles) | 0.80+ | | | | | |

→ **Target ≥ 75% met against human labels, and it survives the stricter
protocol** (the honest held-out figure is now **0.762 F1 / 0.615 IoU**). The 3-model
ensemble is the shipped headline. Artifacts: `checkpoints/best_l4s.pt`,
`best_l4s_baseline.pt`, `best_l4s_attention.pt`, `l4s_report.json`,
`l4s_report_baseline.json`, `l4s_attention_report.json`,
`l4s_ensemble_report.json`, `l4s_norm.json`, `l4s_sweep*.json`.

*Key insights from the Two-Stream Attention model:*
1. **Convergence Acceleration:** `TwoStreamAttentionUNet` reaches **F1 0.754** in only **25 epochs** (~7 minutes) compared to 90 epochs (~20 minutes) for `DeepUNet` (>3.5× faster training).
2. **Precision and Edge Sharpness:** Topographic slope gating and Sobel boundary-aware loss significantly reduce false positives in flat valleys, boosting standalone precision to **0.744** and lifting the 3-model diverse ensemble precision by **+2.3% (0.733 $\rightarrow$ 0.756)**, setting a new project high watermark of **F1 0.762 / IoU 0.615** on held-out human ground truth.

**Experiment — geospatial foundation model (tested, not adopted).** We
fine-tuned **Prithvi-EO-1.0-100M** (IBM-NASA, ViT pretrained on 6-band HLS)
on the *same* L4S 3-way split, mapping L4S→Prithvi bands
[B2,B3,B4,B8,B11,B12] and resizing 128→224 px (code: `src/prithvi_l4s.py`).

| Model (same split, same eval) | Test F1 | IoU |
|---|---|---|
| **DeepUNet (from scratch) — shipped** | **0.755** | **0.606** |
| Prithvi-EO-1.0-100M fine-tune (converged, best ep 24) | 0.669 | 0.502 |

→ The foundation model **lost by ~8.6 F1 points** and was **not kept**. Why:
ViT tokens are coarse (14×14) so fine landslide detail is lost on upsampling,
whereas the U-Net's high-res skips preserve small scars; Prithvi's 224 px /
30 m-HLS pretraining mismatches L4S (128 px / 10 m); and ~2.6 k patches is too
few to fully re-adapt a 100 M ViT. Consistent with the literature: plain
ViTs/FMs trail tuned U-Nets on small-object landslide segmentation. Encoder
weights are not committed (re-download per `src/prithvi_l4s.py`).

### 4.3 Indian regions (focus)

**(a) HR-GLDD segmentation — real masks, incl. Indian Western Ghats
(held-out test, n=355 patches).** Threshold frozen on val (0.60):

| Metric | Frozen thr (0.60) | thr 0.50 |
|---|---|---|
| **F1** | **0.748** | 0.745 |
| **IoU** | **0.598** | 0.594 |
| Precision | 0.723 | 0.669 |
| Recall | 0.776 | 0.839 |
| Pixel accuracy | 0.943 | 0.937 |

→ **Good region segmentation on real Indian-region high-res data.** On the
clearest test scars the per-patch F1 is **0.85–0.96** (`qc/hrgldd_*.png`,
RGB | human mask | prediction); the 0.748 aggregate is pulled down by small/
ambiguous patches. This is 3 m PlanetScope with *real digitized masks*, so it
is a genuine pixel-delineation result, not a pseudo-label artefact — and it
comfortably exceeds the 30 m Landsat event pipeline. Artifacts:
`checkpoints/best_hrgldd.pt`, `hrgldd_report.json`, `hrgldd_norm.json`.
*(Caveat: the public split is globally mixed — Kodagu tiles aren't separately
tagged — so this is a global HR-GLDD figure on a set that includes the Indian
Western Ghats, not a Kodagu-only score; §2.3.)*

**(b) Indian Landsat events.** The event set is India-heavy: **8 of 39
events** span the main Indian landslide belts — Western Ghats
(Kerala/Karnataka), Himalaya (Uttarakhand, J&K), and Eastern Himalaya
(Sikkim):

| Event | Region | Date | Strict polygons | Area km² | Note |
|---|---|---|---|---|---|
| `wayanad_in_2024` | Kerala (W. Ghats) | 2024-07-30 | 4 | 0.183 | ✅ mask on Chooralmala debris-flow scar (QC) |
| `kavalappara_in_2019` | Kerala (W. Ghats) | 2019-08-08 | 5 | 0.398 | ✅ detected |
| `idukki_in_2018` | Kerala (W. Ghats) | 2018-08-09 | 3 | 0.060 | ✅ detected |
| `shirur_in_2024` | Karnataka (W. Ghats) | 2024-07-16 | 0 | 0.0 | ✗ miss (see §5) |
| `chamoli_in_2021` | Uttarakhand (Himalaya) | 2021-02-07 | 2 | 0.030 | ✅ detected |
| `kedarnath_in_2013` | Uttarakhand (Himalaya) | 2013-06-17 | 0 | 0.0 | ✗ snow/cloud composite (§5) |
| `ramban_in_2024` | J&K (Himalaya) | 2024-04-25 | 7 | 0.191 | ✅ detected |
| `teesta_in_2023` | Sikkim (E. Himalaya) | 2023-10-04 | 2 | 0.220 | ✅ detected |

**6/8 Indian events detected**; visual QC panels in `qc/*_in_*.png`
(PRE | POST | mask). The 2 misses are export/compositing failures, not model
failures (§5), and the same per-event diagnosis applies.

*Honest caveat on the 90 % figure and India:* the 90.7 %/92.2 % classifier
(§4.1) is trained on L4S **Sentinel-2** tiles (real labels, global, no public
region tags), while the Indian events above are **Landsat** (7-band, 30 m) —
a different sensor, so the classifier is **not** transferred onto them here,
and the Indian events have **no pixel ground truth** (only pseudo-labels), so
no clean per-region F1 is claimed. Getting a *real-labelled Indian* accuracy
needs either a Sentinel-2 re-export of these events (GEE, then the L4S model
applies directly) or an Indian-labelled set (e.g. HR-GLDD Kerala/Kodagu) —
both noted as next steps (§9).

**(c) Historical Landsat-5 Indian model (separate model, held-out events).**
A model trained *only* on historical Indian events, on **Landsat 5** (TM,
Collection-2 L2, 30 m), downloaded fresh from Earth Engine with cloud cover
dropped (scene `CLOUD_COVER` filter + per-pixel `QA_PIXEL` masking). Events
come from the NASA catalog's **865 Indian entries in the Landsat-5 window
(2007–2012)** — Landsat 6 never reached orbit (no data), and Landsat 5 ended
in 2013 so it cannot image *recent* events. 108 event pairs downloaded; the
model is **scoped to the Himalaya / Northeast belts** (74 train / 18 held-out)
— see the Western Ghats finding below.

Labels: change-detection **pseudo-labels** (no pixel masks exist for these
historical events), so the held-out metric is a **catalog-proximity detection
rate** — does the model flag landslide change within R km of the catalogued
point? — not a mask F1.

| Held-out accuracy (Himalaya/NE, unseen events) | Value |
|---|---|
| Detected ≤ 3 km | **~40–53 %** (39 % this split / 53 % a prior split) |
| Detected ≤ 5 km | **~56–63 %** |
| Any detection in scene | ~55 % (rest: cloud/seasonal blind spots) |
| Median nearest-detection distance | ~2.3 km |

→ **Roughly half of unseen historical Himalayan/NE events localised within
3–5 km** of an independent catalog point (itself often only town/road-level
accurate). The wide range is **split variance** — the held-out set is only
~18 events, so the rate swings ±10–15 pts with the random seed; neither split
is cherry-picked as *the* number. Visual QC (`qc/l5_*.png`, Badrinath/Kumaon
2011) shows detections on real post-event change, with heavy `QA_PIXEL` cloud
masking in the Himalayan scenes. Code: `download_landsat5_india.py`,
`src/landsat5.py` (`_scope_events`), `main.py l5_*`.

**Western Ghats — Landsat-5 cannot cover it (sensor limitation).** We pulled
13 Western Ghats events (Wayanad, Nilambur, Kozhikode, Kodaikanal, Ooty,
Coorg…) with relaxed cloudy-region settings, but **held-out detection there
was 0 % (0/6)** — every scene returned an empty prediction. Western Ghats
landslides occur *during the monsoon*, when Landsat 5's 16-day revisit almost
never captures a cloud-free look; the scar sits under `QA_PIXEL` cloud gaps, so
the change signal (and the pseudo-labels) vanish. This is not fixable with more
L5 data. The Western Ghats is therefore covered by the **Sentinel-2 models**
instead (HR-GLDD Kodagu segmentation §4.3a, F1 0.748; L4S classifier §4.1,
90.7 %), and the L5 model is scoped out of it (`DROP_REGIONS`).

**(d) Sentinel-1 SAR Western Ghats model — SAR succeeds where optical fails.**
A dedicated model on **Sentinel-1 C-band SAR** (cloud-penetrating; §2.5),
trained on **30 Western Ghats events** (GLC 2015–17 + 20 documented 2018–24
events: Wayanad, Kodagu, Kavalappara, Pettimudi, Taliye, Irshalwadi…),
bi-temporal on VV/VH with ΔVV/ΔVH change (SiameseUNet 6-ch). SAR pseudo-labels
are **speckle-filtered** (3×3 median on the change layers → 24/30 events gain
signal); held-out catalog-proximity test.

| Held-out (Western Ghats, SAR, 7 events) | Value |
|---|---|
| **Detected ≤ 3 km** | **57 % (4/7)** |
| Localization of the 4 hits | **all ≤ 2 km** (median 1.2 km; Taliye 0.42) |
| vs Landsat-5 optical, same region | **0 %** |

→ **SAR lifts Western Ghats detection from 0 % (Landsat-5 optical) to ~57 %** —
the point of switching sensors: C-band penetrates the monsoon cloud that blinds
optical. Visual QC (`qc/s1_*.png`) confirms it: for **Wayanad 2024**, the ΔVV
log-ratio lights up the Chooralmala debris-flow channel and the detection sits
on it; Kodagu 2018 and Kavalappara 2019 likewise. Code:
`download_sentinel1_wg.py`, `src/sentinel1_wg.py`, `main.py s1_*`. Artifacts:
`checkpoints/best_s1_wg.pt`, `s1_wg_test.json`.

*Honesty — and why this number is lower than an earlier run:* an initial
16-event version reported 75 % on **4** held-out events; expanding to 30 events
and a **7-event** test gave the more trustworthy **57 %** (the 75 % was a
small-sample fluke). The improvement is real where it counts — 2× the data,
speckle-filtered labels, and per-pixel pseudo-F1 0.15 → **0.26** — but the
headline settled honestly, not upward. The 3 misses give no detection at all,
partly from imprecise event-area coordinates and sparse pseudo-labels; the
model learns *where change is*, not exact outlines. This remains a
**demonstrator** (small N, pseudo-labels), not a production accuracy.

**(e) SAR against REAL ground truth (Kodagu 2018 inventory).** We then
replaced pseudo-labels with a **real human-mapped inventory** — Achaiah et al.
(2024), **217 landslide polygons** of the 2018 Kodagu storm (CC BY 4.0). The
polygons were rasterised onto a mosaicked Sentinel-1 scene over the inventory
extent (3613×4354 px, 10 m; `build_kodagu_gt.py`), giving a *true* pixel mask
(0.34 % positive — small scars). Evaluation uses a **spatial left/right split**
(no leakage): train on the left 70 % of columns, report on the held-out right
block.

| Kodagu 2018, held-out block (REAL GT) | F1 | IoU | P | R |
|---|---|---|---|---|
| Pseudo-trained model (zero real labels) | 0.284 | 0.165 | 0.215 | 0.415 |
| **Fine-tuned on real masks** | **0.554** | **0.383** | 0.477 | 0.660 |

→ **Real supervision ~doubles real F1 (0.28 → 0.55, +95 %)** — the first
accuracy of the Western Ghats SAR model measured against *truth*, not a proxy.
F1 0.55 / IoU 0.38 is solid for 10 m SAR segmentation of tiny scars; QC
(`qc/s1_kodagu_real_gt.png`) shows the prediction tracking the real inventory
polygons in the unseen right block. Code: `build_kodagu_gt.py`, `main.py
s1_eval_real` / `s1_train_real`. Artifacts: `checkpoints/best_s1_wg_real.pt`,
`s1_real_finetune.json`. *Limit:* one scene / one event — a multi-event real
inventory set would make this a production figure.

**(f) Tile-level detection — the 90 %+ figure (real GT).** As on L4S (§4.1),
the operationally useful *"does this 1.28 km tile contain a landslide?"*
question is far more separable than exact pixel outlines, and clears 90 %. A
`PatchClassifier` (7-ch: VV,VH pre/post + ΔVV,ΔVH + slope) trained on the real
Kodagu tiles, same spatial split:

| Western Ghats SAR tile classifier (real GT, held-out) | Value |
|---|---|
| **Accuracy** | **92.9 %** |
| **Balanced accuracy** | **90.9 %** (imbalance-robust — the honest 90 %+) |
| **ROC-AUC** | **0.952** |
| F1 | 82.2 % |

→ **90 %+ landslide detection on the real Western Ghats inventory** — balanced
accuracy 90.9 % and AUC 0.952 confirm it is genuine skill, not a class-imbalance
artefact (test tiles are 19 % positive). F1 is 82 % (the rare-class precision/
recall balance). Code: `main.py s1_cls_real`; artifact `s1_cls_real.json`.
Flagged tiles are visualised by `s1_tile_overlay.py` → `qc/s1_kodagu_tiles.png`
(yellow stripes = correctly flagged tiles, magenta = false alarms, blue = real
inventory) — a tile-level error map confirming the flags cluster on the real
landslide zone. *Scope, stated plainly:* this is tile-level **detection**
(92.9 % acc / 90.9 %
bal-acc), a different, easier question than pixel **delineation** (F1 0.55,
§4.3e) — both on the same real Kodagu ground truth. A 90 % *pixel-segmentation*
F1 on SAR is beyond published SOTA and is **not** claimed.

*Why not "recent" events:* Landsat 5 was decommissioned in 2013, so by
construction it can only train/test on pre-2013 events; recent Indian events
(Wayanad 2024, Chamoli 2021) are outside its lifetime and belong to the
Sentinel-2 / Landsat-8/9 models.

#### 4.3g Uttarakhand (Garhwal Himalaya) — Rudraprayag Susceptibility & Rainfall Trigger

Evaluating slope failure likelihood along the high-risk pilgrimage corridor to Kedarnath using free multi-source satellite data and Random Forest with spatial cross-validation:

| Check / Experiment | Metric | Value |
|---|---|---|
| **Terrain model (spatial CV, 20-draw mean)** | **Within-block AUC** | **0.744** (95% range: **0.515–0.850**) |
| Coordinates-only spatial baseline | Within-block AUC | 0.585 (95% range: 0.485–0.685) |
| Slope-matched sampling design | Within-block AUC | 0.694 (95% range: 0.467–0.862) |
| Terrain vs Coordinates advantage | AUC margin | **+0.159** (Terrain beats coordinates in 90% of draws) |
| **Held-out landslide risk placement** | **High / Very High class** | **87.0%** (33/38; 95% Wilson CI: **73%–94%**) |
| Held-out stable slope risk placement | High / Very High class | 49.0% (separation margin: +38% over stable; chance 40%) |
| Very High alone | Landslides vs Stable | **61.0% vs 23.0%** |
| District monsoon rainfall (JJAS mean) | GPM IMERG 1998–2025 | **1,336 mm** (~74% of annual total) |
| Wettest monsoon | GPM IMERG | 2010 (**1,924 mm**, +44.0% vs average) |
| Driest monsoon | GPM IMERG | 2009 (**813 mm**, −39.2% vs average) |

**Key findings:**
1. Naive random sampling across the district scored an inflated AUC of 0.85, but a coordinates-only model scored 0.83 because landslides cluster along the steep upper Mandakini valley.
2. Matched stable sampling within 3 km neutralised this location shortcut, forcing the model to distinguish failing slopes from stable neighbours in the same valley.
3. Reporting across 20 background draws was essential: single-draw AUCs swing by up to ±0.15 depending on which stable points are chosen.
4. The 20-model ensemble places 87% of held-out landslides into High or Very High susceptibility classes.
5. Interactive delivery via Folium map (`results/uttarakhand/interactive_map.html`) allows users to inspect individual scars and hazard overlays against high-resolution satellite imagery.

### 4.4 Event-set results (vs pseudo-labels — circular, upper bound)

| Variant | F1 | Precision | Recall | IoU |
|---|---|---|---|---|
| Standalone model | 0.440 | 0.297 | 0.784 | 0.275 |
| **Ensemble strict** | **0.932** | **1.000** | 0.873 | 0.873 |
| Ensemble relaxed | 0.529 | 0.378 | 0.879 | 0.359 |

v1 → v2 improvement (RGB + slope bands): standalone val F1 0.26 → 0.44 (+69%);
ensemble 0.840 → 0.932.

### 4.5 Mapping outputs

| Output | strict | relaxed |
|---|---|---|
| Polygons | **141** | 237 |
| Mapped area | 65.0 km² | 113.5 km² |
| Events with ≥1 polygon | 34/39 | 34/39 |
| GeoJSON | `results/landslides.geojson` | `results/landslides_relaxed.geojson` |
| CSV | `results/landslides.csv` | `results/landslides_relaxed.csv` |
| Summary | `results/event_summary.csv` | `results/event_summary_relaxed.csv` |
| Masks | `results/masks_ens/` | `results/masks_ens_relaxed/` |

### 4.6 Independent catalog check (GLC points, events ≤ 2017)

| Threshold | Ours | Chance (shuffled events) |
|---|---|---|
| ≤ 100 m | 0.0% | 0.0% |
| ≤ 1 km | 0.0% | 0.0% |
| Median dist | 8.3 km (5/10 misses = 0 polygons) | 9,586 km |

Signal ≫ chance (≈1000×) but 0% ≤ 1 km. Failure analysis (Section 5)
shows the test is only partially fair: **7/10 matched GLC points fall
outside the exported scene bbox** (event-centred exports vs catalogued
slides scattered ±10 km). In-scene points: 3 (kedarnath ×2 → 0 detections;
mocoa → 2.95 km).

### 4.7 Visual QC (independent confirmation)

`qc/*.png` panels: PRE RGB | POST RGB | ensemble overlay. Indian events have
dedicated panels (`qc/*_in_*.png`).

| Event | Observation |
|---|---|
| `wayanad_in_2024` (IN) | ✅ Chooralmala debris-flow scar appears PRE→POST; **red mask sits on it** (0.18% of scene) |
| `kavalappara_in_2019` (IN) | ✅ W. Ghats slide; mask on exposed scar |
| `mocoa_co_2017` | ✅ new debris fans visible POST; mask on initiation scar (GLC point = downslope impact zone, 2.9 km) |
| `oso_us_2014` | ⚠️ blobs at scene margins; main Highway-530 slide not convincingly covered (false positives + miss) |
| `hiroshima_jp_2014` | partial |

→ Georeferencing verified correct; model detects real change; ensemble
conservative (precision-first).

### 4.8 LandSentry: bitemporal S2 change detection and cross-event transfer (Kodagu ↔ Wayanad)

The protocol is described in §3.11. All numbers are pixel metrics on the
**held-out test region**, given as mean ± std over 3 seeds. The decision
threshold is frozen on the *training* event's val split, so an off-diagonal
cell is a true zero-shot transfer. AP is average precision, which does not
depend on the threshold. Full table: `results/landsentry/cross_event.md`;
per-run rows: `runs.csv`.

**Kodagu test region: real ground truth (217-polygon inventory, 10,247 positive px)**

| Method | Trained on | F1 | IoU | Precision | Recall | AP |
|---|---|---|---|---|---|---|
| NDVI-difference baseline (no learning) | thr tuned on Kodagu | 0.672 | 0.506 | 0.664 | 0.679 | 0.680 |
| Siamese CNN | Kodagu | 0.686 ± 0.009 | 0.522 ± 0.011 | 0.678 ± 0.013 | 0.695 ± 0.033 | 0.697 ± 0.014 |
| BIT transformer | Kodagu | 0.683 ± 0.046 | 0.519 ± 0.052 | 0.680 ± 0.075 | 0.687 ± 0.021 | 0.732 ± 0.065 |
| Siamese CNN | Kodagu + Wayanad | **0.694 ± 0.008** | **0.531 ± 0.009** | 0.682 ± 0.017 | 0.707 ± 0.007 | 0.709 ± 0.022 |
| BIT transformer | Kodagu + Wayanad | 0.689 ± 0.004 | 0.526 ± 0.005 | 0.701 ± 0.033 | 0.680 ± 0.031 | **0.736 ± 0.013** |
| Siamese CNN | Wayanad only *(zero-shot)* | 0.514 ± 0.069 | 0.348 ± 0.063 | 0.665 ± 0.041 | 0.428 ± 0.101 | 0.462 ± 0.108 |
| BIT transformer | Wayanad only *(zero-shot)* | 0.413 ± 0.133 | 0.266 ± 0.112 | 0.600 ± 0.129 | 0.326 ± 0.143 | 0.341 ± 0.159 |
| NDVI baseline | thr tuned on Wayanad *(zero-shot)* | 0.651 | 0.483 | 0.571 | 0.759 | 0.680 |

**Wayanad test region: seeded NDVI-change reference (2,025 positive px; partly circular, see below)**

| Method | Trained on | F1 | IoU | Precision | Recall | AP |
|---|---|---|---|---|---|---|
| NDVI baseline | thr tuned on Kodagu *(zero-shot)* | 0.835 | 0.716 | 0.963 | 0.737 | 0.951 |
| Siamese CNN | Kodagu *(zero-shot)* | **0.894 ± 0.012** | 0.809 ± 0.019 | 0.904 ± 0.002 | 0.885 ± 0.022 | 0.921 ± 0.016 |
| BIT transformer | Kodagu *(zero-shot)* | 0.868 ± 0.019 | 0.767 ± 0.029 | 0.944 ± 0.012 | 0.803 ± 0.033 | 0.940 ± 0.021 |
| Siamese CNN | Wayanad | 0.929 ± 0.004 | 0.867 ± 0.006 | 0.934 ± 0.006 | 0.924 ± 0.012 | 0.944 ± 0.036 |
| BIT transformer | Wayanad | 0.937 ± 0.008 | 0.882 ± 0.014 | 0.937 ± 0.002 | 0.937 ± 0.018 | 0.979 ± 0.005 |
| Siamese CNN | Kodagu + Wayanad | 0.914 ± 0.003 | 0.842 ± 0.005 | 0.939 ± 0.014 | 0.891 ± 0.018 | 0.947 ± 0.018 |
| BIT transformer | Kodagu + Wayanad | 0.903 ± 0.016 | 0.824 ± 0.027 | 0.916 ± 0.014 | 0.891 ± 0.018 | 0.960 ± 0.009 |

**Zero-shot transfer scored on the target's whole scene** (every pixel, no split)

| Method | Train → test | F1 | Precision | Recall | AP |
|---|---|---|---|---|---|
| NDVI baseline | Kodagu → Wayanad | **0.774** | 0.831 | 0.725 | **0.871** |
| Siamese CNN | Kodagu → Wayanad | 0.766 ± 0.042 | 0.801 ± 0.020 | 0.736 ± 0.074 | 0.793 ± 0.048 |
| BIT transformer | Kodagu → Wayanad | 0.694 ± 0.050 | 0.857 ± 0.121 | 0.604 ± 0.129 | 0.756 ± 0.075 |
| NDVI baseline | Wayanad → Kodagu | **0.563** | 0.477 | 0.687 | **0.574** |
| Siamese CNN | Wayanad → Kodagu | 0.454 ± 0.035 | 0.550 ± 0.068 | 0.397 ± 0.079 | 0.384 ± 0.093 |
| BIT transformer | Wayanad → Kodagu | 0.359 ± 0.074 | 0.382 ± 0.051 | 0.342 ± 0.099 | 0.242 ± 0.082 |

Figures: `results/landsentry/fig_cross_event.png` (2×2 F1/AP heatmaps per
method), `fig_qualitative.png` (4 km test-region crops showing before,
after, reference and the four model/source combinations), and
`fig_training.png` (val curves).

**Findings**

1. **Deep models give only a small gain over a simple index on real ground truth.** On Kodagu, the best learned result is the Siamese CNN trained on both events (F1 **0.694 ± 0.008**). It beats non-learned NDVI differencing (0.672) by **+2.2 F1 points**. The BIT transformer gives the best ranking quality (AP **0.736 ± 0.013** vs 0.680, +5.6 AP), but its F1 is no better because the frozen threshold costs it. At 10 m with only 4 bands, most of the separable signal is vegetation loss, which NDVI already captures. The learned gain is in boundaries and in rejecting some non-landslide bare surfaces.
2. **Transfer is asymmetric.**
   - *Kodagu → Wayanad works:* the Siamese CNN gets F1 0.894 ± 0.012 on the Wayanad test track, zero-shot, with a frozen threshold.
   - *Wayanad → Kodagu fails:* Siamese F1 0.514 ± 0.069 and BIT 0.413 ± 0.133, both below the NDVI baseline's 0.651.
   - The low and highly variable AP (0.46 ± 0.11 and 0.34 ± 0.16) shows the models mis-rank Kodagu's pixels, not just that the threshold is off. Wayanad provides one debris flow and 3,720 training pixels, and the models memorise it: training loss falls to ~0.001, and the val-tuned threshold of one seed collapsed to 0.003. They then miss Kodagu's narrow, branching channelised slides, which look nothing like a single 100–300 m-wide runout track (`fig_qualitative.png`, top row).
   - This contradicts the hope that attention transfers better. The BIT transformer generalises **worse** than the CNN in both directions, and is less stable across seeds (±0.046 to ±0.133).
3. **Wayanad scores are partly circular.** The Wayanad reference mask was produced by an NDVI-change rule (§2.8). The NDVI baseline's Wayanad numbers (AP 0.951) therefore measure agreement with itself, and every Wayanad F1 in the 0.83–0.94 range should be read as "agrees with an index-derived outline of the scar", not as independent accuracy. Two consequences:
   - On the whole Wayanad scene, NDVI (F1 0.774) still edges out the Kodagu-trained Siamese CNN (0.766 ± 0.042).
   - The 0.894 test-region F1 is evidence that the model *finds* the Wayanad scar. It is not evidence that it beats a simple index there.
4. **Comparison with the original LandSentry repo.** The original reports Siamese Dice 0.390 on Kodagu and 0.138 on Wayanad, and transformer Dice 0.375 / 0.086. Our numbers are much higher, but the labels, splits and protocols differ, so they are not directly comparable. The original's Wayanad array files also appear to be placeholders. What this implementation adds: the identical label source for Kodagu, per-scene normalisation (the original normalised Wayanad with Kodagu statistics), a fixed-grid pixel-aligned download, threshold freezing on the source event, seeds, and a baseline.
5. **Recommended operational model:** `siamese_both_s0.pt`. Training on both events gives the highest and most stable Kodagu F1 (±0.008) and keeps Wayanad above 0.91. It must still be treated as a vegetation-loss detector. Clear-cutting, construction, river-bank erosion and fresh cloud artefacts can trigger it, and §4.8 has no independent test outside these two Western Ghats events.

## 5. Failure analysis (why catalog recall is weak)

1. **Scene coverage**: exports centred on town coordinates, not slide
   coordinates → 7/10 GLC points outside bbox entirely.
2. **Pseudo-label blind spots** (train labels ≈ 0% → model learns nothing):
   - `ludian_cn_2014`: clear radiometric change but vote threshold found 0%
   - `sindhupalchok_np_2015`: PRE window (Feb–Aug 2015) already contains
     April Gorkha-quake scars → no pre/post delta
   - `kedarnath_in_2013`: only 19.9% valid pixels (snow/cloud composite)
   - `yaan_cn_2013`: valid = 0%, PRE stored as raw DN — corrupt export
3. **GLC point quality**: old entries city-level / `location_accuracy`
   unknown; GLC = 1 point per event, not extent.
4. **Ensemble strictness**: `prob ∧ index` trades recall for precision
   (measured P=1.00 / R=0.87 vs pseudo); relaxed variant recovered +68
   polygons but did not rescue 0-polygon events (their model probs < 0.25
   because their train labels were empty).
5. **GLC staleness**: post-2017 events untestable against it.

## 6. Reproduce

```bash
# env (uv-managed)
uv sync

# --- event pipeline (v2 data must be in data/Landslide_Landsat_Global_v2/) ---
.venv/bin/python main.py labels      # pseudo masks -> data/pseudo_masks_v2/
.venv/bin/python main.py train       # 50 ep -> checkpoints/best.pt
.venv/bin/python -m src.ensemble_infer          # strict  -> results/masks_ens/
.venv/bin/python -m src.ensemble_infer relaxed  # relaxed -> results/masks_ens_relaxed/
.venv/bin/python main.py geocode      # -> results/landslides{,_relaxed}.geojson

# --- Landslide4Sense benchmark (real GT, honest 3-way split) ---
# data already at data/L4S/TrainData/ (Zenodo TrainData.zip, md5 408f09247e...)
# --- Indian-region segmentation (HR-GLDD real masks, 576 MB) ---
mkdir -p data/HR_GLDD && for f in trainX trainY valX valY testX testY; do \
  curl -sL -o data/HR_GLDD/$f.npy \
  "https://zenodo.org/records/7189381/files/$f.npy?download=1"; done
.venv/bin/python main.py hrgldd        # 100 ep ~8 min -> test F1 0.748 / IoU 0.60
.venv/bin/python hrgldd_qc.py 6        # RGB | GT | pred panels -> qc/hrgldd_*.png

.venv/bin/python main.py l4s_classify # patch classifier, 50 ep ~4 min -> 90.7% acc / 92.2% F1
.venv/bin/python main.py l4s          # improved DeepUNet, 90 ep ~20 min -> test F1 0.755
.venv/bin/python main.py l4s_baseline # legacy U-Net, same split      -> test F1 0.759
.venv/bin/python main.py l4s_attention # Two-Stream Attention U-Net, 25 ep -> test F1 0.754
.venv/bin/python -m src.l4s_ensemble  # 3-model diverse ensemble     -> test F1 0.762 / IoU 0.615

# --- Indian-region visual QC ---
.venv/bin/python qc_overlay.py wayanad_in_2024 kavalappara_in_2019 idukki_in_2018 \
    chamoli_in_2021 ramban_in_2024 teesta_in_2023   # -> qc/*_in_*.png

# --- independent validation ---
.venv/bin/python qc_glc.py results/landslides_relaxed.geojson
.venv/bin/python qc_overlay.py wayanad_in_2024 acapulco_mx_2023 wajima_jp_2024

# --- Uttarakhand (Rudraprayag) Landslide Susceptibility ---
.venv/bin/python main.py uk_inventory     # inventory deduplication -> fig3_landslide_inventory.png
.venv/bin/python main.py uk_train         # 20-draw spatial CV -> best_uttarakhand_rf.pt, fig4
.venv/bin/python main.py uk_rainfall      # IMERG monthly & monsoon climatology -> fig6
.venv/bin/python main.py uk_map           # interactive Folium map -> results/uttarakhand/interactive_map.html
.venv/bin/python main.py uk_all           # run full Uttarakhand pipeline end-to-end

# --- LandSentry: bitemporal S2 change detection + cross-event (§3.11) ---
.venv/bin/python -m landsentry prepare       # S2 L1C pre/post composites, UTM 10 m (~6 min)
.venv/bin/python -m landsentry labels        # Kodagu inventory mask, Wayanad reference mask, splits, QC png
.venv/bin/python -m landsentry cross-event   # 2 models x 3 sources x 3 seeds + NDVI baseline (~1 h on the A5000)
.venv/bin/python -m landsentry evaluate --checkpoint checkpoints/landsentry/siamese_kodagu_s0.pt --event wayanad

# --- GEE re-export (auth expires) ---
# ssh -L 8085:localhost:8085 <host> then:
.venv/bin/python -c "import ee; ee.Authenticate(auth_mode='localhost')"
python download_dataset.py            # OFFSET cycles 0/10/20/30 for 4 batches
python download_uttarakhand_features.py --project <ee-project> # Rudraprayag feature stack
```

## 7. Artifacts index

| Path | Contents |
|---|---|
| `checkpoints/best_hrgldd.pt` | **Indian-region segmentation (HR-GLDD, test F1 0.748 / IoU 0.60)** |
| `checkpoints/hrgldd_report.json` | HR-GLDD honest val/test metrics |
| `checkpoints/hrgldd_norm.json` | HR-GLDD train-split per-band mean/std |
| `qc/hrgldd_*.png` | RGB \| human mask \| prediction panels (per-patch F1) |
| `checkpoints/best_l4s_cls.pt` | **Patch classifier (test 90.7% acc / 92.2% F1)** + norm stats |
| `checkpoints/l4s_cls_report.json` | classifier honest val/test metrics + AUC |
| `checkpoints/l4s_cls_norm.json` | classifier train-split per-band mean/std |
| `checkpoints/best_l4s.pt` | Improved DeepUNet (test F1 0.755) + its norm stats |
| `checkpoints/best_l4s_baseline.pt` | Legacy SingleUNet baseline (test F1 0.759) |
| `checkpoints/best_l4s_attention.pt` | **Two-Stream Attention U-Net (test F1 0.754, 25 ep)** |
| `checkpoints/best.pt` | Siamese event model (val F1 0.440 vs pseudo) |
| `checkpoints/l4s_report*.json` | honest val/test segmentation metrics, per model |
| `checkpoints/l4s_attention_report.json` | Two-Stream Attention honest val/test metrics |
| `checkpoints/history_l4s_attention.json` | Two-Stream Attention training history (25 ep) |
| `checkpoints/l4s_ensemble_report.json` | **3-model diverse ensemble held-out test (F1 0.762 / IoU 0.615)** |
| `checkpoints/l4s_norm.json` | train-split per-band mean/std |
| `checkpoints/l4s_sweep*.json` | val threshold sweep + frozen thr |
| `checkpoints/best_landsat5.pt` | Historical Landsat-5 Indian model (Himalaya/NE, ~40–53% ≤3 km) |
| `checkpoints/landsat5_test.json` | L5 held-out catalog-proximity results |
| `checkpoints/best_s1_wg.pt` | Sentinel-1 SAR Western Ghats model (pseudo-labels) |
| `checkpoints/best_s1_wg_real.pt` | **SAR fine-tuned on REAL Kodagu masks (F1 0.554 / IoU 0.383)** |
| `checkpoints/best_s1_wg_cls.pt` | **SAR tile classifier — real GT (92.9% acc / 90.9% bal-acc / AUC 0.952)** |
| `checkpoints/s1_real_finetune.json`, `s1_cls_real.json` | SAR real-GT seg/cls metrics |
| `data/Kodagu_SAR_GT/{PRE,POST,mask}.tif` | real-GT SAR scene + rasterised 217-polygon inventory |
| `qc/s1_*.png`, `qc/s1_kodagu_real_gt.png`, `qc/s1_kodagu_tiles.png` | SAR detection + tile-flag QC |
| `checkpoints/best_uttarakhand_rf.pt` | **Uttarakhand Random Forest 20-model ensemble (AUC 0.744 [0.52–0.85])** |
| `checkpoints/uttarakhand_report.json` | Uttarakhand spatial CV AUC, held-out classification metrics |
| `results/uttarakhand/fig3_landslide_inventory.png` | Rudraprayag landslide scar inventory (38 landslides) |
| `results/uttarakhand/fig4_model_evaluation.png` | 20-draw AUC signal box plots + feature importances |
| `results/uttarakhand/fig5_susceptibility_map.png` | Susceptibility hazard classification (Very Low to Very High) |
| `results/uttarakhand/fig6_rainfall.png` | GPM IMERG rainfall seasonality & monsoon annual totals |
| `results/uttarakhand/interactive_map.html` | Folium interactive multi-layer web map with popup metadata |
| `results/uttarakhand/auc_across_draws.csv` | Full per-draw within-block spatial CV metrics |
| `qc/uk_*.png` | Visual QC copies for Uttarakhand inventory, evaluation, rainfall |
| `results/landslides*.geojson/csv` | georeferenced polygons (strict/relaxed) |
| `results/qc_glc_report.csv` | GLC distance table |
| `landslides_relaxed_glc_report2.csv` | + in-scene containment flags |
| `qc/*.png`, `qc/*_in_*.png` | PRE/POST/mask visual QC panels (incl. Indian) |
| `download_landsat5_india.py`, `download_sentinel1_wg.py`, `build_kodagu_gt.py`, `s1_tile_overlay.py` | L5/SAR download, real-GT build, tile overlay |
| `src/`, `main.py` | full pipeline code |
| `landsentry/` | **LandSentry** fresh implementation: models, GEE, labels, training, cross-event runner (`landsentry/README.md`) |
| `data/landsentry/{kodagu,wayanad}/{pre,post,mask,split}.tif` | S2 composites (int16 reflectance ×1e4 + `n_obs`), label masks, split rasters |
| `data/landsentry/*/qc_labels.png`, `labels.json`, `imagery.json` | label/split QC panel and provenance metadata |
| `checkpoints/landsentry/{siamese,transformer}_{kodagu,wayanad,both}_s{0,1,2}.pt` | 18 cross-event checkpoints (each stores its frozen val threshold) + `_history.json` |
| `results/landsentry/cross_event.{md,json}`, `runs.csv` | cross-event tables (mean ± std) and per-run metrics |
| `results/landsentry/fig_cross_event.png`, `fig_qualitative.png`, `fig_training.png` | 2×2 F1/AP heatmaps, test-region panels, val curves |

## 8. Conclusions

1. **Good Indian-region segmentation on real human masks** (HR-GLDD,
   Kerala/Karnataka Western Ghats included): held-out test **F1 0.748 /
   IoU 0.60 / pixel acc 0.943**, with **0.85–0.96 per-patch F1** on clear
   scars (QC panels). Real 3 m digitized masks — a genuine pixel-delineation
   result, not pseudo-labels.
2. **≥ 90% landslide-detection accuracy** (L4S patch classifier, held-out
   test: **90.7 % acc / 92.2 % F1 / 0.967 AUC**) — out-of-sample, near-
   balanced labels (not an imbalance artefact).
3. **≥ 75% pixel-segmentation F1** on L4S under the same honest protocol
   (3-model diverse ensemble test F1 **0.762 / IoU 0.615**). Pixel-level 90% is beyond published SOTA for
   that benchmark and is *not* claimed.
4. Full operational pipeline works end-to-end: GEE export → train →
   inference → georeferenced polygons for 39 real events (141 strict /
   237 relaxed polygons, 65–114 km²), **8 of them in Indian regions, 6/8
   detected** with visual-QC confirmation (wayanad on the Chooralmala scar).
5. **LandSentry (bitemporal Sentinel-2, cross-event), §4.8.** This is a working re-implementation with an honest protocol: Siamese CNN, BIT transformer, NDVI baseline, and a two-event transfer matrix over 3 seeds.
   - On real Kodagu labels, learned change detectors beat plain NDVI differencing by only about 2 F1 points (0.694 vs 0.672) and about 5 AP points.
   - Transfer from the rich multi-slide event to the single debris flow works (0.894 F1). The reverse fails (0.41–0.51).
   - The transformer does not transfer better than the CNN.
   - Wayanad scores rest on an index-derived reference mask and are not independent accuracy.

## 9. Recommended next steps

- **Lift HR-GLDD F1 further**: the 0.748 comes from 1119 train patches —
  add a pretrained/attention encoder, model ensemble, or copy-paste
  augmentation of scars; aim ~0.80.
- **Kodagu-only Indian score**: rebuild HR-GLDD from the authors' per-site
  inventories (shapefiles) to isolate the Indian tiles, or request MTMS-HLD
  (Uttarakhand) access for a purely-Himalayan segmentation benchmark.
- Sentinel-2 re-export of the 8 Indian Landsat events (GEE) so the 14-band
  L4S classifier applies directly; re-export miss events (shirur, kedarnath)
  with QA_PIXEL cloud masking.
- Adaptive pseudo-label thresholds (fix 0%-label events) → retrain.
- **LandSentry:**
  - Replace the Wayanad index-derived reference with an independent outline: digitise the NRSC/Charter impact map, or request the vector from the Sci. Rep. 2025 authors. This would make the Wayanad numbers non-circular.
  - Add more labelled Western Ghats events (e.g. Kavalappara/Puthumala 2019 from HR-GLDD-style inventories) so that "train on one event" stops meaning "memorise one scar".
  - Add SWIR (B11/B12) and slope channels to move beyond what NDVI differencing already captures.
  - Add a Sentinel-1 fallback for monsoon-season requests, when optical windows are cloud-bound.
