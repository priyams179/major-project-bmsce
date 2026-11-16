"""Uttarakhand Landslide Risk Mapping — Python pipeline.

Adapts the Colab-based notebooks (makkergauri/uttarakhand-landslide-risk)
for local execution without Google Colab or Drive dependencies.

Steps:
  1. features.py     — build conditioning-factor raster via GEE export helpers
  2. inventory.py    — deduplicate landslide inventory, sample stable points
  3. model.py        — Random Forest with spatial cross-validation (20-draw)
  4. rainfall.py     — GPM IMERG monsoon rainfall analysis
  5. interactive_map.py — Folium interactive susceptibility map
  6. pipeline.py     — CLI that runs all steps end-to-end
"""

