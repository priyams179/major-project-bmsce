"""
Step 5 — Interactive susceptibility map (Folium).

Reads:
  data/training_points_matched_wgs84.geojson   — training points
  data/rudraprayag_susceptibility.tif          — susceptibility raster (optional)
  data/landslide_review_rudraprayag*.geojson   — review decisions

Writes:
  outputs/interactive_map.html                 — standalone interactive HTML map

The map layers:
  1. Base layers: OpenStreetMap, Esri World Imagery, OpenTopoMap
  2. Landslide inventory markers (red triangles)
  3. Stable-point markers (blue dots)
  4. Susceptibility raster overlay (if rudraprayag_susceptibility.tif exists)
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import folium
import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "outputs"

# Rudraprayag district approximate centre
_CENTRE = [30.49, 79.07]
_ZOOM = 10


def _load_training_points(data_dir: Path) -> gpd.GeoDataFrame:
    path = data_dir / "training_points_matched_wgs84.geojson"
    return gpd.read_file(path).to_crs("EPSG:4326")


def _load_review(data_dir: Path) -> gpd.GeoDataFrame | None:
    files = glob.glob(str(data_dir / "landslide_review_rudraprayag*.geojson"))
    if not files:
        return None
    return gpd.read_file(max(files, key=os.path.getmtime)).to_crs("EPSG:4326")


def _raster_to_png(tif_path: Path) -> tuple[bytes, list[float]] | None:
    """Convert susceptibility raster to PNG bytes + bounds for Folium overlay."""
    try:
        import rasterio
        import io
        from rasterio.enums import Resampling
        from PIL import Image as PILImage
    except ImportError:
        return None

    with rasterio.open(tif_path) as src:
        # downsample to max 512 px on the longer side
        factor = max(src.width, src.height) // 512 + 1
        data = src.read(
            1,
            out_shape=(1, src.height // factor, src.width // factor),
            resampling=Resampling.average,
        ).squeeze()
        bounds = src.bounds

    cmap = matplotlib.colormaps["RdYlGn_r"]
    flat = data[~np.isnan(data)]
    if len(flat) == 0:
        return None
    vmin, vmax = np.quantile(flat, 0.02), np.quantile(flat, 0.98)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = (cmap(norm(data)) * 255).astype(np.uint8)
    rgba[np.isnan(data)] = [0, 0, 0, 0]  # transparent outside district

    img = PILImage.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    folium_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return buf.getvalue(), folium_bounds


def build_map(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # base map with multiple tile layers
    m = folium.Map(location=_CENTRE, zoom_start=_ZOOM, control_scale=True)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite imagery",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap",
    ).add_to(m)

    # susceptibility overlay
    tif_path = data_dir / "rudraprayag_susceptibility.tif"
    if tif_path.exists():
        result = _raster_to_png(tif_path)
        if result is not None:
            png_bytes, folium_bounds = result
            import base64
            png_b64 = base64.b64encode(png_bytes).decode()
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{png_b64}",
                bounds=folium_bounds,
                opacity=0.65,
                name="Susceptibility (20-model average)",
                cross_origin=False,
                zindex=1,
            ).add_to(m)
            print("Susceptibility overlay added")
    else:
        print("Susceptibility raster not found — overlay skipped (run step 3 first)")

    # training points
    try:
        pts = _load_training_points(data_dir)
        ls_grp = folium.FeatureGroup(name=f"Landslides (n={int((pts['label']==1).sum())})")
        st_grp = folium.FeatureGroup(name=f"Stable points (n={int((pts['label']==0).sum())})")

        for _, row in pts.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            if row["label"] == 1:
                folium.Marker(
                    [lat, lon],
                    popup=folium.Popup(f"Landslide<br>({lat:.4f}°N, {lon:.4f}°E)", max_width=200),
                    icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon"),
                ).add_to(ls_grp)
            else:
                folium.CircleMarker(
                    [lat, lon],
                    radius=5, color="blue", fill=True, fill_opacity=0.7,
                    popup=folium.Popup(f"Stable<br>({lat:.4f}°N, {lon:.4f}°E)", max_width=200),
                ).add_to(st_grp)

        ls_grp.add_to(m)
        st_grp.add_to(m)
        print(f"Added {(pts['label']==1).sum()} landslide and {(pts['label']==0).sum()} stable markers")
    except FileNotFoundError:
        print("training_points_matched_wgs84.geojson not found — markers skipped")

    # review candidates (all 299)
    review = _load_review(data_dir)
    if review is not None:
        cand_grp = folium.FeatureGroup(name="Reviewed candidates (all 299)", show=False)
        for _, row in review.iterrows():
            if row.geometry is None:
                continue
            lon, lat = row.geometry.x, row.geometry.y
            decision = row.get("decision", "?")
            color = {"yes": "red", "no": "gray", "unsure": "orange"}.get(decision, "gray")
            folium.CircleMarker(
                [lat, lon],
                radius=3, color=color, fill=True, fill_opacity=0.6,
                popup=folium.Popup(f"Decision: {decision}", max_width=150),
            ).add_to(cand_grp)
        cand_grp.add_to(m)
        print(f"Added {len(review)} review candidates")

    folium.LayerControl(collapsed=False).add_to(m)

    # legend (HTML element)
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background-color:white; padding:10px; border:1px solid #aaa;
                border-radius:5px; font-size:12px;">
      <b>Susceptibility</b><br>
      <span style="background:#1a9850;padding:0 8px;">&nbsp;</span> Very low<br>
      <span style="background:#91cf60;padding:0 8px;">&nbsp;</span> Low<br>
      <span style="background:#ffffbf;padding:0 8px;">&nbsp;</span> Moderate<br>
      <span style="background:#fc8d59;padding:0 8px;">&nbsp;</span> High<br>
      <span style="background:#d73027;padding:0 8px;">&nbsp;</span> Very high<br>
      <hr style="margin:4px 0">
      <span style="color:red;">▲</span> Landslide &nbsp;
      <span style="color:blue;">●</span> Stable
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_html = out_dir / "interactive_map.html"
    m.save(str(out_html))
    print(f"Interactive map saved → {out_html}")
    return out_html


if __name__ == "__main__":
    build_map()

