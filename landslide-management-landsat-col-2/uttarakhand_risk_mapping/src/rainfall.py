"""
Step 4 — GPM IMERG monsoon rainfall analysis.

Reads:
  data/rainfall_monthly_rudraprayag.csv   — district-avg monthly rainfall (1998–2025)
  data/rainfall_monsoon_totals.csv        — JJAS totals by year

Writes:
  outputs/fig6_rainfall.png              — rainfall figure (3 panels)

If the CSVs are not present, falls back to downloading from GEE
(requires ee authentication and project ID).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "outputs"


# ── GEE download (optional) ──────────────────────────────────────────────────

def _download_via_gee(project_id: str, data_dir: Path) -> None:
    """Download monthly IMERG district averages via GEE and save to CSV."""
    try:
        import ee
    except ImportError:
        raise ImportError("earthengine-api not installed. Run: uv add earthengine-api")

    print(f"Authenticating GEE ({project_id}) …")
    ee.Authenticate()
    ee.Initialize(project=project_id)

    aoi = (
        ee.FeatureCollection("FAO/GAUL/2025/level2")
        .filter(ee.Filter.eq("ISO3_CODE", "IND"))
        .filter(ee.Filter.stringContains("GAUL2_NAME", "Rudra"))
        .geometry()
    )
    imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07").select("precipitation")

    def district_mean(img):
        date = img.date()
        hours = img.date().advance(1, "month").difference(date, "hour")
        mm = img.multiply(hours)
        stat = mm.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=5000, maxPixels=1e8
        )
        return ee.Feature(None, {
            "date": date.format("YYYY-MM"),
            "mm": stat.get("precipitation"),
        })

    fc = imerg.map(district_mean)
    rows = fc.aggregate_array("date").getInfo()
    vals = fc.aggregate_array("mm").getInfo()

    rain = pd.DataFrame({"date": rows, "mm": vals})
    rain["date"] = pd.to_datetime(rain["date"])
    rain["year"] = rain["date"].dt.year
    rain["mon"] = rain["date"].dt.month
    rain = rain.dropna(subset=["mm"]).sort_values("date").reset_index(drop=True)
    rain.to_csv(data_dir / "rainfall_monthly_rudraprayag.csv", index=False)
    print(f"Saved {len(rain)} monthly rows → {data_dir / 'rainfall_monthly_rudraprayag.csv'}")

    jjas = (
        rain[rain["mon"].between(6, 9)]
        .groupby("year")
        .agg(mm=("mm", "sum"), months=("mm", "size"))
    )
    jjas = jjas[jjas["months"] == 4].copy()
    clim = jjas["mm"].mean()
    jjas["vs_average_pct"] = 100 * (jjas["mm"] / clim - 1)
    jjas.to_csv(data_dir / "rainfall_monsoon_totals.csv")
    print(f"Saved monsoon totals → {data_dir / 'rainfall_monsoon_totals.csv'}")


# ── analysis & figures ───────────────────────────────────────────────────────

def _load_rainfall(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rain = pd.read_csv(data_dir / "rainfall_monthly_rudraprayag.csv")
    # Support two column layouts:
    #   (a) original: 'date' column  — from GEE download
    #   (b) pre-built CSV: 'month' column holding the date string
    if "date" not in rain.columns and "month" in rain.columns:
        rain = rain.rename(columns={"month": "date"})
    rain["date"] = pd.to_datetime(rain["date"])
    if "year" not in rain.columns:
        rain["year"] = rain["date"].dt.year
    if "mon" not in rain.columns:
        rain["mon"] = rain["date"].dt.month

    jjas = pd.read_csv(data_dir / "rainfall_monsoon_totals.csv", index_col=0)
    return rain, jjas


def plot_rainfall(rain: pd.DataFrame, jjas: pd.DataFrame, out_path: Path) -> None:
    """Figure 6 — three-panel rainfall figure."""
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1.25], wspace=0.3)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) climatological monthly rainfall
    mon_grp = rain.groupby("mon")["mm"]
    mean_m = mon_grp.mean()
    p10 = mon_grp.quantile(0.1)
    p90 = mon_grp.quantile(0.9)
    bar_cols = ["#1f5fa8" if m in (6, 7, 8, 9) else "#a6c8e8" for m in mean_m.index]
    ax1.bar(mean_m.index, mean_m, color=bar_cols, edgecolor="black", linewidth=0.5,
            yerr=[mean_m - p10, p90 - mean_m],
            capsize=3, error_kw=dict(linewidth=0.8, ecolor="black"))
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax1.set_ylabel("Monthly rainfall (mm)")
    ax1.set_title("(a) Seasonality (1998–2025)\nerror bars = 10th–90th percentile")
    ax1.text(0.98, 0.97, "■ Monsoon (JJAS)", transform=ax1.transAxes,
             color="#1f5fa8", ha="right", va="top", fontsize=8)

    # (b) monsoon totals by year
    if "mm" in jjas.columns:
        years = jjas.index.astype(int)
        mm = jjas["mm"].values
        pct = jjas["vs_average_pct"].values if "vs_average_pct" in jjas.columns else np.zeros_like(mm)
        clim = mm.mean()
        bar_cols2 = ["#c0392b" if p > 0 else "#2980b9" for p in pct]
        ax2.bar(years, mm, color=bar_cols2, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax2.axhline(clim, color="black", linestyle="--", linewidth=1,
                    label=f"Average {clim:.0f} mm")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Monsoon total (mm)")
        ax2.set_title("(b) Monsoon totals by year\n(June–September)")
        ax2.legend(fontsize=8)
        ax2.text(0.02, 0.97, "■ Above average   ■ Below average",
                 transform=ax2.transAxes, fontsize=7, va="top",
                 color="gray")

    # (c) inter-annual variability summary
    ax3.axis("off")
    if "mm" in jjas.columns:
        mm_s = jjas["mm"]
        stats = {
            "Years with data": f"{len(jjas)}",
            "Mean (JJAS)": f"{mm_s.mean():.0f} mm",
            "Min": f"{mm_s.min():.0f} mm ({int(mm_s.idxmin())})",
            "Max": f"{mm_s.max():.0f} mm ({int(mm_s.idxmax())})",
            "Std dev": f"{mm_s.std():.0f} mm",
            "CV": f"{mm_s.std()/mm_s.mean()*100:.1f}%",
        }
        text = "\n".join(f"{k}: {v}" for k, v in stats.items())
        ax3.text(0.1, 0.5, text, transform=ax3.transAxes, fontsize=11,
                 va="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#eaf3fb", edgecolor="#1f5fa8"))
        ax3.set_title("(c) Summary statistics")

    plt.suptitle("Monsoon Rainfall — Rudraprayag District", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 6 saved → {out_path}")


def run_rainfall(
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
    gee_project: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_csv = data_dir / "rainfall_monthly_rudraprayag.csv"

    if not monthly_csv.exists():
        if gee_project:
            print("Downloading rainfall from GEE …")
            _download_via_gee(gee_project, data_dir)
        else:
            raise FileNotFoundError(
                f"{monthly_csv} not found.\n"
                "Either supply --gee-project to download it, or copy from the original repo's data/."
            )

    print("Loading rainfall CSVs …")
    rain, jjas = _load_rainfall(data_dir)
    print(f"Monthly records: {len(rain)}  |  Monsoon years: {len(jjas)}")

    # summary stats
    clim = jjas["mm"].mean() if "mm" in jjas.columns else float("nan")
    print(f"Average JJAS total: {clim:.0f} mm")
    if "mm" in jjas.columns:
        print(f"Wettest year: {jjas['mm'].idxmax()} ({jjas['mm'].max():.0f} mm)")
        print(f"Driest year: {jjas['mm'].idxmin()} ({jjas['mm'].min():.0f} mm)")

    plot_rainfall(rain, jjas, out_dir / "fig6_rainfall.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gee-project", help="GEE project ID (only needed if CSVs absent)")
    args = parser.parse_args()
    run_rainfall(gee_project=args.gee_project)
