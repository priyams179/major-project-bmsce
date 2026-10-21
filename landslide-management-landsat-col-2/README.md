# Landslide Management — Landsat Collection-2 Pipeline

Pre/post Landsat 8/9 detection → georeferenced landslide polygons,
with honest accuracy benchmarking against Landslide4Sense human labels.

## Quick start

```bash
uv sync

.venv/bin/python main.py labels        # pseudo-labels (multi-index consensus)
.venv/bin/python main.py train         # Siamese U-Net, 50 epochs
.venv/bin/python -m src.ensemble_infer # strict ensemble masks
.venv/bin/python main.py geocode       # GeoJSON + CSV polygons
.venv/bin/python main.py l4s           # Landslide4Sense real-GT benchmark (F1 0.75)
```

## Results at a glance

| Check | Number |
|---|---|
| **L4S F1 vs human labels** | **0.754** (target ≥ 75% met) |
| Ensemble F1 vs pseudo-labels | 0.932 (circular — upper bound) |
| Polygons mapped (strict / relaxed) | 141 / 237, 65.0 / 113.5 km² |
| Visual QC | `qc/*.png` — masks on real scars (wayanad, mocoa) |

## Read this first

**[REPORT.md](REPORT.md)** — full technical report: data, method, results,
failure analysis, reproduction commands, artifact index.
