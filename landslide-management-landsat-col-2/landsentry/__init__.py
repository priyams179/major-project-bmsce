"""LandSentry -- bitemporal Sentinel-2 landslide change detection.

A fresh, self-contained implementation (independent of ``src/``) of:

* a Siamese CNN change detector (shared encoder, |pre - post| skip fusion,
  U-Net decoder),
* a BIT-style transformer change detector (semantic tokenizer -> transformer
  encoder over bitemporal tokens -> transformer decoder back to pixels),
* an explicit cross-event generalisation protocol (train on Kodagu 2018 or
  Wayanad 2024, test on both).

Entry point: ``python -m landsentry <command>`` (see ``landsentry/README.md``).
"""
