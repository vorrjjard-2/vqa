# data/

Local-only data directory. Nothing here is committed (see `.gitignore`).

Suggested layout:

- `raw/` — original downloads (MIMIC-CXR images, reports).
- `interim/` — detector outputs, intermediate per-study JSON.
- `processed/` — final `(json_input, target_text)` pairs ready for training.
