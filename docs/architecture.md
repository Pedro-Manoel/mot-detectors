# Architecture

The pipeline is a small package (`core/`) driven by one CLI (`mot.py`). Each module has
a single responsibility and a small public surface.

## Modules

| Module | Responsibility | Key public functions |
|---|---|---|
| `config.py` | Parse a YAML into a typed `Config`, expand it into the run grid, build the manifest. | `load_config`, `expand_grid`, `weight_filename`, `is_valid_combination`, `environment_manifest` |
| `data.py` | Prepare a raw MOT sequence (filter the GT to pedestrians, copy or link the frames, write `seqinfo.ini` and `meta.json`) and load prepared frames. | `prepare_sequence`, `load_sequence` |
| `detect.py` | Load an Ultralytics detector, run per-frame prediction (with the YOLO26 head toggle), parse boxes to `(N,6)`, report parameters and FLOPs, plus GPU setup helpers. | `setup_gpu`, `load_detector`, `predict_frame`, `parse_ultralytics_results`, `model_complexity` |
| `track.py` | Load a BoxMOT tracker and parse its outputs into MOT rows. | `load_tracker`, `parse_track_outputs`, `write_mot_txt` |
| `evaluate.py` | Stage a TrackEval workspace in a temporary folder, run TrackEval as a subprocess in its own venv, parse the summary. | `stage_trackeval_workspace`, `call_trackeval`, `parse_trackeval_summary` |
| `pipeline.py` | Process one (detector, sequence) group: per repetition and tracker, the paper's loop that reads, detects and tracks each frame, or with `shared_detection` one detection pass whose boxes each tracker replays; then evaluate and write `metrics.csv` per run. | `process_group` |
| `metrics.py` | Timing reconstruction: per-frame `imread+det+track` into the `metrics.csv` timing fields. | `run_timing_fields`, `block`, `reconstruct_total` |
| `runner.py` | Resumable batch loop: group by detector-sequence, model cache, skip completed runs, isolate failures, ledger, atomic IO, repository-relative paths in `metrics.csv`. | `run_batch`, `status`, `group_specs`, `ModelCache`, `run_dir_for`, `portable_path`, `write_csv_row_atomic` |
| `report.py` | Concatenate the `metrics.csv` files into `all_results.csv` and compute the quality, efficiency and per-sequence tables (the paper's Tables II to IV also as LaTeX, and Table I from the weights), dispatching the analysis tables below. | `aggregate`, `build_tables`, `quality_table`, `efficiency_table`, `models_table` |
| `analysis.py` | Analyses over `all_results.csv`, no GPU: HOTA decomposition (DetA/AssA/LocA), t-based confidence intervals, density degradation from MOT17 to MOT20, paired Wilcoxon for NMS against E2E (sequence-level pairs, runs averaged), detection-vs-association attribution, the effective BoxMOT hyperparameters. | `hota_decomposition_table`, `degradation_table`, `nms_vs_e2e_significance`, `summary_ci_table`, `attribution_table`, `tracker_hyperparameters` |
| `diagnostics.py` | Detection-level diagnostics from persisted detections and GT (duplicate-hypothesis rate, detection AP/AR) and box jitter from the kept tracks. | `duplicate_rate`, `box_jitter`, `detection_pr`, `average_precision`, `build_diagnostics_table` |
| `reproduce.py` | Reproduction support: check the environment, weights and prepared GT against the paper's (`experiments/paper_environment.json`); compare the quality tables of a run with a reference (the paper's runs in `paper_results/`, or another run). | `check_software`, `check_files`, `compare_results`, `file_sha256` |

## Data flow

```
YAML experiment
   |  config.load_config -> Config
   |  config.expand_grid -> list[RunSpec]  (dataset x sequence x family x scale x mode
   |                                        x tracker x run)
   v
runner.run_batch (resumable, groups by detector-sequence, size-1 model cache):
   for each pending (dataset, seq, family, scale, mode) group: pipeline.process_group
   |  load the detector once (cache)
   |  for rep in 1..runs, for each tracker:
   |     read, detect and track each frame in one loop (the paper's protocol), evaluate
   |     (with shared_detection: one detection pass per rep, replayed by each tracker)
   |  metrics.run_timing_fields: fps_total = 1000/mean(imread+det+track)
   v  atomically writes results/<name>/<DATASET>/<SEQ>/<run_dir>/metrics.csv, one per run
report.aggregate    -> results/<name>/all_results.csv
report.build_tables -> results/<name>/tables/*.csv, *.tex
   |  core: quality_{bytetrack,botsort}, efficiency, per_sequence
   |  analysis: hota_decomposition_*, degradation_*, attribution_*, quality_ci_*,
   |            nms_vs_e2e_significance_*, hyperparameters
   v  diagnostics: detection_diagnostics (save_detections), box_jitter (keep_tracks)
```

With `save_detections: true` the first repetition also writes the raw boxes to
`results/<name>/<DATASET>/<SEQ>/_detections/<family>_<scale>_<mode>.npz`, once per
detector-sequence and git-ignored. `mot.py analyze` scores them against the prepared GT into
`detection_diagnostics.csv` (duplicate-hypothesis rate, AP@0.5, recall and precision).

## Design invariants

### Measurement protocol

`cudnn.benchmark=True` is autotuned, as in the paper. By default each repetition follows the
paper's protocol: for each tracker, one loop reads, detects and tracks every frame, and
`fps_total` comes from the per-frame `imread+det+track` measured in that loop, so detection
runs once per tracker and the work a tracker does between frames can slow down the next
detection. With `shared_detection: true`, a detection-only pass measures detection alone and
caches the boxes, and each tracker then replays them in its own pass: half the detector work,
and a light tracker's FPS is never inflated by a heavy one, but that FPS is a best case (a real
detect-then-track loop leaves the GPU idle between detections) and not comparable with the
paper's. Either way `conf=0.25`, `imgsz=640`, `half=False` and `classes=[0]` are uniform, and
the `metrics.csv` schema is the same. Quality is computed per repetition, since benchmark mode
is not bit-deterministic; the variation is about 0.03 HOTA on average and is smoothed by
averaging over 5 runs, which is the paper's protocol.

### Units of work

`RunSpec` is the unit for resume and output, while the (detector, sequence) group is the
unit of execution because the detector is loaded once per group. There is one `metrics.csv` per `RunSpec`.

### Results are append-only and resumable

`metrics.csv` is written atomically (temp file plus `os.replace`), so its presence always
means a complete run. `run` skips runs whose `metrics.csv` exists and continues from where
it stopped; a failing detector-sequence group is recorded in `results/<name>/_runs.jsonl` and
does not abort the batch. `mot.py status <cfg>` reports done, pending and the groups whose
last attempt failed, without side effects.

### Quick tests are isolated

`run --max-frames N` truncates each sequence and writes to a sibling
`results/<name>_quicktest/` root, always from scratch and with no manifest, no ledger and no
aggregation, so a
development check cannot contaminate canonical results or the resume state.

## Extension points

Keep the numeric path behavior-preserving for the existing configs: run the smoke test before
and after a change (the second with `--name`) and `mot.py compare` the two folders.

### A new detector family

`config.weight_filename` maps `(family, scale)` to a `.pt` filename; add a branch if the new
family does not follow `"{family}{scale}.pt"`, as RT-DETR already does with
`rtdetr-{scale}.pt`. `detect.load_detector` chooses the Ultralytics class (`YOLO` or
`RTDETR`) by substring, so a family needing a different class or a head toggle gets a branch
there. If it supports `end2end`, relax `config.is_valid_combination`, which by default only
allows `yolo26` to run `end2end`. Then add the family to the YAML `detectors:` list and its
variants to `report.MODEL_ORDER`, which sets the labels and the row order of the LaTeX tables
and of Table I; a variant missing there is left out of them.

### A new tracker

`track.load_tracker` instantiates the BoxMOT class by name, so add a branch returning the
configured tracker. Confirm that its output column order matches `track.parse_track_outputs`,
which expects `[x1, y1, x2, y2, id, conf, cls, ...]`, and adjust the parser if it differs.
Then add it to `trackers:` in the YAML, give it a display name in `report.TRACKER_NAMES` and
its constructor defaults in `analysis.TRACKER_SETTINGS` (for `hyperparameters.csv`).
`report.build_tables` writes the per-tracker tables for every tracker in the data, and the
ReID weights go to `botsort` only (`config.expand_grid`).

### A new dataset or sequence

Place the data in MOT Challenge format under `datasets/<DATASET>/train/<SEQUENCE>/`, with
`seqinfo.ini`, `img1/` and `gt/gt.txt`. If the folder layout differs, extend
`data.resolve_sequence_path`, which already handles MOT17 detector suffixes and SOMPT22
naming variants. Confirm the pedestrian class id, since `data.PEDESTRIAN_CLASS_ID` is `1`
for MOT and SOMPT22. Then add the sequence under `sequences:` in the YAML;
`mot.py prepare` filters the GT and caches the prepared sequence. Registering its published
density in `densities.json` is optional: without it, `prepare` warns and falls back to the
GT-computed value.

### Scoring, metrics and tables

Standard MOTChallenge scoring needs the full GT (without the pedestrian filter of
`data.prepare_sequence`) together with `trackeval.do_preproc: true`; on the pedestrian-only
GT that flag changes nothing. For more or fewer
metrics, edit `evaluate.TRACKEVAL_WANTED_METRICS`; each entry must exist in the TrackEval
summary, and the columns then flow automatically into `metrics.csv`. To add a table, add a
builder to `report.py` and call it from `build_tables`; `mot.py analyze` rebuilds
`results/<name>/tables/` after any run.

### Tests

Pure-logic tests live in `tests/` (config expansion, parsers, table values, reproduction
checks, and the paper's quality tables against `paper_results/`) and run without a GPU:
`python -m pytest -q`. The GPU end-to-end check is the smoke test (`experiments/smoke.yml`).
For a full reproduction, `mot.py check` validates the environment against the paper's and
`mot.py compare` compares the re-run with `paper_results/`.
