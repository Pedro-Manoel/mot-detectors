# Architecture

The pipeline is a small package (`core/`) driven by one CLI (`mot.py`). Each module has
a single responsibility and a small public surface.

## Modules

| Module | Responsibility | Key public functions |
|---|---|---|
| `config.py` | Parse a YAML into a typed `Config` and expand it into the run grid. | `load_config`, `expand_grid`, `weight_filename`, `is_valid_combination` |
| `data.py` | Prepare a raw MOT sequence (filter the GT to pedestrians, copy or link the frames, write `seqinfo.ini` and `meta.json`) and load prepared frames. | `prepare_sequence`, `load_sequence` |
| `detect.py` | Load an Ultralytics detector, run per-frame prediction (with the YOLO26 head toggle), parse boxes to `(N,6)`, plus GPU setup helpers. | `setup_gpu`, `load_detector`, `predict_frame`, `parse_ultralytics_results` |
| `track.py` | Load a BoxMOT tracker and parse its outputs into MOT rows. | `load_tracker`, `parse_track_outputs`, `write_mot_txt` |
| `evaluate.py` | Stage a TrackEval workspace in a temporary folder, run TrackEval as a subprocess in its own venv, parse the summary. | `stage_trackeval_workspace`, `call_trackeval`, `parse_trackeval_summary` |
| `pipeline.py` | Process one (detector, sequence) group: per repetition and tracker, the paper's loop that reads, detects and tracks each frame; then evaluate and write `metrics.csv` per run. | `process_group` |
| `metrics.py` | Timing reconstruction: per-frame `imread+det+track` into the `metrics.csv` timing fields. | `run_timing_fields`, `block`, `reconstruct_total` |
| `runner.py` | Resumable batch loop: group by detector-sequence, model cache, skip completed runs, isolate failures, ledger, atomic IO, repository-relative paths in `metrics.csv`. | `run_batch`, `status`, `group_specs`, `ModelCache`, `run_dir_for`, `portable_path`, `write_csv_row_atomic` |
| `report.py` | Concatenate the `metrics.csv` files into `all_results.csv` and compute the quality and efficiency tables (the paper's Tables II to IV, as CSV and in the paper's LaTeX format). | `aggregate`, `build_tables`, `quality_table`, `efficiency_table` |
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
   |  metrics.run_timing_fields: fps_total = 1000/mean(imread+det+track)
   v  atomically writes results/<name>/<DATASET>/<SEQ>/<run_dir>/metrics.csv, one per run
report.aggregate    -> results/<name>/all_results.csv
report.build_tables -> results/<name>/tables/quality_{bytetrack,botsort}.{csv,tex},
                       efficiency.{csv,tex}                    (mot.py analyze)
```

## Design invariants

### Measurement protocol

`cudnn.benchmark=True` is autotuned, as in the paper. Each repetition follows the paper's
protocol: for each tracker, one loop reads, detects and tracks every frame, and `fps_total`
comes from the per-frame `imread+det+track` measured in that loop, so detection runs once per
tracker and the work a tracker does between frames can slow down the next detection.
`conf=0.25`, `imgsz=640`, `half=False` and `classes=[0]` are uniform. Quality is computed per
repetition, since benchmark mode is not bit-deterministic; the variation is about 0.03 HOTA on
average and is smoothed by averaging over 5 runs, which is the paper's protocol.

### Units of work

`RunSpec` is the unit for resume and output, while the (detector, sequence) group is the
unit of execution because the detector is loaded once per group. There is one `metrics.csv`
per `RunSpec`.

### Results are append-only and resumable

`metrics.csv` is written atomically (temp file plus `os.replace`), so its presence always
means a complete run. `run` skips runs whose `metrics.csv` exists and continues from where
it stopped; a failing detector-sequence group is recorded in `results/<name>/_runs.jsonl` and
does not abort the batch. `mot.py status <cfg>` reports done, pending and the groups whose
last attempt failed, without side effects.

### Quick tests are isolated

`run --max-frames N` truncates each sequence and writes to a sibling
`results/<name>_quicktest/` root, always from scratch and with no ledger and no aggregation,
so a development check cannot contaminate canonical results or the resume state.

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
variants to `report.MODEL_ORDER`, which sets the labels and the row order of the LaTeX tables;
a variant missing there is left out of them.

### A new tracker

`track.load_tracker` instantiates the BoxMOT class by name, so add a branch returning the
configured tracker. Confirm that its output column order matches `track.parse_track_outputs`,
which expects `[x1, y1, x2, y2, id, conf, cls, ...]`, and adjust the parser if it differs.
Then add it to `trackers:` in the YAML and give it a display name in `report.TRACKER_NAMES`.
`report.build_tables` writes the per-tracker tables for every tracker in the data, and the
ReID weights go to `botsort` only (`config.expand_grid`).

### A new dataset or sequence

Place the data in MOT Challenge format under `datasets/<DATASET>/train/<SEQUENCE>/`, with
`seqinfo.ini`, `img1/` and `gt/gt.txt`. If the folder layout differs, extend
`data.resolve_sequence_path`, which already handles the MOT17 detector suffixes. Confirm the
pedestrian class id, since `data.PEDESTRIAN_CLASS_ID` is `1` for MOT and SOMPT22. Then add
the sequence under `sequences:` in the YAML; `mot.py prepare` filters the GT and caches the
prepared sequence. Registering its published density in `densities.json` is optional: without
it, `prepare` warns and falls back to the GT-computed value.

### Scoring, metrics and tables

Standard MOTChallenge scoring needs the full GT (without the pedestrian filter of
`data.prepare_sequence`) together with `trackeval.do_preproc: true`; on the pedestrian-only
GT that flag changes nothing. For more or fewer metrics, edit
`evaluate.TRACKEVAL_WANTED_METRICS`; each entry must exist in the TrackEval summary, and the
columns then flow automatically into `metrics.csv`. To add a table, add a builder to
`report.py` and call it from `build_tables`; `mot.py analyze` rebuilds
`results/<name>/tables/` after any run.

### Tests

Pure-logic tests live in `tests/` (config expansion, parsers, table values, reproduction
checks, and the paper's quality tables against `paper_results/`) and run without a GPU:
`python -m pytest -q`. The GPU end-to-end check is the smoke test (`experiments/smoke.yml`).
For a full reproduction, `mot.py check` validates the environment against the paper's and
`mot.py compare` compares the re-run with `paper_results/`.
