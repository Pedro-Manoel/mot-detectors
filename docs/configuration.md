# Configuration reference

An experiment is one YAML file. The CLI expands it into the run grid and writes results
under `results/<name>/`. The repository holds `paper_main.yml` (the grid reported in the
paper) and `smoke.yml` (tiny subset). `run`, `status` and `analyze` accept `--name <other>` to
use `results/<other>/` for the same grid, for instance a second run to compare with the first.

## Full example

```yaml
name: paper_main                 # experiment name, output under results/paper_main/
device: "0"                      # "0" is a GPU index, or "cpu"
runs: 5                          # repetitions per config (FPS and latency statistics)

detector:
  conf: 0.25                     # detection confidence threshold (uniform)
  imgsz: 640                     # inference image size
  half: false                    # FP16 inference

trackeval:
  root: TrackEval                # path to the cloned TrackEval repo
  python: .venvs/trackeval/Scripts/python.exe   # interpreter of the TrackEval venv
  do_preproc: false              # TrackEval preprocessing; a no-op on the pedestrian-only GT

keep_tracks: true                # keep mot_results.txt per run (reproducibility)

paths:
  output_root: results           # results/<name>/...
  dataset_root: datasets         # raw datasets
  prepared_root: prepared        # prepared sequences (filtered GT and frames)

detectors:                       # the detector grid
  - { family: yolo11, scales: [n, s, m, l, x], modes: [nms] }
  - { family: rtdetr, scales: [l, x],          modes: [nms] }
  - { family: yolo26, scales: [n, s, m, l, x], modes: [nms, end2end] }

trackers: [bytetrack, botsort]
reid_weights: osnet_x0_25_msmt17.pt   # ReID weights, BoT-SORT only

sequences:
  MOT17:   [MOT17-02, MOT17-04, MOT17-05]
  MOT20:   [MOT20-02, MOT20-03, MOT20-05]
  SOMPT22: [SOMPT22-07, SOMPT22-10, SOMPT22-12]
```

## Fields

| Key | Type | Default | Notes |
|---|---|---|---|
| `name` | str | required | Experiment id; output goes to `results/<name>/`. |
| `device` | str | required | `"0"` for GPU or `"cpu"`. |
| `runs` | int | required | Repetitions per config; the tables average over them. |
| `detector.conf` | float | required | Confidence threshold, the same for all detectors. |
| `detector.imgsz` | int | required | Inference image size. |
| `detector.half` | bool | `false` | FP16 inference. |
| `trackeval.root` | str | required | Path to the TrackEval clone. |
| `trackeval.python` | str | required | TrackEval venv interpreter; a relative path is resolved to absolute, and the Windows (`Scripts/python.exe`) and POSIX (`bin/python`) venv layouts are interchangeable. |
| `trackeval.do_preproc` | bool | `false` | TrackEval's distractor preprocessing. The paper uses `false`; `true` changes nothing on the pedestrian-only GT that `prepare` writes, so standard MOTChallenge scoring also needs the full GT (see the README, *Scoring choice*). |
| `keep_tracks` | bool | `true` | Keep the raw `mot_results.txt` per run. |
| `paths.output_root` | str | `results` | Root for results. |
| `paths.dataset_root` | str | `datasets` | Raw datasets. |
| `paths.prepared_root` | str | `prepared` | Prepared sequences. |
| `detectors[].family` | str | required | `yolo11`, `yolo26` or `rtdetr`. |
| `detectors[].scales` | list | required | Subset of `n,s,m,l,x`; RT-DETR only has `l,x`. |
| `detectors[].modes` | list | required | `nms` and/or `end2end`; `end2end` is only valid for `yolo26`. |
| `trackers` | list | required | `bytetrack`, `botsort`. |
| `reid_weights` | str | `""` | OSNet weights, applied to BoT-SORT only. |
| `sequences` | map | required | `DATASET: [SEQUENCE, ...]`; a MOT17 detector suffix is dropped (`MOT17-02-FRCNN` is `MOT17-02`). |

## Recipes

Add a sequence, one line:

```yaml
sequences:
  MOT17: [MOT17-02, MOT17-04, MOT17-05, MOT17-09]   # added MOT17-09
```

Restrict the scales, for instance to evaluate only YOLO26 medium and large:

```yaml
detectors:
  - { family: yolo26, scales: [m, l], modes: [nms, end2end] }
```

Fast partial run for debugging:
`python mot.py run experiments/myexp.yml --max-frames 50` (always from scratch, in
`results/<name>_quicktest/`)

Repeat the paper's grid in a separate folder and compare it with a first run in
`results/paper_main/`:

```
python mot.py run     experiments/paper_main.yml --name paper_repro
python mot.py analyze experiments/paper_main.yml --name paper_repro
python mot.py compare results/paper_main results/paper_repro
```

Compare a run of the grid with the paper's runs:
`python mot.py compare paper_results results/paper_main`

## Validity

`config.expand_grid` enforces that `end2end` is only emitted for `yolo26`. Invalid
combinations such as `yolo11 + end2end` are silently skipped, matching the original study.
`config.load_config` rejects unknown keys and modes other than `nms` and `end2end`, so a
misspelled key cannot change an experiment without notice.
