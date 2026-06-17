# E164 Default Probe: cropdense_min050_trackletveto_delay4_sizeadaptive_probation48

This is the current E164 offline-selector probe after adding the second-stage crop classifier guard and testing two global REACQUIRE tracklet guards.

The strict positive tracklet-confirmation guard blocks seg02 background reacquire, but over-suppresses a visually valid seg03 late reacquire. Plain tracklet-rejection veto preserves seg03 but leaves a new seg02 f86/f90 background reacquire. The current script therefore uses a delayed global confirmation window: neutral global seeds need 4 consecutive confirmations, while a later positive tracklet signal can shortcut back to the normal 2-frame confirmation.

The current default also enables a size-adaptive small-area global REACQUIRE path. Normal global seeds still require `--reacquire-global-min-area 1000`. Small global seeds may enter from `100 px^2` upward only when they are tracklet-confirmed, are not explicitly sequence-gate rejected, pass the crop gate across the pending window, and stay within the small-area global distance cap. After a confirmed small-area global seed, memory-near REACQUIRE is suppressed for 48 frames, while the normal `max_recover_frames` grace is kept so the immediate bridge back into TRACK is not broken.

## Runner

Use:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_e164_default_probe.ps1
```

Add `-SaveVideo` when an annotated review video is needed.

## Default Inputs

- predictions: `C:\Users\pc\Desktop\tiny object detection\runs\ura27_sequence_gate_smoke\dji_broad_seg01_v2\predictions.jsonl`
- video: `D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\raw_videos\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg01_000000_000598.mp4`
- crop classifier: `C:\Users\pc\Desktop\tiny object detection\runs\dji_dense_stage_b_hardneg_repair_20260530\models\crop_binary_dji_hardneg.pt`
- output: `runs\e164_selector_probe\dji_broad_seg01_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48`

## Pinned Selector Settings

- `--reacquire-max-distance-px 240`
- `--reacquire-confirm-frames 2`
- `--reacquire-stale-after-frames 10`
- `--reacquire-global-min-area 1000`
- `--reacquire-global-small-min-area 100`
- `--reacquire-global-small-max-distance-px 1200`
- `--reacquire-global-small-memory-probation-frames 48`
- `--reacquire-global-max-area 6000`
- `--reacquire-global-min-detector-score 0.32`
- `--reacquire-global-reject-tracklet-rejected`
- `--reacquire-global-delayed-confirm-frames 4`
- `--reacquire-appearance-weight 0.50`
- `--reacquire-crop-weight 1.0`
- `--reacquire-min-crop-drone-score 0.50`

## Evidence From Probe Run

Default script run summary:

- total frames: `541`
- accepted candidate frames: `237`
- reacquired frames: `26`
- reacquire memory frames: `24`
- reacquire global frames: `2`
- reacquire global small-area frames: `0`
- crop rejected frames: `33`
- crop soft-scored frames: `72`
- global tracklet rejected frames: `1`
- delayed global confirm frames: `4`
- memory recover frames: `47`
- lost frames: `257`

Behavioral check:

- f190/f200/f201 remain `LOST`, blocking the previous background reacquire.
- f258 still reacquires the red target with crop score `1.000000` through `confirmed_global_candidate_after_stale_lost`.
- f258-f540 zoom review on the current delay4 output shows the box continuing on the red target, first on the picnic table and later near the wall/ground. No obvious seg02-style background drift was observed in the sampled frames.

Review artifacts:

- annotated video: `runs\e164_selector_probe\dji_broad_seg01_v2_e164_default_cropdense_min050_trackletveto_delay4\selector_annotated.mp4`
- f258-f540 zoom sheet: `runs\e164_selector_probe\dji_broad_seg01_v2_e164_default_cropdense_min050_trackletveto_delay4\selector_zoom_sheet_f258_f540_trackletveto_delay4.jpg`

## Current Decision

Pre-probation decision: `cropdense_min050_trackletveto_delay4_sizeadaptive` was the strongest E164 offline-selector probe at this point. It preserved seg01 f258, blocked seg02 f86/f90 and the previous high-crop background reacquires, preserved the seg03 f268-f304 red-target recovery, and recovered seg04 small-target global seeds without globally lowering the area floor. The current default is the probation48 update below.

## Held-Out Segment 02 Cross-Check

Segment:

- video: `D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\raw_videos\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg02_001831_002429.mp4`
- sequence-gated predictions: `runs\ura27_sequence_gate_smoke\dji_broad_seg02_v2\predictions.jsonl`

Sequence gate setup:

- source predictions: `D:\datasets\stage_a_mixed\ura25_full_pipeline_formal_20260525\dji\broad_recall\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg02_001831_002429\predictions.jsonl`
- raw drone predictions: `87`
- filtered drone predictions: `42`
- rejected drone predictions: `45`

Soft crop control, without hard crop threshold:

- output: `runs\e164_selector_probe\dji_broad_seg02_v2_reacquire240_confirm2_global_hi032_appw050_cropdense_w100`
- reacquired frames: `17`
- global reacquire frames: `3`
- crop rejected frames: `0`

Default `cropdense_min050`:

- output: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050`
- accepted candidate frames: `418`
- reacquired frames: `11`
- global reacquire frames: `2`
- crop rejected frames: `36`
- memory recover frames: `29`
- lost frames: `123`

Default `cropdense_min050_trackletconfirm`:

- output: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050_trackletconfirm`
- accepted candidate frames: `2`
- reacquired frames: `0`
- global reacquire frames: `0`
- crop rejected frames: `8`
- global tracklet rejected frames: `126`
- memory recover frames: `6`
- lost frames: `562`

Default `cropdense_min050_trackletveto`:

- output: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050_trackletveto`
- accepted candidate frames: `4`
- reacquired frames: `2`
- global reacquire frames: `1`
- crop rejected frames: `94`
- global tracklet rejected frames: `37`
- memory recover frames: `12`
- lost frames: `554`

Default `cropdense_min050_trackletveto_delay4`:

- output: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050_trackletveto_delay4`
- accepted candidate frames: `2`
- reacquired frames: `0`
- global reacquire frames: `0`
- crop rejected frames: `95`
- global tracklet rejected frames: `37`
- delayed global confirm frames: `4`
- memory recover frames: `6`
- lost frames: `562`

Blocked by the crop gate:

- f2/f4: low crop score `0.000000`, blocked early false reacquire around the lower bench/table region.
- f482: crop score `0.106115`, blocked wall/grass background reacquire.
- f500/f504/f506: crop scores `0.000000`, `0.000000`, `0.003856`, blocked table/ground background reacquire.

Residual failure:

- The retained seg02 reacquires at f24/f37/f131/f193/f303/f406/f408/f412/f414/f432/f440 score high under the crop classifier (`0.867970` to `0.999912`) but visually cluster around the wall/grass/shrub line. This means crop classifier gating helps, but is not sufficient as a general held-out fix.
- With tracklet confirmation enabled, those same frames become `LOST` instead of `REACQUIRE`: f24/f37/f131/f193/f303/f406/f408/f412/f414/f432/f440 are all blocked.
- With tracklet-rejection veto enabled, those same 11 old failure frames are also blocked, but f86/f90 becomes a new background reacquire around trees/grass. The f86 crop score is still high (`0.945308`), so crop threshold alone still cannot fix this.
- With tracklet-rejection veto plus delay4, seg02 has `reacquired_frames=0`; f86 stays `LOST` as a pending global candidate and never writes to memory.

Review artifact:

- blocked-candidate contact sheet: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050\seg02_crop_gate_blocked_contact_sheet.jpg`
- accepted-reacquire contact sheet: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050\seg02_min050_reacquire_contact_sheet.jpg`

Updated decision:

Strict `trackletconfirm` is too hard; plain `trackletveto` is too soft. `trackletveto_delay4` is the current balanced probe because it delays neutral global seeds while allowing positive tracklet evidence to shortcut the delay.

## Held-Out Segment 03 Cross-Check

Segment:

- video: `D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\raw_videos\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg03_003901_004499.mp4`
- sequence-gated predictions: `runs\ura27_sequence_gate_smoke\dji_broad_seg03_v2\predictions.jsonl`
- source predictions: `D:\datasets\stage_a_mixed\ura25_full_pipeline_formal_20260525\dji\broad_recall\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg03_003901_004499\predictions.jsonl`

Sequence gate summary:

- raw drone predictions: `192`
- filtered drone predictions: `4`
- rejected drone predictions: `188`
- confirmed tracklets: `2`
- rejected tracklets: `544`

Strict `cropdense_min050_trackletconfirm`:

- output: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_trackletconfirm`
- accepted candidate frames: `153`
- reacquired frames: `2`
- global reacquire frames: `0`
- global tracklet rejected frames: `40`
- lost frames: `379`

No-guard control:

- output: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_noguard_control`
- accepted candidate frames: `174`
- reacquired frames: `8`
- global reacquire frames: `1`
- lost frames: `341`
- f259-f304 zoom sheet: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_noguard_control\selector_zoom_sheet_f259_f304_noguard_control.jpg`

Tracklet-rejection veto:

- output: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_trackletveto`
- accepted candidate frames: `174`
- reacquired frames: `8`
- global reacquire frames: `1`
- global tracklet rejected frames: `1`
- lost frames: `341`

Tracklet-rejection veto with delay4:

- output: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_trackletveto_delay4`
- accepted candidate frames: `163`
- reacquired frames: `9`
- global reacquire frames: `1`
- global tracklet rejected frames: `1`
- delayed global confirm frames: `4`
- lost frames: `353`
- f258-f304 zoom sheet: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_trackletveto_delay4\selector_zoom_sheet_f258_f304_trackletveto_delay4.jpg`

Behavioral finding:

- The strict guard blocks the visually continuous f259-f304 red-target reacquire because f259 is `tracklet_is_drone=false` with a very low tracklet probability, even though the candidate is not explicitly `tracklet_rejected`.
- The veto guard preserves that segment by allowing neutral/unapplied tracklet rows, but seg02 shows that this is not enough to prevent all background global reacquire.
- Delay4 keeps seg03 in pending through f258-f261, confirms global at f268, and then reacquires/tracks the same red target through f304 in the zoom sheet.

## Held-Out Segment 04 Cross-Check

Segment:

- video: `D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\raw_videos\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg04_005971_006569.mp4`
- source predictions: `D:\datasets\stage_a_mixed\ura25_full_pipeline_formal_20260525\dji\broad_recall\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg04_005971_006569\predictions.jsonl`
- annotations: `D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\annotations\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg04_005971_006569_boxes.csv`

Source prediction setup:

- raw drone predictions: `3283`
- filtered drone predictions: `996`
- rejected drone predictions: `2287`
- sequence-gated `runs\ura27_sequence_gate_smoke\dji_broad_seg04_v2` input does not exist yet, so this check used the source tracklet-filtered predictions directly.

Current default `cropdense_min050_trackletveto_delay4` with `global_min_area=1000`:

- output: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_sourcepred`
- accepted candidate frames: `25`
- reacquired frames: `0`
- global reacquire frames: `0`
- crop rejected frames: `17`
- global tracklet rejected frames: `2`
- delayed global confirm frames: `4`
- lost frames: `539`

No-delay control with the same crop/tracklet settings:

- output: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_nodelay_sourcepred`
- reacquired frames: `0`
- global reacquire frames: `0`
- lost frames: `539`

Behavioral finding:

- Under the current default, seg04 has no REACQUIRE even when the delayed global window is removed. This means the miss is not caused by delay4 over-delaying a true target.
- The real seg04 target is much smaller than the current global area floor. Annotated boxes are roughly `20-44 px` wide by `8-16 px` high, so many true target boxes are below `1000 px^2`.
- The source predictions do contain true target proposals on sparse annotated frames: IoU >= `0.1` on `6/10` annotated frames in both `predictions_raw.jsonl` and `predictions.jsonl`. The default selector misses them because the early memory locks onto a wrong candidate and later global REACQUIRE excludes small target boxes with `--reacquire-global-min-area 1000`.

No-crop diagnostic:

- no-crop no-delay output: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_trackletveto_nocrop_nodelay_sourcepred`
- no-crop delay4 output: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_trackletveto_nocrop_delay4_sourcepred`
- no-crop no-delay reacquired `7` frames, while no-crop delay4 reacquired `0` and left `17` pending candidates.
- Interpolated sparse-GT sanity check shows those no-crop/no-delay reacquires are not the true target: f114/f262/f272/f292/f294/f296/f441 all have interpolated IoU `0.000` and center distance roughly `340-932 px`.
- review sheet: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_trackletveto_nocrop_delay4_sourcepred\seg04_delay4_vs_nodelay_reacquire_zoom_sheet.jpg`

Area-floor diagnostic:

- delay4 with `--reacquire-global-min-area 100`: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_area100_sourcepred`
- no-delay with `--reacquire-global-min-area 100`: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_nodelay_area100_sourcepred`
- both area100 runs hit `5/10` sparse GT annotation frames at IoU >= `0.1` and recover true target frames around f88 and f206.
- delay4 area100: `reacquired_frames=12`, `global=6`, `lost_frames=287`
- no-delay area100: `reacquired_frames=12`, `global=6`, `lost_frames=286`
- Delay4 therefore does not appear to over-suppress true seg04 recovery once the area floor allows the small target.
- However, area100 also admits background drift after the target chain: f294/f296/f298/f396/f402/f413/f425/f551 have interpolated IoU `0.000` in the delay4 area100 run, often with very high crop scores.

Area100 regression check on earlier held-out segments:

- seg02 area100 delay4 output: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050_trackletveto_delay4_area100`
- seg02 area100 delay4: `reacquired_frames=9`, `global=2`, `lost_frames=471`
- seg03 area100 delay4 output: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_trackletveto_delay4_area100`
- seg03 area100 delay4: `reacquired_frames=11`, `global=3`, `lost_frames=261`
- Since seg02 default delay4 had `reacquired_frames=0`, lowering the global area floor globally is not safe.

Size-adaptive update:

- implementation: `--reacquire-global-small-min-area 100` plus `--reacquire-global-small-max-distance-px 1200`
- small-area admission rule: `area >= 100` and `area < 1000`, tracklet-confirmed, not sequence-gate rejected, crop gate passes, and distance from predicted memory is within `1200 px`
- seg01 output: `runs\e164_selector_probe\dji_broad_seg01_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive`
- seg01 summary: `reacquired_frames=26`, `global=2`, `small_global=0`, `lost_frames=257`
- seg02 output: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive`
- seg02 summary: `reacquired_frames=0`, `global=0`, `small_global=0`, `lost_frames=562`
- seg03 output: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive`
- seg03 summary: `reacquired_frames=9`, `global=1`, `small_global=0`, `lost_frames=353`
- seg04 output: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_sourcepred`
- seg04 summary: `reacquired_frames=7`, `global=4`, `small_global=4`, `lost_frames=369`
- seg04 sparse annotation check: selected `5/10` annotated frames, IoU >= `0.1` on `4/10`
- seg04 interpolated sanity: small-global f88 and f206 align with the interpolated sparse GT (`IoU=0.440` and `0.495`); f76/f256 are near but below IoU threshold under interpolation.
- seg04 review sheet: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_sourcepred\seg04_sizeadaptive_reacquire_zoom_sheet.jpg`

Updated decision:

Pre-probation decision: `cropdense_min050_trackletveto_delay4_sizeadaptive` kept the seg02 false-reacquire block from delay4, kept the seg03 late recovery, and recovered several seg04 small-target global seeds without globally lowering `--reacquire-global-min-area`. That run exposed downstream memory-near drift after a small global seed, which is addressed by the probation48 update below.

## Small-Global Memory Probation Update

Implementation:

- added `--reacquire-global-small-memory-probation-frames 48` to the default probe script.
- after a confirmed small-area global seed, memory-near/deep REACQUIRE is blocked through the probation window.
- memory RECOVER keeps the normal `max_recover_frames` grace immediately after the seed, then is blocked for the rest of the probation window so predicted boxes are not repeatedly written back into memory.
- new summary counters: `reacquire_memory_probation_rejected_frames` and `memory_recover_probation_rejected_frames`.

Outputs:

- seg01: `runs\e164_selector_probe\dji_broad_seg01_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48`
- seg02: `runs\e164_selector_probe\dji_broad_seg02_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48`
- seg03: `runs\e164_selector_probe\dji_broad_seg03_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48`
- seg04: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48_sourcepred`
- seg04 annotated video: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48_sourcepred\selector_annotated.mp4`

Replay summaries:

| Segment | Accepted | Reacquired | Memory | Global | Small global | Memory REACQ blocked | RECOVER blocked | Recover | Lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seg01 | 237 | 26 | 24 | 2 | 0 | 0 | 0 | 47 | 257 |
| seg02 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 562 |
| seg03 | 163 | 9 | 8 | 1 | 0 | 0 | 0 | 24 | 353 |
| seg04 | 183 | 6 | 0 | 6 | 6 | 152 | 12 | 22 | 365 |

Seg04 behavior:

- f256 small-area global seed is kept.
- f257-f258 RECOVER grace is kept, so f259-f292 still TRACK.
- f293-f302 are now LOST; the previous f294/f296/f298 memory-near REACQUIRE drift is gone.
- f304 is still accepted as a new `confirmed_global_small_area_candidate_after_stale_lost`, so the remaining failure mode is now repeated small-area global seed admission, not memory-near drift.

Seg04 sparse checks:

- exact sparse annotations: selected `5/10`, IoU >= `0.1` on `4/10`.
- interpolated true-ish small global hits: f88 `IoU=0.440`, f206 `IoU=0.495`.
- remaining interpolated misses: f76 `IoU=0.000`, f256 `IoU=0.000`, f304 `IoU=0.000`, f397 `IoU=0.000`.

Updated decision:

Promote `cropdense_min050_trackletveto_delay4_sizeadaptive_probation48` as the current E164 default probe. It preserves seg01, seg02, and seg03 behavior from sizeadaptive, preserves the seg04 f259-f292 continuation, and removes the f294-f298 memory-near drift chain. The next highest-priority experiment is a repeated-small-global admission guard, focused on f304/f397 without killing f88/f206.

## Seed Admission Audit Dataset

Implementation:

- every `offline-selector-replay` run now writes `reacquire_seed_audit.csv`.
- the audit records global seed admission features needed for a later learned admission head: bbox area, detector score, selector score, motion/memory consistency, jump/distance, crop score, appearance score, tracklet fields, sequence-gate fields, small-global repeat state, and final pending/confirmed/rejected event.
- the selector summary now includes `reacquire_seed_audit_csv` and `reacquire_seed_audit_rows`.
- `build-seed-admission-dataset` converts the audit rows into supervised training rows by aligning each seed bbox to exact or linearly interpolated sparse GT and adding `seed_label`, `label_reason`, `gt_iou`, `gt_center_distance_px`, and `sample_weight`.
- rows outside GT coverage are ignored with `sample_weight=0` instead of being treated as background negatives.

This turns f304/f397-style failures into supervised seed-admission rows instead of only hard-coded threshold anecdotes.

Seg04 repeat48d120 labeled export:

- command: `python -m qstr_dronedet.cli build-seed-admission-dataset --seed-audit runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48_repeat48d120_sourcepred\reacquire_seed_audit.csv --annotations D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\annotations\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg04_005971_006569_boxes.csv --out runs\e164_selector_probe\seed_admission_datasets\seg04_repeat48d120_seed_admission.csv --dataset-source dji_broad_seg04 --run-id cropdense_min050_trackletveto_delay4_sizeadaptive_probation48_repeat48d120`
- output: `runs\e164_selector_probe\seed_admission_datasets\seg04_repeat48d120_seed_admission.csv`
- summary: `total_rows=22`, `labeled_rows=22`, `positive_rows=4`, `negative_rows=18`, `ignored_rows=0`
- positives: f87/f88 and f205/f206.
- hard negatives include f303/f304, f306, and f397; f256 is also labeled negative under sparse linear interpolation and should be reviewed visually before using as a hard negative.

Default probation48 labeled export:

- audit source outputs:
  - `runs\e164_selector_probe\seed_admission_audit_sources\seg01_probation48`
  - `runs\e164_selector_probe\seed_admission_audit_sources\seg02_probation48`
  - `runs\e164_selector_probe\seed_admission_audit_sources\seg03_probation48`
  - `runs\e164_selector_probe\seed_admission_audit_sources\seg04_probation48_sourcepred`
- labeled CSV outputs:
  - `runs\e164_selector_probe\seed_admission_datasets\seg01_probation48_seed_admission.csv`: `12` rows, `0` positive, `12` negative
  - `runs\e164_selector_probe\seed_admission_datasets\seg02_probation48_seed_admission.csv`: `2` rows, `0` positive, `2` negative
  - `runs\e164_selector_probe\seed_admission_datasets\seg03_probation48_seed_admission.csv`: `8` rows, `0` positive, `8` negative
  - `runs\e164_selector_probe\seed_admission_datasets\seg04_probation48_seed_admission.csv`: `20` rows, `4` positive, `16` negative
- merged CSV: `runs\e164_selector_probe\seed_admission_datasets\seg01_04_probation48_seed_admission.csv`
- merged summary: `42` total rows, `4` positive, `38` negative, `0` ignored.
- training caution: positives currently come only from seg04 under sparse GT interpolation. Before training a default admission head, add more true-reacquire positives or manually review ambiguous negatives to avoid a segment-specific classifier.

Seed-admission classifier smoke:

- implementation: `qstr_dronedet.tracking.seed_admission_classifier`
- Modal/local wrapper: `tools\modal_seed_admission_train.py`
- local smoke command: `python tools\modal_seed_admission_train.py --dataset runs\e164_selector_probe\seed_admission_datasets\seg01_04_probation48_seed_admission.csv --out runs\e164_selector_probe\seed_admission_smoke\local_seg01_04_probation48 --epochs 5 --hidden 16 --thresholds 0.25 0.5 0.75 --smoke`
- local smoke outputs:
  - weights: `runs\e164_selector_probe\seed_admission_smoke\local_seg01_04_probation48\seed_admission_classifier.pt`
  - metrics: `runs\e164_selector_probe\seed_admission_smoke\local_seg01_04_probation48\metrics.json`
  - threshold sweep: `runs\e164_selector_probe\seed_admission_smoke\local_seg01_04_probation48\threshold_sweep.csv`
- local smoke result on the current imbalanced dataset: best threshold `0.25`, `tp=4`, `fp=38`, `fn=0`, `tn=0`, precision `0.095`, recall `1.000`, F1 `0.174`.
- interpretation: the trainer I/O works, but the dataset is not ready for a deployable admission model. The next data step is still to add more positive REACQUIRE segments and audit ambiguous negatives.
- Modal entrypoint check: `modal run tools\modal_seed_admission_train.py --help` lists `--dataset`, `--out`, `--epochs`, `--lr`, `--hidden`, `--thresholds`, and `--smoke`.

Held-out seg05 seed-admission refresh:

- selector output: `runs\e164_selector_probe\seed_admission_audit_sources\seg05_probation48_sourcepred`
- labeled CSV: `runs\e164_selector_probe\seed_admission_datasets\seg05_probation48_seed_admission.csv`
- summary: `2` total rows, `2` positive, `0` negative, `0` ignored.
- positive rows: f212/f213, both center-match positives with GT center distance about `16 px`, crop scores about `0.994`, and tracklet classifier probability about `0.798`.
- merged CSV: `runs\e164_selector_probe\seed_admission_datasets\seg01_05_probation48_seed_admission.csv`
- merged summary: `44` total rows, `6` positive, `38` negative, `0` ignored.
- seg01-05 local smoke output: `runs\e164_selector_probe\seed_admission_smoke\local_seg01_05_probation48`
- seg01-05 smoke result: best threshold `0.50`, `tp=6`, `fp=34`, `fn=0`, `tn=4`, precision `0.150`, recall `1.000`, F1 `0.261`.
- interpretation: seg05 adds useful positive examples and improves the smoke metric, but false positives remain too high. Continue mining additional true REACQUIRE positives and hard negatives before promoting any learned admission model.

## Candidate Repeated-Small-Global Guard

Implementation:

- added optional CLI args:
  - `--reacquire-global-small-repeat-cooldown-frames`
  - `--reacquire-global-small-repeat-max-distance-px`
- when enabled, a small-area global seed inside the cooldown is rejected if its center is too far from the last confirmed small-area global seed.
- default remains disabled; this is an experiment guard, not the current default probe.

Seg04 candidate run:

- output: `runs\e164_selector_probe\dji_broad_seg04_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48_repeat48d120_sourcepred`
- settings: repeat cooldown `48`, repeat max distance `120 px`
- summary: `reacquired_frames=6`, `global=6`, `small_global=6`, `repeat_rejected=6`, `seed_audit_rows=22`, `lost_frames=367`
- exact sparse annotations: selected `5/10`, IoU >= `0.1` on `4/10`
- f303/f304 are rejected by `reacquire_global_small_repeat_inconsistent`.
- f257-f258 RECOVER grace and f259-f292 TRACK are preserved.
- f306 and f397 still become small-area global REACQUIRE; therefore repeat48d120 is useful diagnostic evidence but not sufficient as a default guard.

Updated next step:

Do not promote the repeated-small-global rule by itself. Export seed-admission datasets for seg01-04 with the same command shape, visually audit ambiguous negative rows such as seg04 f256, then train or fit a small seed-admission head/logistic model on Modal and compare it against `probation48` and `repeat48d120` on held-out segments.
