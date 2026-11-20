# MAV-VO-MCH: MAC-VO with the FlowFormerCov frontend retrained for EIVA underwater data

Fork of MAC-VO (`upstream` = MAC-VO/MAC-VO). The work here retrains the frontend network (matcher + covariance head) on
EIVA dives and ships it as a drop-in plugin for MAC-VO and the C++ runtime (`../MACVO_TRT`).

## Read first

- **`docs/reproduce_solution_B.md`: the recommended frontend (B: retrained matcher + retrained head on top of it),
  step by step from training to engine, with the reference numbers for every check.** A (head only on the original
  matcher) is the fallback.
- `docs/covhead_retrain_summary.md`: what was tried, what worked, the two solutions (A, B), the recipes and commands.
- `docs/covhead_retrain_handoff.md`: the full chronological log with every number (long; search it, don't read it whole).
- `Model/MACVO_FrontendCov_EIVA_README.txt`: the deliverables (weights, ONNX, engines), how to rebuild engines.
- `../MACVO_TRT/docs/`: the C++ runtime side (engine study, retrained-engine VO results).

## Branches

- `covhead-minimal` (pushed): the working solutions only. Work here.
- `covhead-retrain` (local only): every experiment, including the rejected ones (single-shot / inlier heads, mixture
  and beta-NLL losses, teacher targets, underwater and camera-turn augmentations, `voxel_clean.py` for OREC).
- `main`: upstream plus the handoff brief.

## Ground rules (from the user)

- **Plugin upgrade, not a VO change.** Improve the network and its training; MAC-VO's main flow stays as upstream.
  Minor API accommodations in the VO are fine, but parity with upstream must hold so gains are measured against the
  pure MAC-VO. Judge at the shipped VO settings (`min_flow_cov` 0.25 px) in the deployed precision (fp16 encoder /
  bf16 decoder, 704x704).
- **Held out by site.** Claims about generalisation need the site folds (fold W: train Welland, test Vobster; fold V:
  the reverse) and TartanAir `seasidetown` (8 sequences never trained on). The production A and B saw every EIVA dive.
- **MAC-VO is not bit-deterministic** (about +-7% ATE, +-3% frame-to-frame on one dive). Differences of ~10% on one
  sequence are noise; repeat runs (`<result dir>_r2`, ...) before claiming them.
- **Data: `D:\Datasets` only. Never modify, move or delete existing data**; new outputs go to new folders. **Keep more
  than 800 GB free on D:** (check `df -h /d` before writing large outputs; it was 699 GB on 2026-11-20 after new
  datasets were added, so new large outputs need the user's OK).
- **One job per GPU, strictly** (see below). Don't co-schedule jobs on a card, even small ones.

## Machine

- 2x RTX 4090 24 GB, Windows 11 (WDDM): a card that runs out of memory pages to host memory and crawls instead of
  failing, so two jobs on one card stall for hours.
- Python 3.11.5 (`C:\Program Files\Python311`, system install, packages from `requirements.txt` + onnx, onnxsim,
  tensorrt 10.13). Set `PYTHONIOENCODING=utf-8` (the console is cp1252). Shell scripts are bash (Git Bash) with LF
  endings (`.gitattributes`).
- TensorRT: `trtexec` 10.13 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\trtexec.exe`; the C++
  builder `MACVO_TRT_BuildCpp/build/Release/macvo_trt_build.exe`; `Build.py` (Python tensorrt).

## Running GPU jobs

Start every GPU job (training, MAC-VO evaluation, GT rendering, export) through `Scripts/launch.sh`:

    Scripts/launch.sh <gpu 0|1> <run name> <command ...>

It sets `CUDA_VISIBLE_DEVICES` to that one card, refuses to start if the card already runs a launch.sh job (lock in
`runs/.gpu<N>.lock`), and writes `runs/<name>_<stamp>.info` (commit, GPU, command, exit code), `.log` and `.err.log`.
Pinning matters beyond politeness: `train_flowformer.py` wraps the model in `nn.DataParallel` over every visible GPU,
and the configs' batch sizes (6 for heads, 2 for the fp32 matcher fine-tune, 21.7 GB peak) assume one card.

To use both cards, run two independent queues: a queue is a bash script that calls `Scripts/launch.sh` for one job
after another on the same GPU; start one queue per GPU in the background (Bash `run_in_background`) and wait for the
completion notification instead of polling. A training run takes about 2 h (15k head steps or 20k matcher steps).
Checkpoints go to `Model/<config name><MM-DD-HH-MM-SS>/<step>.pth`.

## Data

| Path | What |
|---|---|
| `D:/Datasets/EIVA/vobster_quay/plane_{mid,nose,tail1,tail2}` | Vobster site, 4 dives |
| `D:/Datasets/EIVA/welland/{Welland_test_first_dive,welland_rear}` | Welland site (`welland`, `welland_rear`) |
| `.../<dive>/processed/{left,right}`, `pose_gt.txt`, `pointcloud_gt.txt` | rectified 2816x2816 stereo, photogrammetry poses and cloud |
| `D:/Datasets/EIVA_GT_v2/<dive>` | training GT (depth, flow, `index.json` with the scale / flow-error / photo gates) |
| `D:/Datasets/TartanAirV1/<env>/<Easy,Hard>/*.zip` | TartanAir v1, read straight from the zips |
| `D:/Datasets/EIVA_GT`, `EIVA_teacher` | superseded (v1 GT, teacher experiment); leave them |
| `D:/Datasets/EIVA/orec/OREC_CPS` | parked: pose errors next to unregistered images |
| `eiva_pool/mr_anderson`, `LargeAreaScan1`, `Wellhead`, `plane_nose_2026`, `D:/Datasets/EIVA_DEEP_Campus_Nov_2026` | added later, not in any training |

A new dive with `pose_gt.txt` + `pointcloud_gt.txt` gets GT with `make_eiva_gt.py` -> `check_eiva_gt.py` ->
`check_eiva_photo.py` (summary section 4), then a sequence list in `Config/Sequence/Training_Dataset/` (`EIVA_*`)
and a MAC-VO data config in `Config/Sequence/EIVA/`. It is also the chance for a held-out site (summary section 6).

## Training (summary section 4)

- Starting checkpoints (`Model/` is not in git; the user backs it up):
  - `Model/MACVO_FrontendCov.pth`: the shipped frontend (MAC-VO download).
  - `Model/MACVO_FrontendCov_Chol25k_base.pth`: the recurrent Cholesky head on TartanAir, the start of both A and B.
    It is step 25000 of a 30k-step `CovHeadChol` run from the shipped weights (step 30000 was worse in the tail), so
    recreating it means running `CovHeadChol` and keeping `25000.pth`.
  - `Model/MACVO_FrontendCov_EIVA_B_MatcherStage.pth`: B's fine-tuned matcher (`EIVA_RefinedMatcher`), the
    start of B's head. A retrain of B produces a new one: point the head config's `restore_ckpt` at it.
  - The fold head configs restore fold matchers from their run folders in `Model/`; retrain those first if missing.
- A: `Config/Train/CovHeadEIVA_AllWater_v2neg.yaml --training_mode cov` (folds `CovHeadEIVA_{Welland,Vobster}Water_v2neg`).
- B: `EIVA_RefinedMatcher.yaml --training_mode flow`, then `EIVA_RefinedHead.yaml --training_mode cov`
  (folds `FlowFTEIVA_{Welland,Vobster}A_water_gaps` -> `CovHeadEIVA_{WellandBWater_v2_gaps,VobsterBWater_v2neg_gaps}_crossfit`).
- Command: `python -u -m Train.MatchingNet.train_flowformer --config <config> --training_mode <cov|flow>`. The mode
  comes from the command line only and defaults to `cov`: the `FlowFT*` matcher configs need `--training_mode flow`,
  or they silently train the head instead.
- Data: A uses whole dives (`EIVA_All_Train_v2`: 4 Vobster dives x27, Welland x10; no welland_rear, gaps or image
  gate). B's two lists (`EIVA_All_TrainA_v2_water_gaps`, `EIVA_All_TrainB_v2_gaps`: all 6 dives, cross-fit halves,
  gaps, image gate) come from `python -m Scripts.Dataset.make_eiva_lists`; add a new dive to its `DIVES` and rerun.
  The fold lists are fixed files. Both add TartanAir v1 and the 7 training seasidetown sequences x10.
- Provenance: A was trained at commit 48569a6, B at ee35095 (`Model/MACVO_FrontendCov_EIVA_training_logs/`). The
  configs here are those, minus the removed `beta: 0.0` (plain NLL, the code's only mode now) and with `restore_ckpt`
  renamed to byte-identical copies. Training is not bit-deterministic, so a retrain matches in recipe, not in bits.

## Evaluation and release

- Network level: `python -m Train.MatchingNet.evaluate --config Config/Train/CovHeadCholWater.yaml --precision fast --ckpt <ckpt> ...`
  (held-out seaside pairs; a fold config's Evaluate split for EIVA).
- Matcher fine-tunes: `Scripts/Dataset/check_eiva_water.py` on a site the matcher has not seen (open water must stay far).
- MAC-VO: `Scripts/Eval/run_macvo.sh <odom config> <result dir> <data configs ...>` through `Scripts/launch.sh`, with
  `Config/Experiment/MACVO/EIVA/MACVO_EIVA_Fast[_A|_B].yaml` (change `weight:` for a new checkpoint) on
  `Config/Sequence/EIVA/*.yaml` and `Config/Sequence/TartanAirV1_Seaside/*.yaml`; then
  `python -m Scripts.Eval.eiva_table <result dirs>` (split protocol, ATE m / frame-to-frame mm) and
  `python -m Scripts.Eval.tartanair_table <result dirs>` (ATE / RTE / ROE). Compare against shipped run the same way.
- Engine: `python Export.py --mode fast --ckpt <pth> --cov-layout log-sigma --onnx <out>.onnx`, build with the C++
  builder or `Build.py`, parity with `Scripts/Parity/make_reference.py --ckpt <pth> --out runs/ref_x` then
  `Scripts/Parity/compare.py --trt <plan> --ref runs/ref_x`. Name deliverables as in the Model README.

## Known open items

Summary section 6: the PyTorch frontend's marginal output (last VO-side change), B's sharp-turn weakness
(`plane_tail1` #331), fragile frames with few keypoints, TartanAir E003 regression of fine-tuned matchers, the head's
stereo tail on unseen sites, no held-out EIVA site for the production models.
