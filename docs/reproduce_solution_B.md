# Reproducing solution B: retrained matcher + retrained covariance head (the recommended frontend)

B is the frontend we recommend: the FlowFormer matcher fine-tuned on EIVA, and the covariance head retrained on top of
that matcher. Deliverable: `Model/MACVO_FrontendCov_EIVA_B_MatcherHead.pth` (+ `_fast.onnx`, `_fast_o5.plan`).
Background and every experiment behind it: `docs/covhead_retrain_summary.md` (map) and `docs/covhead_retrain_handoff.md`
(log). Machine rules (one job per GPU, data on D:) and the launcher: `CLAUDE.md`.

Why B over A (head only, original matcher), held out by site, ATE m / frame-to-frame mm at the shipped VO settings:

| | Vobster (fold W) | Welland (fold V) | TartanAir seaside ATE / RTE / ROE | flow error vs shipped |
|---|---|---|---|---|
| shipped | 0.051 / 3.73 | 0.119 / 1.53 | 1.354 / 0.0120 / 0.0625 | |
| A | **0.029** / 2.57 | 0.109 / 1.12 | 1.144 / 0.0112 / **0.0537** | unchanged (same matcher) |
| **B** | 0.035 / **2.44** | **0.083 / 1.06** | **1.109 / 0.0110** / 0.0593 | -20% temporal, -36% stereo (TartanAir); -26% stereo (held-out EIVA) |

B has the more accurate matches and depth and the better frame-to-frame accuracy everywhere. Its weak spots:
very large motion (the sharp turn `plane_tail1` #331) and TartanAir `seaside_E003`; A stays the fallback.

## The recipe in one picture

```
shipped MACVO_FrontendCov.pth
  -- CovHeadChol (cov mode, TartanAir, 30k-step schedule, keep step 25000) -->  Model/MACVO_FrontendCov_Chol25k_base.pth
  -- EIVA_RefinedMatcher (flow mode, 20k steps)                            -->  B matcher stage (20000.pth)
       matcher decoder + cost layers trained, image encoders and head frozen, fp32, batch 2, lr 2.5e-5;
       TartanAir v1 + 7 seaside sequences x10 + the EVEN blocks of all 6 EIVA dives with frame gaps 1-16,
       the image gate and a far stereo target on open water (water_disparity 0.25)
  -- EIVA_RefinedHead (cov mode, 15k steps)                                -->  B (15000.pth)
       head only, matcher frozen, in the deployed precision (fp16 encoder / bf16 decoder, head fp32), batch 6,
       lr 6e-5, stereo steps, open-water negatives (water_sigma 32); TartanAir + seaside + the ODD blocks of all
       6 EIVA dives (frames the matcher never saw: cross-fitting) with frame gaps and the image gate
```

All settings live in the two configs, each with a header listing its settings and why:
`Config/Train/EIVA_RefinedMatcher.yaml` and `Config/Train/EIVA_RefinedHead.yaml` (formerly `FlowFTEIVA_AllA_water_gaps`
and `CovHeadEIVA_AllB_v2neg_gaps_prod`, the names in the handoff log and the training records). They are the settings B
was trained with (commit ee35095; parsed, they differ only in `name`, which names the output folder). One
training step of each stage with today's code matches the training-time code to rounding (loss within 0.3%, gradient
cosine >= 0.9999), so a retrain is equivalent, not bit-identical (GPU training is not deterministic).

## 0. Prerequisites

- Data (see `CLAUDE.md`): the six dives under `D:/Datasets/EIVA/...` with their GT in `D:/Datasets/EIVA_GT_v2/<dive>`
  (`index.json` carries the gates), TartanAir v1 zips in `D:/Datasets/TartanAirV1`.
- `Model/MACVO_FrontendCov_Chol25k_base.pth` (the start of the matcher stage) and `Model/MACVO_FrontendCov.pth`
  (shipped, the baseline).

Starting from MAC-VO's released weights only (no base at hand): the chain starts there anyway. `CovHeadChol` restores
`Model/MACVO_FrontendCov.pth` and trains only the head (the old 2-channel log-sigma head becomes the 3-channel Cholesky
head, its new channel starting at 0); its matcher stays the shipped one bit for bit (all 403 matcher tensors of the
base equal the release's). About 4.5 h to step 25000 on one RTX 4090; keep `25000.pth` (the run goes on to 30000,
which is worse in the tail) and point `EIVA_RefinedMatcher`'s `restore_ckpt` at it:

    Scripts/launch.sh 0 chol_base python -u -m Train.MatchingNet.train_flowformer \
        --config Config/Train/CovHeadChol.yaml --training_mode cov
    -> Model/FlowFormerCovChol<MM-DD-HH-MM-SS>/25000.pth

Don't skip this stage. The matcher stage itself would not need it (flow mode trains on the flow loss alone, with the
head frozen), but the matcher checkpoint carries the head into the head stage, and that stage (15k steps, lr 6e-5) is
tuned as a fine-tune of this trained Cholesky head; starting it from the release's 2-channel head is untested.
- Two free GPUs or one; every job through `Scripts/launch.sh <gpu> <name> <command>`, one job per GPU.

New or changed dive? Give it GT first (`make_eiva_gt` -> `check_eiva_gt` -> `check_eiva_photo`, summary section 4),
add it to `DIVES` in `Scripts/Dataset/make_eiva_lists.py` and run `python -m Scripts.Dataset.make_eiva_lists`, which
rewrites both B lists (`EIVA_All_TrainA_v2_water_gaps`, `EIVA_All_TrainB_v2_gaps`). Unchanged, it reproduces them
byte for byte.

## 1. Matcher stage (about 2 h on one RTX 4090)

    Scripts/launch.sh 0 refined_matcher python -u -m Train.MatchingNet.train_flowformer \
        --config Config/Train/EIVA_RefinedMatcher.yaml --training_mode flow

`--training_mode flow` is required: the trainer defaults to `cov` and would train the head instead. The log should
say `Flow training (decoder+cost)`. Output: `Model/EIVARefinedMatcher<MM-DD-HH-MM-SS>/20000.pth`.

Check before going on (CPU is fine for this):

    python -m Scripts.Dataset.check_eiva_water --seq D:/Datasets/EIVA/vobster_quay/plane_tail1 \
        --gt D:/Datasets/EIVA_GT_v2/plane_tail1 --frames 240 251 252 --ckpt Model/EIVARefinedMatcher<time>/20000.pth

The production matcher saw every dive, so its open water should read far: B's was 0.2-0.5 px on `plane_tail1` and
0.3-1.8 px on Welland (frames 1200 2400 4400). A matcher that reports open water as near (tens of px) breaks
trajectories; the held-out version of this check is in step 4.

## 2. Head stage (about 2 h)

Point the head at the new matcher: in `Config/Train/EIVA_RefinedHead.yaml` set

    restore_ckpt: "Model/EIVARefinedMatcher<time>/20000.pth"

(as committed it points at `Model/MACVO_FrontendCov_EIVA_B_MatcherStage.pth`, B's own matcher stage: that retrains
only the head). Then

    Scripts/launch.sh 0 refined_head python -u -m Train.MatchingNet.train_flowformer \
        --config Config/Train/EIVA_RefinedHead.yaml --training_mode cov

Output: `Model/EIVARefinedHead<MM-DD-HH-MM-SS>/15000.pth`, the new B.

## 3. Evaluate against the reference numbers

Network level, the 600 held-out seaside pairs in the deployed precision (pass shipped and the old B for comparison):

    python -m Train.MatchingNet.evaluate --config Config/Train/CovHeadCholWater.yaml --precision fast --pairs 600 \
        --ckpt Model/MACVO_FrontendCov.pth Model/MACVO_FrontendCov_EIVA_B_MatcherHead.pth Model/EIVARefinedHead<time>/15000.pth

Reference (B): temporal EPE 1.237 px (shipped 1.550), stereo EPE 1.028 (1.601), calibration median 1.16 / 1.08,
stereo NLL 4.18 (shipped 13.5).

MAC-VO: copy `Config/Experiment/MACVO/EIVA/MACVO_EIVA_Fast_B.yaml` to a new name and change its `weight:` to the new
checkpoint (and the `name:`), then run it, the shipped config and B's on the EIVA dives and the 8 held-out seaside
sequences (one model per GPU at a time):

    Scripts/launch.sh 1 eval_newB Scripts/Eval/run_macvo.sh Config/Experiment/MACVO/EIVA/<new B>.yaml runs/macvo/newB \
        Config/Sequence/EIVA/*.yaml Config/Sequence/TartanAirV1_Seaside/*.yaml
    python -m Scripts.Eval.eiva_table runs/macvo/shipped runs/macvo/B runs/macvo/newB
    python -m Scripts.Eval.tartanair_table runs/macvo/shipped runs/macvo/B runs/macvo/newB

Reference, split protocol at 0.25 px (EIVA is in-sample for the production model: other frames of the same dives;
TartanAir seaside is held out):

| ATE m / f2f mm | shipped | B |
|---|---|---|
| plane_mid | 0.018 / 2.35 | 0.009 / 1.33 |
| plane_nose | 0.078 / 5.52 | 0.042 / 3.84 |
| plane_tail1 | 0.060 / 2.92 | 0.071 / 3.44 |
| plane_tail2 | 0.048 / 4.12 | 0.032 / 3.03 |
| welland | 0.119 / 1.53 | 0.073 / 0.81 |
| seaside (8) ATE / RTE / ROE | 1.354 / 0.0120 / 0.0625 | 1.109 / 0.0110 / 0.0593 |

MAC-VO varies about +-7% in ATE and +-3% in frame-to-frame from run to run (single runs above), so judge the means and
repeat a run (`runs/macvo/newB_r2`, averaged by `tartanair_table`) before reading anything into a ~10% difference on
one sequence. Known steps every model takes: `plane_tail1` #331 (the sharp turn; B 135 mm) and `plane_nose` #545 (a
diver's bubbles; B 247 mm). A check of today's scripts reproduced B at plane_mid 0.008 / 1.32 and E003 1.162.

## 4. Held-out evidence (the site folds)

The production model saw every dive, so its EIVA numbers are in-sample. A recipe change needs the folds: the same two
stages trained on one site and tested on the other. Fold head configs restore their fold matcher from a run folder:
set their `restore_ckpt` to the new fold matcher's `20000.pth` as in step 2.

| fold | matcher (flow mode) | head (cov mode) | test | reference (shipped) |
|---|---|---|---|---|
| W | `FlowFTEIVA_WellandA_water_gaps` | `CovHeadEIVA_WellandBWater_v2_gaps_crossfit` | 4 Vobster dives | 0.035 / 2.44 (0.051 / 3.73); TartanAir ATE 1.103 |
| V | `FlowFTEIVA_VobsterA_water_gaps` | `CovHeadEIVA_VobsterBWater_v2neg_gaps_crossfit` | Welland | 0.083 / 1.06 (0.119 / 1.53) |

Fold W per dive: plane_mid 0.009 / 1.32, plane_nose 0.026 / 2.33, plane_tail1 0.076 / 3.00, plane_tail2 0.029 / 3.13.
Water on the unseen site (`check_eiva_water` with the fold matcher): fold V on Welland 6.3 px (far); fold W on
`plane_tail1` #240 / #251 / #252 reads 57 / 45 / 24 px, nearer than the original matcher (10-17 px), but its head
flags that water (stereo sigma p10 / median 20.6 / 33.6 px on #251), so MAC-VO's selector skips it. Without the far-water
label the fold W matcher read that water at 85-115 px and trajectories broke. Large motion, fold W on held-out
`plane_tail1` #331 (256 px): EPE 25.3 px (original matcher 5.2, gap-free fine-tune 69.6).

## 5. Deliverables

1. Copy the new `15000.pth` to `Model/MACVO_FrontendCov_EIVA_B_MatcherHead.pth` (keep the old one under another name
   until the new one is accepted) and its training records (`runs/<name>_*`) to `Model/MACVO_FrontendCov_EIVA_training_logs/`.
2. Drop-in ONNX (static batch 1, opset 18, fp16 / bf16, the shipped 2-channel log-sigma output):

       python Export.py --mode fast --ckpt Model/MACVO_FrontendCov_EIVA_B_MatcherHead.pth --cov-layout log-sigma \
           --onnx Model/MACVO_FrontendCov_EIVA_B_MatcherHead_fast.onnx

3. Engine with the C++ builder's defaults (static batch 1, weight streaming, opt level 5, avg timing 8); the existing
   timing cache seeds the same tactics on the same GPU and TensorRT version (A's was seeded from B's, a full hit):

       MACVO_TRT_BuildCpp/build/Release/macvo_trt_build.exe --onnx Model/MACVO_FrontendCov_EIVA_B_MatcherHead_fast.onnx \
           --out Model/MACVO_FrontendCov_EIVA_B_MatcherHead_fast_o5.plan [--timing-cache <cache> --error-on-timing-cache-miss]

   (`python Build.py --onnx ... --out ...` does the same with the Python tensorrt package.)
4. Parity of the engine against PyTorch fp32:

       python Scripts/Parity/make_reference.py --ckpt Model/MACVO_FrontendCov_EIVA_B_MatcherHead.pth --out runs/ref_B
       python Scripts/Parity/compare.py --trt Model/MACVO_FrontendCov_EIVA_B_MatcherHead_fast_o5.plan --ref runs/ref_B

5. The C++ runtime: MACVO_TRT `macvo.exe` on the dives (`../MACVO_TRT/docs/tensorrt-engine-study.md`, "Final
   artifact"; B's static engine: 7-dive mean ATE 0.078).
6. Update `Model/MACVO_FrontendCov_EIVA_README.txt` (sources, numbers, SHA-256) and back up `Model/`.
