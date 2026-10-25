# MAC-VO frontend retraining: what was tried, what worked (summary, 2026-10-25)

The consolidated outcome of branch `covhead-retrain` (~110 commits). The full chronological record, with every number
and its context, is `docs/covhead_retrain_handoff.md`; this document is the map for deciding what to keep.

## 0. Ground rules the results are judged by

- **A plugin upgrade, not a VO change.** Only the network (FlowFormerCov frontend: matcher + covariance head) changes.
  MAC-VO's main flow stays as shipped, so improvements are measured against the pure upstream VO.
- **Primary setting: the shipped VO settings** (`min_flow_cov` 0.25 px). Other floors are reported separately.
- **Held out by site.** Fold W trains on Welland and is tested on the four Vobster dives; fold V the other way round.
  TartanAir `seasidetown` (8 sequences) is held out from every training.
- **Split protocol** for EIVA trajectories (`Scripts/Dataset/eval_eiva_trajectory.py`): each dive is split where the
  GT itself breaks (GT check > 3 px, missing poses) and every segment is aligned on its own. The old whole-dive
  numbers were dominated by a few broken GT frames.
- **Deployed precision** (fp16 encoder / bf16 decoder, 704x704 INTER_AREA) for every MAC-VO number.

## 1. Outcome: two working solutions

| ATE m / frame-to-frame mm, 0.25 px, split | Vobster (mean of 4) | Welland | TartanAir seaside ATE m / RTE / ROE |
|---|---|---|---|
| shipped | 0.051 / 3.73 | 0.119 / 1.53 | 1.354 / 0.0120 / 0.0625 |
| **A. head only**: negatives head on the original matcher | **0.029** / 2.57 | 0.109 / 1.12 | 1.144 / 0.0112 / **0.0537** |
| **B. matcher + head**: gaps + gate recipe | 0.035 / **2.44** | **0.083 / 1.06** | **1.109 / 0.0110** / 0.0593 |

Vobster / Welland columns are the held-out site folds of each recipe; TartanAir is the production model of each
recipe (it never saw seaside). Against shipped:

- **A** (head only): Vobster -43% ATE / -31% frame-to-frame, Welland -8% / -27%, TartanAir -16% ATE.
  Engine: `Model/FlowFormerCovEIVAAllWaterV2Neg10-24-00-54-50/MACVO_FrontendCov_EIVAAllWaterV2Neg_fast_o5.plan`.
- **B** (matcher fine-tune + head): Vobster -31% / -35%, Welland -30% / -31%, TartanAir -18% ATE; flow error
  -20% (temporal) / -36% (stereo) on TartanAir, -26% stereo on held-out EIVA.
  Engine: `Model/FlowFormerCovEIVAAllBV2NegGapsProd10-25-02-52-11/MACVO_FrontendCov_EIVAProd_gaps_fast_o5.plan`.

Both engines are drop-in: the same inputs, outputs and 2-channel log-sigma layout as the shipped engine
(`Export.py --cov-layout log-sigma`), +3% time (58 ms against 56 ms at batch 2, RTX 4090), parity with PyTorch fp32
0.010-0.019 px on confident pixels. B is better on Welland and in frame-to-frame accuracy everywhere and has the more
accurate depth; A is better in Vobster ATE and TartanAir rotation, and is far simpler (no matcher training, no frame
gaps, no far-water label). **Choosing between them decides how much code the minimal version keeps (section 5).**

## 2. What worked

### Covariance head

| change | what it does | evidence | kept in |
|---|---|---|---|
| log-space NLL | the shipped `cov_loss` treated log sigma as a variance and was NaN on 82% of terms; fixed to `err^2 exp(-2c) + 2c` | the shipped trainer could not have produced the shipped weights; every later head trains with the fix | A, B |
| Cholesky head (`cov_channels: 3`) | full 2x2 covariance; exported as per-axis marginal log sigma | better NLL / ranking; reproduces the shipped head exactly at init. sigma_uv itself adds nothing to trajectories (dropping it: 0.103 vs 0.111 m Vobster), so the drop-in export loses nothing | A, B |
| stereo supervision (`stereo: True`) | alternate stereo (L->R, target `-fx b / depth`) and temporal steps | the only supervision of the disparity sigma; the 25k head (log-space NLL + Cholesky + stereo) took held-out TartanAir stereo NLL 16.5 -> 12.0 and its tail 7.4% -> 5.9%, while temporal stayed level | A, B |
| deployment-precision training (`encoder_dtype: fp16`, `decoder_dtype: bf16`) | the frozen network runs as deployed while the head trains | the fp32-trained head breaks `plane_tail1` in deployed precision (2.32 m); the deploy-trained one does not | A, B |
| EIVA in the head's data (~25% of pairs) | in-domain calibration | held-out Vobster -16..-24% ATE vs shipped; stereo tail 11% -> 3-7% | A, B |
| seaside oversampling (`TartanAirV1_Seaside_Train`) | TartanAir's water / sky next to the EIVA data | removes the EIVA heads' `Hard_P003` failure (2.57 -> 0.87 m) | A, B |
| open-water negatives (`water_sigma: 32`) | hinge pushing sigma above 32 px on open water (no GT within 24 px) | Vobster -17% ATE / -13% frame-to-frame over the same head without; water sigma p10 4 -> 34 px; nothing lost elsewhere. Also what makes B's open-water stereo harmless | A, B |
| cross-fitting (`frame_range`) | matcher on one half of each site's blocks, head on the other | a head trained on the matcher's own frames learns its training-site errors (~1.7x too small sigma^2). On its own the cross-fit did not lower the unseen-site stereo tail (12.1% vs 12-13%); with gaps + gate it is 8.8%. Kept as the sound way to train the head | B |

### Ground truth (EIVA)

| change | what it does | evidence | kept in |
|---|---|---|---|
| `make_eiva_gt.py` (v2) | depth + flow rendered from the 3DF Zephyr cloud and poses at 704x704; statistical outlier removal; 3x3 hole fill | coverage 41-50% -> 54-71%; agrees with FlowFormer to 0.1-0.3 px median | A, B |
| `check_eiva_gt.py` gates | per-frame disparity ratio and per-pair flow error from the frozen FlowFormer; frames / pairs beyond 2.5% / 0.75 px left unsupervised | removes wrong-pose pairs (e.g. `plane_tail2` pair 505, 58 px) | A, B |
| `check_eiva_photo.py` gate (`photo_rel: 0.75`, `photo_floor: 0.5`) | share of GT pixels the images confirm (ZNCC), per frame and pair | finds pose errors the FlowFormer check misses (11% of Welland's stereo GT, 21% of welland_rear's) | B (A was trained before it existed; retraining A with it is an option) |

### Matcher ("eyes")

| change | what it does | evidence | kept in |
|---|---|---|---|
| fine-tune decoder + cost encoder (`--training_mode flow`, `flow_trainable: decoder+cost`) | 7.65 M parameters, image encoders frozen, fp32, 20k steps | flow error -7..-11% temporal, -22..-27% stereo on the held-out site; -20..-22% temporal, -33..-47% stereo on held-out TartanAir | B |
| far-water stereo label (`water_disparity: 0.25`) | open water gets a far stereo target (the synthetic-sky convention) | without it the fine-tune hallucinates near surfaces in unseen water (plane_tail1 open water 14 -> 103 px) and trajectories break; with it 11-20 px, and it generalises to the other site | B |
| frame gaps (`frame_stride` 2-16, half one-frame pairs) | larger displacements from the same footage; flow from GT depth + poses, visibility-checked | the gap-free fine-tune lost large motion (plane_tail1 #331: 70 px vs 5 px) and broke that dive (2.32 m); with gaps 0.076 m, the best frame-to-frame on Vobster and the best result on Welland | B |

### Evaluation and tooling

- `Train/MatchingNet/evaluate.py` rewritten: temporal + stereo, NLL / calibration / AUSE / tails, every checkpoint on
  the same pairs and pixels.
- `Scripts/Dataset/eval_eiva_trajectory.py`: the split protocol.
- `Scripts/Dataset/check_eiva_water.py`: open-water disparity and sigma, the release check for any matcher fine-tune.
- TartanAir v1 read in place from the official zips (`DataLoader/Dataset/Zipped.py`), with the flow-mask polarity fix.
- `Export.py --ckpt --cov-layout log-sigma` and the parity harness for 3-channel heads.
- `train_flowformer.py` fixes: `ConcatDataset` of `SequenceBase`, `DataParallel` with `FusedGRUPair`, the
  `OneCycleLR` overrun.

## 3. What did not work

| tried | why | result | verdict |
|---|---|---|---|
| beta-NLL (beta 0.5) | robustness to hard points | with a frozen flow it only reweights pixels towards the uncertain ones the selector discards | not used (`beta: 0`) |
| masking the covariance loss on occluded / out-of-view pixels | TartanVO excluded them | those are exactly where sigma must be large | not used (`cov_mask: False`) |
| single-shot head (after the decoder loop) | cheaper branch (50.8 vs 56.1 ms) | TartanAir ATE 1.43 vs 1.37 for the recurrent head (5 sequences, 2 repeats); sigma moves ~20% under reduced precision | dropped |
| inlier probability / mixture likelihood (single-shot and recurrent) | heavy-tailed errors | more pixels beyond the 99% bound than the plain head; level on Vobster, worse on Welland | dropped |
| correlation-peak cues + flow update as head input | matching-confidence features | no gain over a control with the same schedule | dropped |
| longer head training (30k, +20k) | | 30k worse in the tail than 25k; +20k adds nothing | 25k / 15k kept |
| sigma_uv in VO (`cov_offdiag`, `flow_score: trace`) | use the full covariance downstream | level (0.103 / 0.105 / 0.111 m) | not needed: the export gives marginals |
| global sigma scale (`cov_scale` x2, x4) | wider sigma for hard stretches | no help (1.18 / 1.33 vs 1.11 m) | diagnostic only |
| GT: front-surface averaging, oriented surfel splatting, dropping filled pixels | cleaner GT | not better than v2 (stereo 0.087-0.115 vs 0.070 px; surfels worse on noisy normals) | v2 kept |
| teacher targets (old network's flow / depth where GT is missing) | keep the original behaviour in open water | keeps the training site's water only, not an unseen site's (85-115 px); costs transfer gains (fold V stereo -22% -> -6%) | dropped |
| underwater image augmentation of TartanAir (Akkaynak & Treibitz model) | generalise to other water | with the teacher: open water 38 / 101 / 37 px, fails the water check | dropped |
| plain matcher fine-tune (no water label) | better matching | breaks water-heavy dives (open-water hallucination) | superseded by the far-water label |
| mixed split (part of each site in training) | generalisation | water fine, but the gain came from the head's extra data, not the matcher; the mixed cross-fit (with teacher) was worse than the mixed head-only baseline | dropped |
| synthetic camera turns (`rotation_aug`) | the sharp-turn pair that gaps leave (#331) | fixes it (EPE 6.8 px) and beats the original matcher at every motion size, but the production model with it is worse than shipped on TartanAir (ATE 1.43, `Hard_P003` / `Hard_P000`) | experimental, not in B |
| voxel-hash cloud cleaning (`Scripts/Dataset/voxel_clean.py`) | OREC_CPS's GT looked noisy | no change: OREC's errors are poses next to unregistered images, not cloud noise | tool only |
| OREC_CPS dive | a third site | pose errors (18-40% image-confirmed GT in stretches) | parked |

Outside the model, for the record: the VO's flow-covariance floor (`min_flow_cov`) is the largest single gain of all
(shipped head at 0.10 px: Welland -33% ATE). It is a runtime setting, so it is reported but not part of either
solution.

## 4. The recipes, end to end

**Data, both solutions.** For each dive with `pose_gt.txt` + `pointcloud_gt.txt`:
`make_eiva_gt.py` -> `check_eiva_gt.py` -> `check_eiva_photo.py` (writes the gate scores into `index.json`).

**A (head only).** Start from the 25k recurrent Cholesky head (`Model/FlowFormerCovChol10-19-17-52-09/25000.pth`,
itself `CovHeadChol` on TartanAir), then `CovHeadEIVA_AllWater_v2neg`: cov mode, deployed precision, 15k steps,
stereo alternation, TartanAir + seaside + both EIVA sites, `water_sigma: 32`. Export with `--cov-layout log-sigma`.

**B (matcher + head).**
1. Matcher: `FlowFTEIVA_AllA_water_gaps` (flow mode, decoder + cost, fp32, 20k steps, batch 2) on every dive's even
   blocks: far water, frame gaps, image gate, TartanAir + seaside.
2. Head: `CovHeadEIVA_AllB_v2neg_gaps` restored from step 1, on the odd blocks: negatives, gaps, gate.
3. Release checks: `check_eiva_water.py` on a site the matcher has not seen; TartanAir MAC-VO.
4. Export with `--cov-layout log-sigma`.

## 5. Code map for the minimal version

Against `main`, outside the docs and the sequence lists: 86 files, +9.3k lines, most of it the 51 added training
configs; plus 35 added sequence lists.

**Needed by both A and B**
- `Module/Network/FlowFormerCov/covhead.py`: the recurrent Cholesky head (`cov_channels` 2 / 3), `cov_entries`,
  `marginal_log_sigma`; `flownet.py`: 2- or 3-channel inference; `FlowFormer/core/decoder.py` (`upsample_flow` scale
  for log sigma); `FlowFormer/core/gru.py` (`FusedGRUPair` weight casts for deployment-precision training).
- `Train/MatchingNet/loss.py`: log-space Cholesky NLL, `water_hinge`; `utils.py`: `stereo_target`, `water_mask`,
  `IndexedDataset`; `train_flowformer.py`: cov mode, stereo alternation, encoder / decoder dtypes, `water_sigma`, the
  loader fixes.
- `DataLoader/Dataset/EIVA.py`: the loader with the scale / flow-error / photo gates; `TartanAir.py` + `Zipped.py`.
- `Scripts/Dataset/make_eiva_gt.py`, `check_eiva_gt.py`, `check_eiva_photo.py`; `Export.py --cov-layout`.

**Needed by B only**
- `train_flowformer.py`: flow mode with `flow_trainable`; `EIVA.py`: `water_disparity`, `frame_range`,
  `frame_stride` / `frame_offset` (gap flow); `Scripts/Dataset/check_eiva_water.py`.

**Evaluation and tooling (keep, not needed to train or run)**
- `Train/MatchingNet/evaluate.py`, `Scripts/Dataset/eval_eiva_trajectory.py`, `Scripts/Parity/*`,
  `Scripts/Dataset/download_tartanair_v1.py`, `make_tartanair_v1_seqs.py`; `Scripts/AdHoc/CovLossCheck.py`,
  `StereoTailAnalysis.py` (diagnostics; candidates to drop).

**Experiment-only: candidates for removal**
- `covhead.py`: `SingleShotCovHead`, the inlier head (`inlier_cholesky`, `cov_channels: 4`), the cues
  (`correlation_peak_stats`, `cov_cues`); `gru.py`'s single-GRU path (`gru_b = None`) that only the single-shot head
  uses; `loss.py`: `cov_mixture_nll`, `beta`, `cov_mask`, `finalcov`.
- `EIVA.py`: `teacher`, `use_filled`; `Scripts/Dataset/make_eiva_teacher.py`.
- `Train/MatchingNet/underwater.py` + `underwater_aug`; `Train/MatchingNet/rotation.py` + `rotation_aug` (or keep
  as experimental for the sharp-turn follow-up).
- `Scripts/Dataset/voxel_clean.py` (tool, no effect on the GT we use).
- VO-side options, all off by default: `cov_scale`, `cov_offdiag` (`Frontend.py`), `flow_score`
  (`KeypointSelector.py`). Removing them returns the VO files to upstream except `Matching.py`'s `from_cov`, which
  only matters if the PyTorch frontend passes 3 channels (see below). `MACVO.py`'s rerun `set_time` change is an
  unrelated compatibility fix.
- `Config/Train`: 51 added configs; keep the recipe configs of the chosen solution (production + its two folds) and
  the base heads they restore from (`CovHeadChol`, `CovHeadCholDeploy`, `CovHeadCholWater`).
  `Config/Sequence/Training_Dataset`: likewise.

**One parity detail to settle.** Every MAC-VO number above came from the PyTorch frontend with the 3-channel output
(sigma_uv used in selection and `Project2to3`). The drop-in engines emit marginals (sigma_uv = 0), which is what the
C++ runtime will see. The `cov_offdiag: false` ablation (marginals only) was level or better on both folds, so this
is expected to hold; making the PyTorch `inference()` return marginal log sigma would also remove the last VO-side
change (`from_cov`) and let one A/B run confirm it.

## 6. Open

1. Choose A or B (or A now, B after more validation), then trim the code to it.
2. The sharp-turn pair (`plane_tail1` #331) is B's weak spot without turns (16.6 px in-sample vs the original's 5.2).
3. Fragile frames: when an occluder (a diver's bubbles, `plane_nose` #544) leaves ~10 keypoints, a few wrong
   disparities next to one-sided occlusion decide the pose. Every model loses a step there.
4. TartanAir seaside E003 regresses with every fine-tuned matcher (1.0-1.2 m vs 0.4-0.55 m).
5. The head's stereo tail on an unseen site (9-13% beyond 9 sigma against 4-5%).
6. No held-out EIVA site is left for either production model; a new dive (or OREC's good stretches) would give one.
