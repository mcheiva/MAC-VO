# Retraining the MAC-VO covariance head on the FlowFormer backbone: handoff brief

Self-contained brief for an agent working on another machine. It describes what to change in
the covariance head, how to train it on TartanAir with the backbone frozen, how to judge the
result, and how to ship it as a TensorRT engine. Read it fully before editing.

## 1. Goal and non-goals

Goal: a better-calibrated, more expressive covariance head for the existing FlowFormerCov
frontend, trained on TartanAir with the FlowFormer backbone frozen, evaluated against the
shipped head on TartanAir v2 test, exported as a drop-in TensorRT engine.

Non-goals (do not do these in this round):
- Do not unfreeze the backbone (`--training_mode flow` or all parameters). The flow must stay
  identical so the head changes get a clean A/B.
- Do not change the decoder iteration count (12), the input resolution (704x704 at
  deployment), or add warm-start initialisation (tested, hurts).
- No quantisation.

## 2. Repository orientation

Clone MAC-VO-MCH (the fork; `git log f0bd298..HEAD` are the recent exact-rewrite commits).
Weights: `Model/MACVO_FrontendCov.pth` (77 MB, not tracked; copy it or
`wget https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth`).

| piece | file |
|---|---|
| network | `Module/Network/FlowFormer/core/*` (vendored FlowFormer, TensorRT-friendly rewrites), `Module/Network/FlowFormerCov/flownet.py` (`FlowFormerCov`, `inference()`, strict `load_ddp_state_dict`), `Module/Network/FlowFormerCov/covhead.py` (`CovHead`, `CovUpdateBlock`, `MemoryCovDecoder`) |
| head internals | `CovHead`: 3x3 convs 128→256→128→64→2. `CovUpdateBlock`: `SepConvGRU(hidden 128, input 384)` + `CovHead` + mask head (128→256→576). `MemoryCovDecoder.forward`: per iteration `fcov_net, delta_cov = cov_update.forward_state(fcov_net, inp_cat)`; `cov_coords1 += delta_cov`; last iteration convex-upsamples `cov_coords1 - cov_coords0` (log sigma, 2 channels) |
| GRU fusion | `Module/Network/FlowFormer/core/gru.py::FusedGRUPair` slices the flow and cov GRU weights at call time (keeps checkpoint keys). Any change to the cov GRU's shapes must keep the `SepConvGRU` parameter layout or adapt `FusedGRUPair`. |
| consumers of cov | `Module/Frontend/Frontend.py` (`FlowFormerCovFrontend.inference_2_depth/_match`), `Module/Frontend/Matching.py` (`IMatcher.Output.cov` is `[B,3,H,W]` = σ_uu, σ_vv, σ_uv; `from_partial_cov` pads σ_uv = 0), `Module/Frontend/StereoDepth.py` (`disparity_to_depth_cov`), `Module/Covariance/Project2to3.py` (`MatchCovariance.estimate`, reads `flow_cov[..., 0/1/2]`), `Module/KeypointSelector.py` (quality `σuu + σvv − 2σuv`) |
| trainer | `Train/MatchingNet/train_flowformer.py`: `nn.DataParallel`, AdamW + OneCycleLR, GradScaler; `--training_mode cov` sets `requires_grad=False` on everything except `memory_decoder.cov_update`; loads `restore_ckpt` via `load_ddp_state_dict` (strict) |
| losses | `Train/MatchingNet/loss.py`: `sequence_loss` dispatches on `cfg.training_mode` (`flow`, `cov`, `finalcov`); `cov_loss` is a γ-weighted per-iteration Gaussian NLL; `depth_loss` exists but is unused |
| eval | `Train/MatchingNet/evaluate.py` (EPE, cov_ratio, NLL on TartanAir v2 test) |
| configs | `Config/Train/Demo.yaml` (restore from the shipped weights, batch 6, 480x640), `Config/Train/FlowFormerCovKITTIthings.yaml` (full recipe, 120k steps), `Config/Sequence/Training_Dataset/*.yaml` (sequence lists; `root:` paths must be edited) |
| pipeline eval | `MACVO.py --odom <cfg> --data <seq>`, `python -m Evaluation.EvalSeq --spaces ...` (ATE/RTE/ROE) |
| export / engine | `Export.py --mode fast` (opset 18, onnxsim fallback, `do_constant_folding=False`), `Build.py --mode fast --timing-cache runs/timing.cache`, `trtexec` for timing |
| parity harness | `Scripts/Parity/make_reference.py`, `compare.py`, `test_exact.py`, `iterations.py`, `profile_agg.py` |

Loader facts (`DataLoader/Dataset/TartanAir.py`): per sequence `image_left/`, `image_right/`,
`flow/` (`*_flow.png` with `compressed: true`, or official `*_flow.npy` + `*_mask.npy` with
`compressed: false`), `pose_left.txt`; `depth_left/` only if `gtDepth: true`. When
`root/<subdir>` does not exist, `resolve_dir` reads it in place from the official v1 zips next to
`root` (`<root.parent>/<modality>.zip`, see `DataLoader/Dataset/Zipped.py`). The trainer
accepts only `TartanAir_NoIMU` / `TartanAirv2_NoIMU` dataset types and trains on left-camera
temporal pairs (`frameData.cur.stereo.imageL`, `frameData.nxt.stereo.imageL`, `gt_flow`, `flow_mask`).
`flow_mask` is a validity mask downstream (`loss.py`: `mask &= flow_mask.bool()`).

## 3. Environment

Training imports the full `Module` package (DataLoader, scipy, pypose). Use the repo
Dockerfile (`Docker/Dockerfile`, CUDA 12.4) or a clean venv with `requirements.txt` plus
torch/torchvision for your CUDA. One 24 GB GPU is enough (the shipped recipe used 16 GB at
batch 6; `cov` mode uses less). Export and the parity harness import `Module/Network`
directly and also work in a lighter env (torch, tensorrt, onnx, onnxsim, numpy, opencv).

## 4. Data

TartanAir v1, all 18 public environments, Easy + Hard: 369 sequences, 306k flow pairs. Stored
as the official zips, never extracted, at
`D:/Datasets/TartanAirV1/<env>/<Easy|Hard>/{image_left,image_right,depth_left,flow_flow,flow_mask}.zip`
(1.03 TB, every file SHA-256 verified). `Scripts/Dataset/download_tartanair_v1.py` fetches or
resumes them. Sequence list: `Scripts/Dataset/make_tartanair_v1_seqs.py --data D:/Datasets/TartanAirV1
--out Config/Sequence/Training_Dataset/TartanAirV1_Zip.yaml` (`compressed: false`, `gtDepth: true`;
`--no-depth` for temporal-only runs). Each `root:` is `<data>/<env>/<level>/P###`, a path that
does not exist on disk; the loader reads it from the zips beside it. Do not use the authors'
`TartanAir_Train.yaml` / `TartanAir_TrainHard.yaml`. They point at an unreleased "v1.5"
(`house`, `house_dist0`, `slaughter`, and ocean Easy P003/P007 are not in public v1).

Flow mask polarity: the official v1 `*_mask.npy` is 0 = valid, 1 / 10 / 100 = out of view or
occluded. The `compressed: false` path converts it to a validity mask (`== 0`). Before this fix,
that path read the mask as channel 2 of the 2-channel `*_flow.npy` and got an empty array. The
`compressed: true` PNG path (v2) passes channel 2 through unchanged. The v2 docs call that format
"same as TartanAir V1", so check the zero fraction on v2 data before trusting `flow_mask.bool()`
there.

TartanAir v2 test (`TartanAirV2_Test.yaml`, 28 sequences, `TartanAirv2_NoIMU`): not in the
`tartanair` pip package's download index. The authors say it is released via tartanair.org
(MAC-VO issue #35, March 2026). It needs `image_lcam_front/`, `image_rcam_front/`,
`pose_lcam_front.txt`, and `flow_lcam_front/` for `evaluate.py`. Not downloaded yet.

## 5. Head changes (in this order)

### 5.1 Verify the training-time covariance representation first
`MemoryCovDecoder` accumulates log σ and `FlowFormerCov.inference` applies `exp(2·cov)`.
`cov_loss` in `loss.py`, however, uses `cov_preds[i]` directly as a variance
(`error / exp_cov + log(exp_cov)`). In training mode `forward` returns the raw accumulated
log-space tensors, so as written the loss and the representation disagree. Before anything
else, run one training batch with the shipped head, print `cov_preds[-1]` statistics, and fix
the loss to the log-space form (per axis): `nll = err² · exp(−2c) + 2c`. Do not trust the
shipped trainer blindly; it may be a snapshot that differs from what produced the released
weights. Also check `evaluate.py`: it uses `frameData.cur.imageL` / `gtFlow`, which does not
match the current loader API (`frameData.cur.stereo.imageL`, `gt_flow`).

### 5.2 Full 2x2 covariance (Cholesky parameterisation)
- `CovHead.conv4`: 64→3 channels. Per iteration the head emits Δc ∈ R³; the decoder
  accumulates `c = Σ Δc` (3 channels; `cov_coords1` becomes 3-channel, `upsample_flow` already
  handles any channel count; the cov mask head is unchanged).
- Mapping after upsampling: `L11 = exp(c₁)`, `L22 = exp(c₂)`, `L21 = c₃`;
  `Σ = L Lᵀ`: `σ_uu = L11²`, `σ_uv = L11·L21`, `σ_vv = L21² + L22²`.
- Loss (per pixel, per iteration, γ-weighted as today): with `r = pred − gt`,
  `nll = rᵀ Σ⁻¹ r + log det Σ`, where `log det Σ = 2(c₁ + c₂)` and `Σ⁻¹ r` is a 2x2 solve
  (write it out; no `torch.linalg.inv` on 1x1 batches). Keep the current mask handling.
- `inference()` returns `flow` and the 3-channel covariance `[σ_uu, σ_vv, σ_uv]`;
  `inference_2_match` then builds `IMatcher.Output` directly (no `from_partial_cov`), and
  `inference_2_depth` uses σ_uu as the disparity variance (as it uses channel 0 today).
- Initialise the GRU, mask head and `conv1`–`conv3` from `MACVO_FrontendCov.pth`; only the
  new `conv4` is random (load the checkpoint with the `conv4` keys dropped, strict otherwise).
  With the Cholesky form the shipped diagonal solution corresponds to `c₃ = 0`; initialise
  `conv4`'s third output row to zero and copy the two existing rows for c₁, c₂ so training
  starts from the shipped behaviour.

### 5.3 β-NLL
Scale the per-pixel NLL by `stopgrad(det Σ)^(β/2)` with β = 0.5 (Seitzer et al., "On the
pitfalls of heteroscedastic uncertainty estimation", 2022). One line in the loss; make β a
config value (β = 0 recovers plain NLL). Superseded for the frozen-backbone run: β = 0, see
section 10.

### 5.4 Stereo supervision
Add stereo batches: inputs `(imageL, imageR)` of the same frame, target flow
`(−f·b / depth, 0)` with `gtDepth: true`, validity mask from finite positive depth. Use the
same NLL on them. Mix temporal and stereo batches 1:1 (or alternate). This is the only place
the disparity σ that drives the 3D covariance is ever supervised.

### 5.5 Optional, second run: lighter branch
`SepConvGRU(hidden 64)` + `CovHead(64, ...)` for the covariance branch (~10% engine time).
Do this as a separate training so its effect is separable from 5.2–5.4. It changes parameter
shapes, so no warm start from the shipped head and `FusedGRUPair` must be adapted (it assumes
both GRUs share the hidden/input widths).

Keep as is: additive accumulation across iterations, γ weighting, per-iteration supervision,
`Project2to3.py`.

## 6. Trainer changes

- `train_flowformer.py` as shipped (and upstream `main`) could not run: `ConcatDataset` rejected
  `SequenceBase` (an `IterableDataset`), the `DataParallel` replica on GPU 1 used GPU 0's
  `FusedGRUPair` weights, and the loop overran `OneCycleLR` by one step. All three are fixed on
  `covhead-retrain` (section 10); sequences that fail to load are still dropped, but the loaded
  count is logged.
- `train_flowformer.py`: load the checkpoint minus the replaced `conv4` keys; keep
  `--training_mode cov`; add stereo batches to the loop; log the new metrics
  (NLL, mean |c₃|, calibration ratio `rᵀ Σ⁻¹ r` which should average ≈ 2).
- `loss.py`: new `cov_loss` (log-space, full 2x2, β-NLL); keep `flow_loss`.
- Config: start from `Config/Train/Demo.yaml`; batch 6, 480x640 crops, lr 12.5e-5 OneCycle,
  20k–40k steps, `autosave_freq 5000`. Hours on a 4090.

## 7. Evaluation gates

1. Sanity on the shipped head: `evaluate.py` (after the API fix) reproduces reasonable
   EPE/NLL numbers; the calibration ratio `rᵀ Σ⁻¹ r` near 2 on TartanAir v2 test.
2. New head vs shipped head on TartanAir v2 test with `evaluate.py`: NLL must improve;
   EPE must be unchanged (frozen backbone).
3. Full pipeline on TartanAir v2 test (both heads, same config otherwise):
   `python MACVO.py --odom Config/Experiment/MACVO/MACVO_Performant.yaml --data <seq>` per
   sequence, then `python -m Evaluation.EvalSeq --spaces ...`. RTE/ROE must not regress; the
   expected gain is on hard sequences and in fewer outlier keypoints.
4. Export: adapt `Export.py`'s wrapper to the 3-channel covariance output; outputs
   `flow`, `flow_raw`, `cov` `[B,3,704,704]`, `cov_raw`. Emit covariance entries in fp32
   in-graph (no `exp` left to the consumer). Build with `Build.py --mode fast`, time with
   `trtexec --shapes=image_1:2x3x704x704,image_2:2x3x704x704`. Reference: current engine
   278 ms batch-2 / 146 ms batch-1 on an RTX 2000 Ada laptop, 56 ms batch-2 on a 4090.
5. Parity: regenerate the fp32 reference (`Scripts/Parity/make_reference.py`, adapt for 3
   channels), then `compare.py --trt <plan> --pairing b2`.
6. Runtime (`MACVO_TRT`, C++): read 3 covariance channels, drop the `exp(2·cov)` kernel,
   feed σ_uv into the covariance model and selector. Re-tune `--lm-cov-scale` / robust
   thresholds only after the network-level gates pass.

## 8. Known facts to keep in mind

- bf16/fp16 inference deviates ~0.8 px mean EPE from fp32 on all pixels, concentrated in
  low-texture regions; the covariance never sees that error. Evaluate the trained head under
  the deployment dtype as well (`Export.py --mode fast` → `compare.py`).
- The decoder is unconverged at 12 iterations on all pixels but within 0.07 px on selectable
  pixels (stereo/temporal); do not change the iteration count to "fix" the head.
- Context mixing across batch samples was a real upstream bug and is fixed in this fork; the
  training batch size must not be assumed to be 1 anywhere in the head.

## 9. Opening prompt for the new session

"Read docs/covhead_retrain_handoff.md. Do section 5.1 first (verify the training-time
covariance representation against the loss with one batch on the shipped weights) and report
the finding before changing anything else. Then implement 5.2–5.4 and section 6, run one
short training to confirm the loop works end to end, and stop for review before the full run."

## 10. Status on branch `covhead-retrain` (2026-10-19)

Done, stopped for review before the full run (the section 9 brief).

**Data.** All 180 v1 zips on `D:/Datasets/TartanAirV1`, sizes and checksums verified (939 GiB
left on D:). `Config/Sequence/Training_Dataset/TartanAirV1_Zip.yaml`: 354 sequences, 297k
frames. `TartanAirV1_Zip_Val.yaml`: `seasidetown`, 15 sequences, held out. It is the one public
v1 environment missing from the authors' training lists, so neither head has trained on it.

**5.1 finding.** The shipped weights emit log sigma. `Scripts/AdHoc/CovLossCheck.py`, 96 training
pairs:
- 80-89% of `c` is negative.
- Read as log sigma, err^2/sigma^2 has mean 0.93 / 0.95 per axis and r^T Sigma^-1 r has mean
  1.87. The median error tracks 0.674 e^c across all c deciles.
- The shipped `cov_loss` is NaN on 82% of pixel terms and on every batch. It cannot have
  produced the released weights. Upstream `main` has the same `loss.py`, and no GitHub issue
  reports it.

The paper's loss (Eq. 1, log-space sigma updates) matches the fix.

**Masking decision: keep the covariance loss unmasked.** The pixels that a v1 mask flags are the
ones where the head must learn a large sigma. Shipped head, 96 pairs from 8 training environments:

| class | share | median sigma | median err | dropped by MAC-VO's 1.5x median filter |
|---|---|---|---|---|
| valid | 93% | 0.27 | 0.27 | 39% |
| self-occluded | 1.8% | 1.38 | 1.73 | 96% |
| out of view | 4.6% | 1.55 | 1.45 | 92% |
| cross-occluded | 0.4% | 0.36 | 0.38 | 52% |

Cross-occluded pixels are the weak spot. TartanVO excluded exactly this class from flow
training, so excluding it is a candidate ablation. `cov_mask: True` now works, but is an
ablation only.

**beta: 0 for the main run.** With the flow frozen, beta-NLL loses its purpose: it keeps a jointly
learned mean from under-fitting hard points. Each pixel's optimal sigma^2 is E[r^2] whatever beta is, so
beta only reweights pixels by sigma^(2 beta). At 0.5, a sigma = 2 px pixel counts 10x more than a
sigma = 0.2 px one. That moves capacity towards the uncertain pixels the selector discards, so
beta = 0.5 is an ablation (Config `beta`).

**Implemented** (`3d2a7ce`, `12454a8`, `9ff9357`):
- log-space NLL, a 2D Cholesky head (`cov_channels: 3`), optional beta-NLL (`beta`, 0 in the run), and alternating
  stereo steps (`stereo`) in `Config/Train/CovHeadChol.yaml`;
- `inference()` returns 2 or 3 covariance entries, and the frontends keep sigma_uv;
- `Export.py` refuses 3-channel heads until 7.4;
- `Train/MatchingNet/evaluate.py` rewritten for gates 1-2: temporal + stereo, per mask class, all
  checkpoints on the same pairs and pixels.

Checks run (`3d2a7ce`):
- A shipped checkpoint in the Cholesky model reproduces the shipped sigma (2e-4 relative) with
  sigma_uv = 0.
- The NLL matches an explicit inverse and log det.
- Gradients reach `cov_update` only.
- A 200-step run on both GPUs: EPE is unchanged to 4 decimals, |rho| reaches 0.06.

**Baseline (shipped head, `seasidetown`, 60 pairs).**
- Temporal: r^T Sigma^-1 r mean 2.20, median 0.90 (ideal 2, 1.386); NLL 0.58 nats; EPE 0.76.
- Stereo: r^T Sigma^-1 r mean 5.73, median 0.86; NLL 2.06; EPE 0.96. The disparity sigma is
  overconfident in the tail.

The means are outlier-driven: 120 pairs gave temporal 4.38 and stereo 16.8. Gate on NLL and the
medians, over at least 600 pairs.

**Numerics.** With cuDNN's default TF32 convolutions, toggling `requires_grad` changes kernel
choice. Flow then moves by a mean of 5e-4 px (max 0.29 px), and sigma^2 by a median of 2e-4
relative. Compare EPE with a tolerance. `evaluate.py` builds every checkpoint the same way.

**Run.** About 1.5 steps/s on 2x RTX 4090 (`DataParallel`, batch 6), so 30k steps take about
5.5 h.

    python -m Train.MatchingNet.train_flowformer --config Config/Train/CovHeadChol.yaml --training_mode cov
    python -m Train.MatchingNet.evaluate --config Config/Train/CovHeadChol.yaml         --ckpt Model/MACVO_FrontendCov.pth Model/FlowFormerCovChol<time>/30000.pth --pairs 600

**Open.**
- The v2 test set is not downloaded: issue #35 says it is on tartanair.org, but the exact link
  is not found yet.
- Gates 3-6 (pipeline, export, parity, C++ runtime) are not started.
- 5.5 (the lighter branch) is not started.
- The paper's per-iteration decay is 0.8; the config uses gamma 0.85, kept as shipped.

## 11. Beyond the paper: what the head is for, and what to try next (2026-10-19)

**What the head is for.** For every frontend call, the head predicts per pixel how wrong the
frozen matcher is, as a 2D Gaussian. The calls are stereo L->R for depth, temporal t->t+1, and
in `MACVO_TRT` keyframe->current. MAC-VO consumes the prediction in three ways, so three
properties matter:
- **Ranking.** Keypoint selection drops depth σ² above `max_depth_cov`, ranks by depth variance
  x (σ_uu + σ_vv − 2σ_uv), and filters at 1.5x the median (`Module/KeypointSelector.py`).
- **Calibration of the keypoints that survive selection.** Their 3D covariance
  (`Project2to3.py`) weights the pose graph. The absolute scale matters as well, because frame
  covariances are added and the LM/iSAM2 robust thresholds are tuned to the shipped σ.
- **Catching confident failures.** A wrong match that survives selection costs the most.

All-pixel NLL is only a proxy for these.

**TensorRT.** The changes so far are convs and element-wise ops. Map Cholesky -> Σ in fp32
in-graph: σ² = e^(2c) reaches ~1.6e5, beyond fp16's 65504. The `cov` output becomes 3
covariance entries (7.4, 7.6).

**Ranked next steps.** In TensorRT each is element-wise or one extra channel, unless noted.
1. **Measure what VO uses (no model change).** Sparsification curves and AUSE (Ilg et al. 2018),
   reliability curves per σ bin, and calibration on the keypoints MAC-VO's selector picks, all
   per call type.
2. **Inlier probability.** Errors are heavy-tailed: calibration mean ≈ 2 but median ≈ 0.9
   against 1.39. A single Gaussian is therefore too wide on the bulk and too narrow on failures.
   - Model the error as p·N(0, Σ) + (1 − p)·(broad outlier term). Predict Σ and p (one channel
     plus a sigmoid) and train on the mixture likelihood.
   - The selector uses p; the pose graph gets a Σ calibrated on inliers.
   - Lineage: PDC-Net / PDC-Net+ (Truong et al. 2021, 2023), SEA-RAFT's mixture of Laplace.
   - Needs the runtime to read p.
3. **Keyframe pairs in training.** The keyframe call has the largest decoder residual
   (0.35 px) and is never trained. TartanAir ships GT flow for consecutive frames only, so
   t->t+k flow has to come from depth and pose: back-project, transform, re-project, and use a
   depth test for occlusion.
4. **Deployment precision.** fp16/bf16 moves flow by ~0.8 px mean (section 8).
   - Cheap: a per-call-type monotone correction of log σ, fitted on engine outputs over held-out
     data and applied in-graph or in the runtime. It is also the point to re-tune the backend
     thresholds.
   - Better but harder: run the frozen part in the deployment dtype during training.
5. **Focus on the deployed iteration.** Only the 12th σ is used, so `finalcov` mode or a
   smaller γ is a cheap ablation.
6. **Call-type awareness.** If stereo calibration is still off after this run, a per-call-type
   output scale in the runtime is free. A learned call-type input would change the engine
   interface.

**Avoid.**
- β-NLL with a frozen matcher (section 10).
- Masking the unknowable pixels (section 10).
- Left-right or forward-backward consistency features: they need extra network passes at
  deployment.
- A Student-t likelihood alone: the pose graph wants a Gaussian, and item 2 gives the same
  robustness with an explicit outlier probability.

**Order.** After the first full run: 1, then 6 and 3, then 2, then 4. The head only reads the
decoder's 384-channel `inp_cat`, so all of this carries over to SEA-RAFT
(`docs/searaft_cov_handoff.md`). For the replication itself, gate 3 on TartanAir v2 test remains
the yardstick, so finding that data comes first.

## 12. Results so far (2026-10-21)

All heads keep the flow frozen, use the log-space NLL, β = 0, no mask, and alternate stereo steps.
Held-out data is `seasidetown` throughout.

| head | config | starts from | steps | change |
|---|---|---|---|---|
| 25k | `CovHeadChol` | shipped head | 25k of 30k | recurrent Cholesky head (sections 5.2, 5.4) |
| SS A | `CovHeadSingleShot` | fresh head | 30k | single-shot head after the decoder loop |
| SS B | `CovHeadSingleShotInlier` | fresh head | 30k | SS A + inlier probability, mixture NLL |
| cues | `CovHeadCholCues` | 25k | 20k | + correlation-peak cues and flow update as head input |
| cont | `CovHeadCholCont` | 25k | 20k | control for cues: same schedule, no cues |
| deploy | `CovHeadCholDeploy` | 25k | 15k | frozen network in fp16 / bf16 (running) |
| water | `CovHeadCholWater` | 25k | 15k | + 7 seasidetown sequences x10 (running) |

The 30k checkpoint of the first run is worse than 25k in the tail (temporal calib mean 3.15 against
2.35), so 25k is the reference.

**Network level, 600 held-out pairs** (`evaluate.py`):

| | temporal NLL | temporal calib mean / median | temporal AUSE | stereo NLL | stereo calib mean / median | stereo AUSE | stereo tail > 9.21 |
|---|---|---|---|---|---|---|---|
| shipped | 2.25 | 5.18 / 1.03 | 0.138 | 16.5 | 34.4 / 0.95 | 0.277 | 7.4% |
| 25k | 2.35 | 5.49 / 1.06 | 0.142 | 12.0 | 25.8 / 1.08 | 0.236 | 5.9% |

The temporal head is a wash. The gain is in stereo, the call behind every keypoint's depth sigma.

**At the keypoints `CovAwareSelector` picks** (`Scripts/AdHoc/StereoTailAnalysis.py`, 600 pairs):
- Stereo r^T Sigma^-1 r > 9.21: 6.45% (shipped) -> 5.44% (25k). Calibrated would be 1%.
- Confidently wrong keypoints are almost all predicted nearer than GT, in low texture. The two
  sources are water surfaces (the GT depth is likely the seabed) and sky bleeding into nearby
  structure. 25k cuts the sky failures (GT depth > 80 m: 18% -> 3% of catastrophic keypoints).

**MAC-VO, 15 held-out sequences** (`MACVO_Performant` with `CovAwareSelector`, fp32; runs are
deterministic, so one repeat each):

| | ATE m | RTE | ROE |
|---|---|---|---|
| shipped | 1.36 | 0.0094 | 0.056 |
| 25k | 1.04 | 0.0088 | 0.048 |
| cont | 1.04 | 0.0089 | 0.048 |
| cues | 1.11 | 0.0090 | 0.049 |

- Most of the gain is `Hard_P003`: shipped ATE 4.87 m, 25k 0.64 m. The shipped run diverges at
  step 152 on sky keypoints.
- Without `Hard_P003`, the averages are ATE 1.11 (shipped) / 1.07 (25k), RTE 0.0086 / 0.0089,
  ROE 0.047 / 0.047. So 25k removes a failure mode and is level elsewhere. It does not improve
  the typical sequence.
- Cues add nothing over the control, and 20k more steps (cont) add nothing over 25k.

On 5 of these sequences (two repeats each): SS A reaches ATE 1.43 against 1.37 for 25k and 2.17
for shipped. SS B with the moment-matched output is worse (2.03). With the gated output
(Sigma_inlier if p >= tau, else the outlier Sigma) it reaches 1.55. Neither single-shot head beats
the recurrent one.

**TensorRT** (`Export.py --mode fast`, `trtexec` opt level 5, strongly typed, batch 2, RTX 4090):
- Shipped engine: 56.1 ms. Flow only (no covariance branch): 48.8 ms. SS A: 50.8 ms.
- Engines match PyTorch to 0.01-0.02 px and about 1% sigma on confident pixels.
- Under reduced precision, the single-shot sigma moves by about 20% in low texture.
- The numbers above come from a scratch wrapper. `Export.py --ckpt` now exports 3-channel heads
  (section 13).

**Status (superseded by section 13).**
- Fallback: the recurrent 25k head (`Model/FlowFormerCovChol10-19-17-52-09/25000.pth`).
- Run one GPU job per GPU: two jobs on a 24 GB card overflow it, and WDDM paging then stalls both
  for hours. This applies to trainings, MAC-VO, `evaluate.py` and `trtexec` alike.

## 13. EIVA underwater data: in-domain training and evaluation (2026-10-22)

**Data.** `D:/Datasets/EIVA` has five rectified stereo dives at 2816x2816: four at Vobster Quay
(`plane_mid`, `plane_nose`, `plane_tail1`, `plane_tail2`, 1,858 frames) and Welland (4,718 frames).
Each comes with a 3DF Zephyr reconstruction: per-image poses (`pose_gt.txt`, world to camera,
read as in MACVO_TRT's `scripts/gt_to_tum.py`) and a dense coloured point cloud
(`pointcloud_gt.txt`, 2-56M points).

**Ground truth** (`Scripts/Dataset/make_eiva_gt.py`, output in `D:/Datasets/EIVA_GT`, ~6 GB):
- Depth and flow to the next posed frame are rendered at 704x704 (K / 4, the frontend's
  resolution).
- A GPU z-buffer keeps the nearest point per pixel. A pixel is dropped when it lies more than 5%
  behind the nearest depth in its 7x7 window, which removes points seen through holes.
- The flow is the displacement of the point each pixel keeps.
- 41-50% of pixels are covered; water and the far seabed have none.
- The frozen FlowFormer agrees with it to a 0.10-0.26 px median on Vobster and 0.06-0.4 px on
  Welland. On its most confident pixels the error is 0.06-0.2 px.
- `Scripts/Dataset/check_eiva_gt.py` measures each frame's median ratio of predicted to rendered
  disparity. Vobster is within 0.5% on 90% of frames. On Welland, 4.5% of frames are off by more
  than 2.5% and 3.8% by more than 5%, which points to local reconstruction errors. `EIVA_NoIMU` leaves those
  frames unsupervised (`max_scale_error` 0.025).

**Training.** `CovHeadCholDeploy`'s recipe (25k head, frozen network in fp16 / bf16, 15k steps,
lr 6e-5, stereo alternation) plus one EIVA site at ~25% of the pairs. The split is by site:
- `CovHeadEIVA_Vobster` (fold V) trains on Vobster and is tested on Welland.
- `CovHeadEIVA_Welland` (fold W) trains on Welland and is tested on Vobster.

Each fold takes ~2 h on one 4090 (2.1 steps/s).

**Network level** (`evaluate.py --precision fast`, 600 held-out pairs; the fold column is the
fold that did not train on that site):

| Welland (fold V) | shipped | 25k | deploy | fold V |
|---|---|---|---|---|
| temporal NLL / tail > 9.21 / AUSE | 2.44 / 7.2% / 0.108 | 2.20 / 7.6% / 0.109 | 2.54 / 8.7% / 0.105 | **1.56 / 6.1% / 0.099** |
| stereo NLL / tail > 9.21 / AUSE | 13.3 / 7.2% / 0.460 | 15.8 / 7.2% / 0.465 | 22.7 / 11.1% / 0.444 | **10.1** / 7.3% / **0.384** |

| Vobster (fold W) | shipped | 25k | deploy | fold W |
|---|---|---|---|---|
| temporal NLL / tail > 9.21 / AUSE | 1.25 / 4.1% / 0.125 | 1.32 / 4.8% / 0.127 | 1.36 / 5.2% / **0.117** | **0.94 / 2.4%** / 0.121 |
| stereo NLL / tail > 9.21 / AUSE | 1.88 / 3.9% / 0.145 | 2.29 / 4.9% / 0.144 | 2.79 / 6.5% / **0.137** | **1.24 / 3.0%** / 0.149 |

- The TartanAir-only heads are overconfident underwater (stereo tail up to 11%).
- Both folds fix the calibration tail.
- Fold W's sigma is wide on the bulk (temporal calib median 0.71, ideal 1.386): it learned
  Welland's noisier GT (stereo median 0.3-0.4 px against 0.15 px on Vobster) as uncertainty.
  Fold V, trained on the cleaner Vobster GT, also improves ranking.

**MAC-VO** (`MACVO_Performant` + `CovAwareSelector`, deployed fp16 / bf16, 704x704 INTER_AREA as
the runtime feeds the engine). Positions are SE3-aligned to the GT (no scale, as MACVO_TRT's
`compare_trajectories.py`). ATE in m; frame-to-frame displacement error in mm:

| sequence (GT path) | shipped | 25k | deploy | EIVA, held out |
|---|---|---|---|---|
| plane_mid (15.7 m) | 0.018 / 2.35 | 0.018 / 2.29 | 0.017 / 2.10 | **0.010 / 1.57** (W) |
| plane_nose (68.0 m) | 0.404 / 8.76 | 0.365 / 8.92 | 0.370 / 8.31 | **0.319 / 7.37** (W) |
| plane_tail1 (52.0 m) | 0.060 / 2.92 | 2.324 / 45.5 | 0.077 / 3.26 | **0.041 / 2.79** (W) |
| plane_tail2 (77.8 m) | 1.616 / 43.0 | 0.743 / 19.3 | 0.442 / 11.7 | **0.151 / 6.77** (W) |
| welland (184.1 m) | 0.217 / 2.13 | 0.171 / 1.61 | 0.174 / 1.86 | **0.136 / 1.36** (V) |
| mean | 0.463 / 11.8 | 0.724 / 15.5 | 0.216 / 5.45 | **0.131 / 3.97** |

- Whole-dive `plane_tail2` numbers are chaotic, so this table overstates the gain (see below).
  Two heads with identical network metrics, fold W v1 and v2, give 0.15 m and 0.74 m there.
- The stable comparison cuts `plane_tail2` at frame 486, before the stretch where every head
  breaks. Mean over the five dives, ATE m / frame-to-frame mm:
  - shipped 0.148 / 3.60
  - 25k 0.586 / 12.1
  - deploy 0.142 / 3.56
  - EIVA held out, v1 0.113 / 3.04
  - EIVA held out, v2 0.113 / 3.01
- So the EIVA heads cut ATE by ~24% and frame-to-frame error by ~16% against shipped. They are
  better on four of five dives; on `plane_tail2` before the cut shipped is best (0.041 m against
  0.047-0.061 m).
- The 25k head (trained in fp32) breaks in deployed precision on `plane_tail1`. The deploy head
  (trained in fp16 / bf16) does not, so deployment-precision training matters here.
- MAC-VO runs are deterministic, so each number is one run. The scripts are in the session
  scratchpad (`macvo/eval_eiva.py`, `macvo/cfg_eiva`).

**Production head** (`CovHeadEIVA_All`: the same recipe on both sites, Vobster x27 + Welland x10,
~24% of the pairs). Current: v2, trained with the temporal GT gate,
`Model/FlowFormerCovEIVAAll10-22-08-46-09/15000.pth`. v1 is `...EIVAAll10-22-02-48-27`, trained
before the gate. The ONNX (`Export.py --mode fast --ckpt`) and the opt-level-5 engine sit next to
each checkpoint (`MACVO_FrontendCov_EIVAAll_fast.onnx`, `..._o5.plan`).
- v2 engine against PyTorch fp32 on confident pixels: EPE 0.011 / 0.017 px, |d log sigma| 1.1%.
- v2 engine timing: 58.6 ms against 56.8 ms for the shipped engine in the same session.
- The production head cannot be scored on held-out EIVA data, since it trained on all of it; the
  folds validate the recipe.

**TensorRT (gates 4-5) for the production head.**
- `cov` is `[B, 3, 704, 704]` `[sigma_uu, sigma_vv, sigma_uv]` in fp32, computed in-graph.
- Against PyTorch fp32 on confident pixels: EPE 0.010 px (stereo) / 0.017 px (temporal),
  |d log sigma| 1.1% / 1.2%. PyTorch in fp16 / bf16 alone gives 0.009 / 0.015 px and
  1.05% / 0.92%, so the engine adds almost nothing beyond the precision.
- Batch 2 on an otherwise idle 4090: 57.8 ms, against 56.1 ms for the shipped engine (+3%: the
  third channel and the in-graph fp32 Sigma).
- The C++ runtime (gate 6) still reads 2 channels of log sigma. To use this engine it must read 3
  channels, drop its `exp(2 * cov)`, and feed sigma_uv into `covariance_2to3` and the selector.

**Does EIVA training cost anything on TartanAir?** MAC-VO on the 15 seasidetown sequences, deployed
precision:

| | shipped | 25k | deploy | fold V | production v1 | production v2 |
|---|---|---|---|---|---|---|
| ATE m / RTE / ROE | 1.07 / 0.0085 / 0.047 | 1.09 / 0.0087 / 0.047 | 1.05 / 0.0090 / 0.048 | 1.14 / 0.0094 / 0.053 | 1.12 / 0.0094 / 0.053 | 1.16 / 0.0095 / 0.054 |
| `Hard_P003` ATE | 0.61 | 0.66 | 1.42 | 2.33 | 2.23 | 2.57 |

- On the other 14 sequences all heads are level (fold V ATE 1.06 against 1.10 shipped; ROE 0.047
  both).
- All three EIVA heads fail `Hard_P003`, the sky-heavy sequence where shipped failed in fp32, so
  this is consistent, not chance. It is the trade-off of the underwater data.
- In deployed precision none of the heads gains on TartanAir. The fp32 gain of section 12
  (shipped's `Hard_P003` failure) does not occur in fp16 / bf16.

**Water fine-tune** (`CovHeadCholWater`, the 8 held-out seasidetown sequences, deployed precision):
ATE 1.32 / RTE 0.0116 / ROE 0.055. Shipped gives 1.36 / 0.0120 / 0.062; 25k and deploy give
1.33-1.38 / 0.0124-0.0128 / 0.061-0.062. Water is best on all three, with ROE 12% better.

**plane_tail2: a hard stretch that breaks most heads.** Whole-dive ATE on `plane_tail2`:

| head | ATE |
|---|---|
| shipped | 1.62 m |
| 25k | 0.74 m |
| deploy | 0.44 m |
| fold V | 1.12 m |
| production | 1.11 m |
| fold W | **0.15 m** |

The production checkpoints give 0.32 / 1.26 / 1.11 m at 5k / 10k / 15k; fold W's give 0.18 / 0.15 m at 10k / 15k.

- Every head breaks in the same stretch, image frames ~486-516, where the reconstruction lost 42
  cameras.
- Before it (frames 0-485) the production head tracks best of all: ATE 0.030 m, against 0.041 m
  for shipped, 0.070 m for deploy and 0.061 m for fold W.
- A per-pair temporal check found wrong relative poses in that stretch: pair 505 is off by 58 px,
  504 by 2 px, against 0.1-0.2 px elsewhere. `check_eiva_gt.py` now stores `flow_error` per pair,
  and `EIVA_NoIMU` drops pairs beyond 0.75 px (2.9% of `plane_tail2`, 2.3% of Welland, <=0.2% of
  the other dives).
- The gate is not the fix. Fold V retrained with it (v2) still fails `plane_tail2` in-sample
  (1.28 m), and shipped and 25k fail it without any EIVA data. Fold V v2 matches v1 out of sample
  (Welland ATE 0.147 against 0.136 m; network metrics within 0.3%).
- Wider sigma is not what carries fold W through. A global sigma^2 scale on the production head
  does not help (x2 1.18 m, x4 1.33 m, against 1.11 m).
- Robustness under small perturbations (`cov_scale` x0.8 / 0.9 / 1 / 1.1 / 1.25, whole-dive ATE):

  | head | median | runs with ATE > 0.4 m |
  |---|---|---|
  | shipped | 1.57 m | 5/5 |
  | deploy | 0.44 m | 5/5 |
  | fold W v1 | 0.15 m | 0/5 |
  | fold W v2 | 0.71 m | 3/5 (0.14 m at x0.8-0.9, 0.71-0.74 m from x1.0) |

  - The outcome is a property of the trained head: three of the four are stable.
  - Two training runs that differ by 2% of Welland pairs land on either side of the edge. How a
    head copes with this stretch varies between training runs, so judge it on several runs or
    seeds, not one.
  - A third fold W run (`CovHeadEIVA_Welland_seed2`, seed 4321) is robust again: median 0.21 m,
    0/5 runs over 0.4 m. So two of three Welland-trained runs survive the stretch, against none
    of the shipped and deploy runs.

**Run-to-run variance of fold W (held out on Vobster, `plane_tail2` cut at frame 486).** Mean ATE:

| | shipped | deploy | v1 | v2 | seed 2 |
|---|---|---|---|---|---|
| mean ATE (m) | 0.131 | 0.134 | 0.108 | 0.104 | 0.105 |

The ~20% gain over shipped and deploy holds in all three runs. The network-level metrics of the
three runs agree within a few percent.

**Run-to-run variance of fold V (held out on Welland, 184 m).** ATE m / frame-to-frame mm:

| | shipped | 25k | deploy | v1 | v2 | seed 2 |
|---|---|---|---|---|---|---|
| Welland | 0.217 / 2.13 | 0.171 / 1.61 | 0.174 / 1.86 | 0.136 / 1.36 | 0.147 / 1.38 | 0.145 / 1.43 |

- About 33% better ATE than shipped and 17% better than deploy, in all three runs.
- Stereo AUSE is 0.373-0.383 in all three, against 0.458 for shipped.

**EIVA + water head** (`CovHeadEIVA_AllWater`: the production mix +
`TartanAirV1_Seaside_Train`; `Model/FlowFormerCovEIVAAllWater10-22-13-17-15/15000.pth`).
- Its ONNX and opt-level-5 engine sit next to it (`MACVO_FrontendCov_EIVAAllWater_fast.onnx`,
  `..._o5.plan`).
- Engine against PyTorch fp32 on confident pixels: EPE 0.011 px (stereo) / 0.019 px (temporal),
  |d log sigma| 1.2-1.3%.
- Batch 2: 57.9 ms against 55.9 ms for the shipped engine (+4%).

On the 8 held-out seasidetown sequences, deployed precision, ATE m / RTE / ROE:

| head | all 8 | without `Hard_P003` | `Hard_P003` ATE |
|---|---|---|---|
| shipped | 1.36 / 0.0120 / 0.062 | 1.46 / - / 0.063 | 0.61 |
| deploy | 1.33 / 0.0128 / 0.061 | 1.32 / - / 0.061 | 1.42 |
| water | 1.32 / 0.0116 / 0.055 | 1.41 / - / 0.054 | 0.71 |
| production v2 | 1.50 / 0.0138 / 0.073 | 1.35 / - / 0.064 | 2.57 |
| EIVA + water | **1.11 / 0.0116 / 0.054** | **1.15** / - / 0.055 | 0.87 |

- The seaside data removes the EIVA heads' `Hard_P003` failure (2.57 -> 0.87 m, ROE 0.141 ->
  0.052). It holds under perturbation (`cov_scale` x0.8 / 0.9 / 1 / 1.1 / 1.25, ATE m):
  - EIVA + water: 0.70-0.94, ROE 0.049-0.054. Never fails, best median ROE.
  - shipped: 0.58-2.39, fails at x0.8 / 0.9.
  - deploy: 0.53-2.33, fails at x0.8-1.0.
  - production v2: 1.90-2.85, fails at every scale.
- It is the best head on held-out TartanAir.
- On the EIVA dives (in-sample, like the production head) it matches production v1: 0.012 /
  0.321 / 0.035 m on `plane_mid` / `plane_nose` / `plane_tail1`, 0.132 m on Welland. It breaks in
  `plane_tail2`'s hard stretch like every in-sample head (1.30 m).
- Its recipe has a held-out EIVA test of its own: the site folds `CovHeadEIVA_VobsterWater` /
  `CovHeadEIVA_WellandWater`, deployed precision. ATE m / frame-to-frame mm, `plane_tail2` cut at
  frame 486:

  | held out | shipped | deploy | EIVA only (3 runs) | EIVA + water |
  |---|---|---|---|---|
  | Welland | 0.217 / 2.13 | 0.174 / 1.86 | 0.136-0.147 / 1.36-1.43 | 0.145 / 1.51 |
  | Vobster (mean of 4) | 0.131 / 3.97 | 0.134 / 3.98 | 0.104-0.108 / 3.36-3.46 | 0.111 / 3.42 |

  - Adding the water data keeps the underwater gain within the EIVA-only runs' spread.
  - Held-out NLL on Welland improves: temporal 1.42, stereo 9.65, against 1.49 and 10.1 for fold
    V.
  - On Vobster the stereo ranking is slightly worse than shipped's (AUSE 0.144 against 0.132),
    yet the trajectories are better.
  - Fold W + water gets through `plane_tail2`'s hard stretch (whole dive 0.33 m, shipped 1.62 m).

**GT v2** (`D:/Datasets/EIVA_GT_v2`, `make_eiva_gt.py` defaults). The v1 renders had speckle and small square
holes. Two causes, both fixed:

1. **Reconstruction floaters**, 1.8-2.6% of each cloud. They carried wrong depth, and each punched a 7x7 hole through
   the occlusion test. Removed by statistical outlier removal (mean distance to the 8 nearest neighbours above
   mean + 2 std).
2. **Pixels no point lands on.** Filled from a 3x3 splat. The fill runs after the occlusion test, so pixels whose own
   point was a see-through reject take the surface in front, then the test runs again; their flow back-projects the
   pixel centre.

Mean coverage per dive, v1 -> v2:

| | plane_mid | plane_nose | plane_tail1 | plane_tail2 | welland |
|---|---|---|---|---|---|
| v1 | 50% | 42% | 41% | 48% | 50% |
| v2 | 58% | 54% | 54% | 61% | 71% |

- The consistency checks pass as for v1 on all five dives (Welland: 4.6% of frames beyond the 2.5% scale gate, 2.4%
  of pairs beyond 0.75 px).
- Filled pixels against the frozen FlowFormer, all five dives:
  - within each confidence band they agree as well as the pixels with a point of their own (plane_mid); they
    concentrate in the hard, sparsely reconstructed regions;
  - where they disagree by more than 3 px, the cause is not the fill. On plane_nose 55-81% of those pixels map
    outside the other image (near range). On Welland they lie in the textureless haze, where FlowFormer itself is
    unreliable, and a patch-correlation check could not decide which of the two is right;
  - on the pixels both versions cover, v2 agrees with FlowFormer better than v1.
- The trainings use v2 with the filled pixels. `EIVA_NoIMU`'s `use_filled: false` drops them; the ablation was
  dropped, since nothing showed the filled pixels to be wrong.

Two alternatives were tested on held-out frames against the frozen FlowFormer, and neither is better:

- **Front-surface averaging** (mean of the points within 0.3-1% of the nearest): stereo error on confident pixels
  rises from 0.070 to 0.087-0.115 px, and coverage drops without the fill. The extra points in a pixel lie mostly
  behind the visible surface.
- **Oriented surfel splatting** (ray-disc depth, radius = point spacing, back-face culling when the normals allow):
  - on plane_mid, equal (0.070 against 0.071 px; 12 pixels differ by more than 5%);
  - on plane_tail2, worse (0.214 against 0.193 px), because noisy normals tilt discs past silhouettes.

Decision: v2 as rendered. The fold trainings below use it.

**Using the head downstream: sigma_uv, the selector, the flow-covariance floor.**

How the frontend's covariance reaches MAC-VO:

- Stereo sigma only drives keypoint selection.
  - The pose graph's depth variance comes from the depth map's spread in a patch shaped by the *flow* covariance
    (`MatchCovariance`, `Project2to3.py:162`).
- Flow sigma drives both selection and the pose-graph weights.
  - Its floor is `min_flow_cov` (0.25 px), and the head predicts 0.08-0.15 px on confident pixels.

1. **sigma_uv.** On held-out data it lowers temporal NLL (|rho| 0.2-0.3, better on 60-65% of pixels) but not the
   trajectories. On the EIVA + water folds:
   - dropping it (`cov_offdiag: false`): Welland 0.142 against 0.145 m, Vobster 0.103 against 0.111 m;
   - ranking by the trace (`flow_score: trace`; upstream uses sigma_uu + sigma_vv - 2 sigma_uv): Welland 0.140 m,
     Vobster 0.105 m;
   - held-out TartanAir: RTE / ROE level.

   The runtime can therefore take sigma_uu, sigma_vv from the 3-channel engine and ignore sigma_uv (gate 6 shrinks
   to that).
2. **The floor** (`min_flow_cov`; `m_MinFlowCov` in MACVO_TRT). Held-out EIVA, deployed precision, ATE m /
   frame-to-frame mm:

   | | Welland | Vobster mean (cut) |
   |---|---|---|
   | shipped, 0.25 px | 0.217 / 2.13 | 0.131 / 3.97 |
   | shipped, 0.10 px | 0.122 / 1.41 | 0.119 / 3.83 |
   | EIVA + water, 0.25 px | 0.145 / 1.51 | 0.111 / 3.42 |
   | EIVA + water, 0.15 px | 0.134 / 1.41 | 0.106 / 3.33 |
   | EIVA + water, 0.10 px | 0.118 / 1.34 | 0.104 / 3.36 |

   - The floor is the largest single gain: shipped improves by 44% on Welland from the setting alone.
   - At 0.10 px the EIVA head adds -3% ATE / -5% frame-to-frame on Welland and -13% / -12% on Vobster. The
     larger gaps at 0.25 px came from the floor holding the shipped head back.
   - On held-out TartanAir (EIVA + water head) 0.10 px is neutral: ATE 1.15 against 1.11 m, RTE 0.0113 against
     0.0116.
   - Recommended runtime setting: 0.10 px, also with the shipped head. Compare all heads at 0.10 px from here on.
   - 0.05 px (EIVA + water folds): Welland 0.108 / 1.31, Vobster 0.104 / 3.35, so no worse anywhere; the shipped
     head at 0.05 px and held-out TartanAir at 0.05 px are queued.

**Evaluation protocol correction: breaks in the GT trajectory (2026-10-23).** The per-pair GT check
(`check_eiva_gt.py`'s `flow_error`, the frozen FlowFormer against the rendered flow) is ~0.2 px almost everywhere, but
a few pairs are far off:

- `plane_nose` 513 -> 514: 318 px. The vehicle swings ~20 deg in 1.2 s with half the image open water. Either the
  GT pose or the matcher fails there; every head's trajectory turns ~35 deg against the GT within 20 frames.
- `plane_tail2`: two pairs at 59 and 8 px, and 42 frames without a GT pose. This is the "hard stretch" the cut at
  frame 486 removed.
- Welland 1863 -> 1871: three pairs at 8-11 px.

With one SE3 alignment per dive, every frame after a break counts against every head, and these few frames dominated
the averages. `plane_nose`'s last 50 frames alone made up most of the "Vobster mean". New protocol
(`Scripts/Dataset/eval_eiva_trajectory.py`, the default; `--no-split [--cut 486]` gives the old numbers):

- Split each dive at every pair whose GT check is above 3 px and at every frame without a GT pose.
- Align each segment on its own; drop segments shorter than 10 frames.
- ATE: RMS over all frames. Frame-to-frame: mean over the pairs inside segments.
- No cut, and the rule uses the GT check only, never a head's result.

Held-out EIVA, ATE m / frame-to-frame mm, old protocol -> split:

| | Welland | Vobster mean |
|---|---|---|
| shipped, 0.25 px | 0.217 / 2.13 -> 0.119 / 1.53 | 0.131 / 3.97 -> 0.051 / 3.73 |
| shipped, 0.10 px | 0.122 / 1.41 -> **0.080** / 1.19 | 0.119 / 3.83 -> 0.038 / 3.50 |
| deploy, 0.10 px | 0.129 / 1.45 -> 0.112 / 1.34 | 0.121 / 3.69 -> 0.041 / 3.19 |
| EIVA + water fold, 0.25 px | 0.145 / 1.51 -> 0.115 / 1.36 | 0.111 / 3.42 -> 0.038 / 2.73 |
| EIVA + water fold, 0.10 px | 0.118 / 1.34 -> 0.093 / 1.20 | 0.104 / 3.36 -> 0.032 / 2.61 |
| EIVA + water fold, 0.05 px | 0.108 / 1.31 -> 0.085 / **1.17** | 0.104 / 3.35 -> **0.029 / 2.54** |
| EIVA only folds (3 runs), 0.25 px | 0.111-0.116 / 1.27-1.32 | 0.031-0.042 / 2.61-2.83 |

- The floor stays the robust gain, for every head and under both protocols.
- The EIVA-tuned head's own trajectory gain is smaller than reported above:
  - Vobster: -16% ATE / -25% frame-to-frame against shipped at 0.10 px;
  - Welland: none. Shipped at 0.10 px has the better ATE (0.080 against 0.093), and frame-to-frame is equal.
- The network-level gains (NLL, calibration, ranking) are unaffected.
- The remaining large errors sit where the matcher itself fails, which is what the flow fine-tune targets.
- Report both protocols from here on, the split one as primary.

**Settling the head: GT v2, the inlier head, the floor (2026-10-23).** Site folds with the EIVA + water recipe,
deployed precision, held out as before. MAC-VO ATE m / frame-to-frame mm, split protocol (old protocol in brackets):

| held out: Vobster (mean of 4) | 0.25 px | 0.10 px |
|---|---|---|
| shipped | 0.051 / 3.73 (0.131 / 3.97) | 0.038 / 3.50 (0.119 / 3.83) |
| deploy | 0.059 / 3.70 (0.134 / 3.98) | 0.041 / 3.19 (0.121 / 3.69) |
| v1 GT, 3-channel | 0.038 / 2.73 (0.111 / 3.42) | 0.032 / **2.61** (0.104 / 3.36) |
| v2 GT, 3-channel | 0.035 / 2.97 (0.118 / 3.55) | **0.027** / 2.75 (0.110 / 3.43) |
| v2 GT, inlier | 0.033 / 2.77 (0.113 / 3.47) | 0.032 / 2.62 (0.110 / 3.39) |

| held out: Welland | 0.25 px | 0.10 px |
|---|---|---|
| shipped | 0.119 / 1.53 (0.217 / 2.13) | **0.080 / 1.19** (0.122 / 1.41) |
| deploy | 0.131 / 1.61 (0.174 / 1.86) | 0.112 / 1.34 (0.129 / 1.45) |
| v1 GT, 3-channel | 0.115 / 1.36 (0.145 / 1.51) | 0.093 / 1.20 (0.118 / 1.34) |
| v2 GT, 3-channel | 0.096 / 1.26 (0.137 / 1.44) | 0.097 / 1.26 (0.119 / 1.41) |
| v2 GT, inlier | 0.123 / 1.37 (0.150 / 1.49) | 0.113 / 1.28 (0.134 / 1.40) |

Network level on the held-out site (scored on v2 GT):

- The v2 3-channel heads rank best on both folds. Temporal / stereo AUSE: Vobster 0.103 / 0.123 against 0.114 /
  0.154 for v1; Welland 0.116 / 0.450 against 0.120 / 0.482.
- Their NLL is better on Vobster (0.96 / 0.80 against 0.98 / 0.89) and a little worse on Welland's stereo (7.7
  against 6.1, from more tail).

1. **GT v2 against v1: a draw on trajectories, better ranking.** v2 has the better ATE on three of the four fold /
   floor cells, v1 the better frame-to-frame on Vobster; the gaps are within run-to-run noise (three EIVA-only fold
   runs per site spread by 5% (Welland) and 35% (Vobster) in ATE under the split protocol). Keep v2: the flow fine-tune trains on it.
2. **The inlier head is rejected.** It ranks marginally better, but its output to MAC-VO (Sigma / p) has more
   pixels beyond the 99% bound than the 3-channel head (tail > 9: temporal 4.2% against 3.5% on Vobster, stereo
   9.4% against 8.8% on Welland). It is level on Vobster and worse on Welland. Down-weighting by the inlier
   probability does not reach the confidently wrong pixels.
3. **The floor depends on the head.**
   - Shipped is best at 0.10 px. At 0.05 px it stays level on Welland (0.080 / 1.18) and gets worse on Vobster
     (0.044 / 3.65).
   - The EIVA + water folds keep improving to 0.05 px: Welland 0.085 / 1.17, Vobster 0.029 / 2.54.
   - Held-out TartanAir (8 seaside sequences) is level at both floors. Shipped ATE 1.35 / 1.32 / 1.31 m and EIVA
     + water 1.11 / 1.15 / 1.14 m at 0.25 / 0.10 / 0.05 px; RTE 0.0117-0.0120 and 0.0113-0.0116.
4. **What the tuned head adds over shipped, each at its best floor:**
   - Vobster: -24% ATE / -27% frame-to-frame.
   - Welland: level. The ATE ranking flips between the two protocols (0.118 against 0.122 m old, 0.085 against
     0.080 m split). Scale (0.995) and 10-frame drift (5.1% against 5.3%) are equal: Welland's 184 m dive does not
     separate the heads.
   - TartanAir: -16% ATE, RTE level.

**Recommendation** (revised 2026-10-23).
- Now, runtime only: `m_MinFlowCov` 0.10 px with the shipped head. Split protocol: Welland -33% ATE / -22%
  frame-to-frame, Vobster -25% / -6%; TartanAir level.
- With gate 6: an EIVA-tuned 3-channel head on GT v2 at 0.05 px. It gains on Vobster and TartanAir and is level on
  Welland.
  - The production head (`CovHeadEIVA_AllWater` on GT v2) is retrained once the flow fine-tune has decided which
    matcher it sits on.
  - The current `CovHeadEIVA_AllWater` (GT v1) is the fallback, with its engine already built and checked.
- Until gate 6, keep the shipped head in the runtime.
- The fp32-trained 25k head should not be deployed in fp16 / bf16: it breaks on `plane_tail1`.
- Hard stretches like `plane_tail2`'s (where the photogrammetry also lost the camera) break every
  head. How a head copes with one varies between training runs. Judge such stretches on several
  runs, or on perturbations (`cov_scale`), not on one MAC-VO run.

**Fine-tuning the matcher on EIVA (2026-10-23).** `FlowFTEIVA_<fold>` (`--training_mode flow`): the 25k run's
weights, the Twins image encoders frozen (`flow_trainable: decoder+cost`, 7.65 M trainable), fp32, batch 2 (batch 4
pages at 24 GB), 20k steps, lr 2.5e-5, the EIVA + water data of the head folds (GT v2). About 2.8 h per fold.

Mean flow error (px) on held-out data, deployed precision, original matcher -> fine-tuned:

| | fold W (held out: Vobster) | fold V (held out: Welland) |
|---|---|---|
| other EIVA site, temporal | 1.087 -> 0.989 (-9%) | 0.765 -> 0.710 (-7%) |
| other EIVA site, stereo | 0.910 -> 0.665 (-27%) | 3.67 -> 2.86 (-22%) |
| TartanAir val (15 seq.), temporal / stereo | -21% / -41% | -22% / -47% |
| seaside holdout, temporal / stereo | -21% / -33% | -20% / -39% |

The oracle's best 20% of pixels also improve (stereo on Vobster 0.096 -> 0.080 px), and TartanAir does not forget.
The head has to be retrained on the new matcher: the old head's ranking collapses on it (temporal AUSE 0.11 -> 0.23).

**But the trajectories break.** Fold W, matcher fine-tuned + v2 head retrained on it, held-out Vobster, split
protocol, 0.10 px: plane_mid 0.006 m (best so far), but plane_nose 0.20 m, plane_tail1 2.43 m, plane_tail2 0.57 m
(last night's v2 head: 0.028 / 0.033 / 0.038). Ruled out, each on the three failing dives:

- the floor (0.25 px: the same);
- precision (fp32 encoder / decoder: the same);
- the head (fine-tuned matcher + last night's head: breaks as well);
- head overconfidence (cov_scale 2: the same);
- the image resize (INTER_AREA as in training instead of the evaluation's antialiased bilinear: the same).

**Cause: the fine-tuned stereo hallucinates near surfaces in open water.** Instrumented MAC-VO run on plane_tail1
(`macvo/debug_kp.py`, `motion_check.py`):

- At the break (frames 251 -> 253 the estimated step is 25 / 12 mm against 83 / 46 mm GT), half of the 200 keypoints
  have no GT. Their flows are 37-62 px off the flow the true motion implies with their depth (median depth 0.8 m,
  the wreck is at ~2 m). sigma ~5 px: 1% of the weight, but 100% of the weighted squared residual.
- The keypoints with GT are accurate (flow 0.18 px, disparity 0.22 px median).
- Last night's model picks as many keypoints without GT, but at ~5 m depth they agree with the true motion to ~2 px.
- Dense disparity where the GT has none (open water, haze): original 14-17 px (~5 m), fine-tuned 40-87 px (0.9-1.9
  m), 66-76% nearer than 2.6 m. On the GT pixels both agree (24-36 px).

Open water has no target in the EIVA GT, so the fine-tune never saw it, and the network's stereo drifted there towards
the near-range disparities of everything it was supervised on. The network-level evaluation could not see it: it
scores only pixels with GT. Water-heavy dives break; plane_mid, where the wreck fills the view, improves.

**Teacher targets keep the training site's water, not a new site's.** `Scripts/Dataset/make_eiva_teacher.py` stores
the original network's flow and disparity per GT frame; `EIVA_NoIMU`'s `teacher` option fills every pixel without a
target with them (matcher fine-tunes only; the head trains without). Fold W with the teacher (`FlowFTEIVA_Welland_teacher`):

- On the GT pixels it keeps the gain: held-out Vobster temporal 0.970 px (-11%), stereo 0.698 px (-23%), best 20%
  stereo 0.076 px.
- On Welland's own water it stays at the original's disparity (17.7 / 21.3 px against 19.4 / 24.7 px; the plain
  fine-tune was at 12.9 / 14.1 px, so Welland's water was never the problem).
- On held-out Vobster's water it hallucinates as before: plane_tail1 #240 / #251 / #252 at 85 / 105 / 115 px
  (0.7-0.9 m; original 17 / 14 / 10 px, plain fine-tune 55 / 103 / 90 px).

So the failure is generalisation to an unseen kind of water, not drift on the training frames: fine-tuned on Welland's
bright haze, the stereo reads Vobster's dark green water as near. Each fine-tune handles the water of its own site
(fold V on plane_tail1 #251: 22 px), and only the original, trained on synthetic data, is robust to water it has not
seen. Colour is part of the trigger, not all of it: tinting plane_tail1 #251 towards Welland's haze halves the
hallucination (105 -> 50 px); tinting Welland #1200 green does not cause one (21.5 px).

**A far stereo target for open water generalises.** `EIVA_NoIMU`'s `water_disparity` (0.25 px, i.e. far: the
synthetic-sky convention) on Welland's open water (no GT within 24 px), the teacher elsewhere
(`FlowFTEIVA_Welland_teacherwater`). Open-water disparity on held-out plane_tail1, px (m):

| frame | original | plain fine-tune | teacher | teacher + far water |
|---|---|---|---|---|
| 240 | 16.9 (4.6) | 54.7 (1.4) | 85.1 (0.9) | 20.5 (3.8) |
| 251 | 14.1 (5.6) | 103.4 (0.8) | 105.0 (0.7) | 33.6 (2.3) |
| 252 | 10.1 (7.7) | 89.7 (0.9) | 114.8 (0.7) | 27.3 (2.9) |

The matching gain stays: held-out Vobster temporal 0.977 px (-10%), stereo 0.691 px (-24%), best 20% stereo 0.077
px. An explicit physical label carries over to water the network has not seen; copying the old network's output does
not. Trajectories with its retrained head, held-out Vobster, split protocol, ATE m / frame-to-frame mm:

| | plane_mid | plane_nose | plane_tail1 | plane_tail2 |
|---|---|---|---|---|
| shipped, 0.25 px | 0.018 / 2.35 | 0.078 / 5.52 | 0.060 / 2.92 | 0.048 / 4.12 |
| v2 head, original matcher, 0.25 px | 0.015 / 1.81 | 0.041 / 3.71 | 0.040 / 2.54 | 0.044 / 3.80 |
| plain fine-tune + head, 0.25 px | 0.007 / 1.46 | 0.195 / 9.80 | 2.414 / 47.3 | 0.533 / 19.0 |
| teacher + far water + head, 0.25 px | 0.007 / 1.33 | 0.054 / 4.46 | 2.317 / 45.0 | 0.058 / 5.57 |
| teacher + far water + head, 0.10 px | 0.005 / 1.29 | 0.054 / 4.10 | 2.326 / 45.3 | 0.051 / 5.40 |

plane_tail2 and plane_nose recover, plane_mid is the best result so far, but plane_tail1 still breaks, now later: a
~53 deg heading change between frames 340 and 370 (frame 251 is tracked). An instrumented run shows inputs consistent
with the true motion there: keypoints with GT 0.2-0.5 px off the flow the true motion implies, those without GT 1-3 px;
next-frame depth within ~1% (with GT) and mostly 2-10% (without). So this break is not the water hallucination and not
a frontend measurement error; it builds up in the pose graph (weighting / conditioning), out of reach of a
matcher-side check. The head's stereo overconfidence on fine-tuned matchers (tail > 9 11.6% against 4.4%) is the
model-side suspect.

**Mixed split: seeing part of a site's water is enough for its other dives.** `FlowFTEIVA_Mixed_teacher` (Welland +
Vobster plane_mid / plane_nose, teacher), held-out plane_tail1 / plane_tail2:

- open water stays far: plane_tail1 #240 / #251 / #252 at 18.7 / 16.7 / 15.2 px (original 16.9 / 14.1 / 10.1);
- flow error: temporal 0.884 px (-18%; Welland-only fine-tune -8%), stereo 0.719 px (-25%), best 20% stereo 0.073 px;
- no trajectory breaks, but not better than the v2 head on the original matcher (split protocol, ATE m / mm):

  | | plane_tail1 0.25 px | plane_tail2 0.25 px | plane_tail1 0.10 px | plane_tail2 0.10 px |
  |---|---|---|---|---|
  | shipped | 0.060 / 2.92 | 0.048 / 4.12 | 0.040 / 2.68 | 0.040 / 4.08 |
  | v2 head, original matcher (Welland only) | 0.040 / 2.54 | 0.044 / 3.80 | 0.033 / 2.49 | 0.038 / 3.73 |
  | mixed fine-tune + head | 0.049 / 2.68 | 0.036 / 4.24 | 0.038 / 2.74 | 0.038 / 4.63 |

So far no matcher fine-tune turns its 20-25% better matching into better trajectories; frame-to-frame error is worse.
The fine-tuned matchers' heads are overconfident (cross-fitting, `frame_range`, is the next fix), and the mixed
head on the original matcher (`CovHeadEIVA_MixedWater_v2`, queued) separates the matcher's contribution from the
head's extra Vobster data.

Also under test (model side only):
- synthetic underwater augmentation of TartanAir (`underwater_aug`, `Train/MatchingNet/underwater.py`: the revised
  underwater image formation model of Akkaynak & Treibitz 2018 from the GT depth, random water per pair);
- covariance negatives on open water (`water_sigma`, `loss.water_hinge`): sigma pushed above 32 px where there is
  no correspondence. The v2 head's water sigma is 19-26 px at the median but 4-5 px at the 10th percentile, the end
  the selector picks from (the keypoints that broke plane_tail1 had ~5 px);
- a mixed split (Welland + Vobster plane_mid / plane_nose, plane_tail1 / plane_tail2 held out).

Any matcher fine-tune needs the water check on a new site before release (`Scripts/Dataset/check_eiva_water.py`).

**The head on a fine-tuned matcher is overconfident.** Retrained on the fine-tune's training frames, it learns errors
smaller than the matcher makes on new data (stereo EPE on the training site -44%, on the held-out site -22%):
held-out stereo tail > 9 at 12-13% (v2 head on the original matcher: 4-9%), median calibration 2.4 (1.39 calibrated),
sigma^2 ~1.7x too small. `EIVA_NoIMU`'s `frame_range` allows cross-fitting: fine-tune the matcher on one set of
blocks of a site, train the head on the others.

**Open.**
1. Gate 6: the C++ runtime has to read 3 covariance channels. It must drop its `exp(2 * cov)` and
   feed sigma_uv into `covariance_2to3` and the selector.
2. Near-range overconfidence at keypoints remains. On `Hard_P003` (`StereoTailAnalysis.py
   --precision fast`) both EIVA heads pick ~2x the confidently wrong stereo keypoints of shipped:
   - production v2: 49, 86% nearer than 5 m;
   - EIVA + water: 56, 91% nearer than 5 m;
   - shipped 27, deploy 24.

   The EIVA + water trajectory is nonetheless robust there. Lowering that tail is the next head
   improvement, e.g. more near-range TartanAir pairs, or a near-range stereo weight.
3. Fold W's sigma is too wide on Vobster: Welland's GT is noisier. Down-weighting that noise
   could help:
   - a per-frame GT noise floor from `check_eiva_gt.py`, or
   - training only on frames whose ratio is within 1%.
4. More EIVA dives with photogrammetry would allow a real held-out test of the production head.
   `make_eiva_gt.py` + `check_eiva_gt.py` turn any dive with `pose_gt.txt` + `pointcloud_gt.txt`
   into training data.
