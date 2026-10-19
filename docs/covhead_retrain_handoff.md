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
