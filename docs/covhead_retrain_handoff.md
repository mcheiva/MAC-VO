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
`compressed: false`), `pose_left.txt`; `depth_left/` only if `gtDepth: true`. The trainer
accepts only `TartanAir_NoIMU` / `TartanAirv2_NoIMU` dataset types and trains on left-camera
temporal pairs (`frameData.cur.stereo.imageL`, `frameData.nxt.stereo.imageL`, `gt_flow`, `flow_mask`).

## 3. Environment

Training imports the full `Module` package (DataLoader, scipy, pypose). Use the repo
Dockerfile (`Docker/Dockerfile`, CUDA 12.4) or a clean venv with `requirements.txt` plus
torch/torchvision for your CUDA. One 24 GB GPU is enough (the shipped recipe used 16 GB at
batch 6; `cov` mode uses less). Export and the parity harness import `Module/Network`
directly and also work in a lighter env (torch, tensorrt, onnx, onnxsim, numpy, opencv).

## 4. Data

TartanAir v1 via `castacks/tartanair_tools` (`download_training.py --rgb --flow`, add
`--depth` for stereo supervision, `--only-left/--only-right` not applicable since both cameras
are needed). Start with a 6–8 environment subset of Easy + Hard (~80 sequences, ~150–300 GB
with `--rgb --flow`); the 20 environments used by the authors are listed in
`Config/Sequence/Training_Dataset/TartanAir_Train.yaml` / `TartanAir_TrainHard.yaml`. Official
flow is `.npy`: set `compressed: false` in the sequence yaml (or convert to the 16-bit PNG form
`TartanAirGTFlowDataset.flow16to32` expects). Edit every `root:` to the local path.

TartanAir v2 test (`TartanAirV2_Test.yaml`, 28 sequences, `TartanAirv2_NoIMU`) via the
`tartanair` pip package: RGB + poses for the pipeline evaluation, flow if you also want
`evaluate.py` numbers.

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
config value (β = 0 recovers plain NLL).

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
