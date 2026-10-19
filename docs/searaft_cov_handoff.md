# SEA-RAFT + MAC-VO covariance head: handoff brief

Goal: replace the FlowFormerCov frontend with SEA-RAFT plus a retrained MAC-VO covariance
head, inside this repository, and ship it as a TensorRT engine for `MACVO_TRT`. Success means
equal or better VO accuracy on TartanAir v2 test and a materially faster frontend.

## Decisions already made

- The host repository is MAC-VO-MCH, not a SEA-RAFT fork. Everything except the matching
  network already exists here: covariance head, NLL loss, `cov`-mode trainer, TartanAir loader
  with ground-truth flow, `IFrontend` contract, export/build scripts, parity harness.
- Target deployment: the C++ pipeline in `C:\src\git\eivacom\MACVO_TRT` (TensorRT 10,
  strongly-typed fp16/bf16 engine, batch-2 stereo+temporal call plus a batch-1 keyframe call,
  704x704 input, images in [0, 1], outputs `flow`, `flow_raw`, `cov` (log sigma), `cov_raw`).
- Resolution stays 704x704. No quantisation.
- Accuracy judge: full-pipeline RTE/ROE on TartanAir v2 test (28 sequences,
  `Config/Sequence/Training_Dataset/TartanAirV2_Test.yaml`) for both frontends. Trajectory APE on
  the in-house sequences has a ~15% engine-to-engine noise band and cannot rank candidates.
- Checkpoint choice: a SEA-RAFT `Tartan-C-T-*` weight (TartanAir + FlyingChairs + FlyingThings).
  The `TSKH` variants include Sintel/KITTI/HD1K, which are non-commercial.

## Where things are

Template to copy (FlowFormer version of exactly this integration):

| piece | file |
|---|---|
| vendored backbone | `Module/Network/FlowFormer/core/*` |
| model wrapper with covariance decoder | `Module/Network/FlowFormerCov/flownet.py`, `covhead.py`, `__init__.py` (`build_flowformer(cfg, enc_dtype, dec_dtype)`) |
| covariance head | `covhead.py`: `CovHead` (4 convs 128→256→128→64→2), `CovUpdateBlock` (SepConvGRU hidden 128 on a 384-ch input, `CovHead`, upsample mask head) |
| frontend adapters | `Module/Frontend/Frontend.py`: `FlowFormerCovFrontend`, `CUDAGraph_FlowFormerCovFrontend`; `inference_2_depth`, `inference_2_match` |
| output contracts | `Module/Frontend/Matching.py` (`IMatcher.Output.from_partial_cov`, zero σ_uv), `Module/Frontend/StereoDepth.py` (`disparity_to_depth`, `disparity_to_depth_cov`) |
| trainer | `Train/MatchingNet/train_flowformer.py` (DataParallel, AdamW, OneCycleLR; `--training_mode cov` freezes all but `memory_decoder.cov_update`) |
| losses | `Train/MatchingNet/loss.py`: `sequence_loss` → `flow_loss`, `cov_loss` (γ-weighted Gaussian NLL per iteration), unused `depth_loss` |
| evaluation | `Train/MatchingNet/evaluate.py` (TartanAir v2 EPE/NLL) |
| train configs | `Config/Train/Demo.yaml` (fine-tune from `Model/MACVO_FrontendCov.pth`), `Config/Train/FlowFormerCovKITTIthings.yaml` (full recipe: batch 6, 480x640, 120k steps, lr 12.5e-5) |
| datasets | `Config/Sequence/Training_Dataset/TartanAir_Train.yaml` (169 Easy seqs), `TartanAir_TrainHard.yaml` (172 Hard), `TartanAirV2_Test.yaml` |
| export / build | `Export.py` (`--mode fast`, `--decoder-depth`, onnxsim fallback, `do_constant_folding=False`), `Build.py` (strongly typed, opt level 5, weight streaming, timing cache) |
| harness | `Scripts/Parity/`: `make_reference.py` (fp32 batch-1 reference set), `compare.py` (torch/ONNX/TRT vs reference), `test_exact.py`, `profile_agg.py`, `iterations.py` |
| runtime config | `Config/Experiment/MACVO/MACVO_Fast.yaml` → `frontend:` block |

Data loader facts (`DataLoader/Dataset/TartanAir.py`): per sequence it needs `image_left/`,
`image_right/`, `flow/` (`*_flow.png` with `compressed: true`, or `*_flow.npy` + mask with
`compressed: false`), `pose_left.txt`. Trainer uses only left temporal pairs
(`frameData.cur.stereo.imageL`, `frameData.nxt.stereo.imageL`) with GT flow and flow mask.
The dataset filter in the trainer accepts only `TartanAir_NoIMU` / `TartanAirv2_NoIMU`.

Environment: the `Module` package imports the full pipeline (DataLoader → scipy). On the
laptop that stack is broken (numpy 2.4 vs scipy); the export and harness import
`Module/Network` directly via `sys.path`. Training needs the proper environment (repo
Dockerfile, CUDA 12.4). SEA-RAFT upstream targets torch 2.2 / py3.10; it has no exotic deps.

## SEA-RAFT facts that shape the port

Source: `princeton-vl/SEA-RAFT` (`core/raft.py`, `core/update.py`, `core/corr.py`,
`core/extractor.py`), BSD-3. Paper arXiv 2405.14793.

- Encoders: truncated ImageNet ResNet with FPN (`ResNetFPN`); `fnet` on each frame (3 ch),
  `cnet` on both frames stacked (6 ch) → split into `net` (hidden, dim 128) and `context` (128).
- Initial flow is regressed directly from `cnet` features (`flow_head` → 6 channels: 2 flow +
  4 Mixture-of-Laplace "info"); RAFT's zero init is gone.
- Update block (`BasicUpdateBlock`): no GRU; `num_blocks` ConvNeXt blocks refine `net` from
  `cat[net(128), context(128), motion(128)]` (384 ch → 128). Motion encoder consumes the
  4-level correlation lookup (`corr_levels=4`, `radius=4`, 324 channels) and the flow.
- Per iteration: `flow_update = flow_head(net)` (6 ch), `upsample_weight(net)` (64*9 mask).
  Outputs lists of `iters + 1` predictions (initial + each refinement).
- Iterations: trained with 4; M infers with 4, L (same weights) with 12.
- Its own uncertainty (MoL α, log β₂) is shared across x and y: isotropic. Ignore it.
- Speed: 487 GMAC (M) vs 1715 for FlowFormer; 71 ms vs 336 ms at 540x960 on a 3090 (PyTorch).
- Check the input scaling in `raft.py` (RAFT-family code usually expects [0, 255] and
  normalises internally); the engine contract here is [0, 1].

## Plan

### 1. Vendor SEA-RAFT
`Module/Network/SEARAFT/` with `core/*` files, upstream commit hash in a header comment,
LICENSE file copied. Fix relative imports. Weights from Hugging Face `MemorySlices/...`
(`Tartan-C-T-*`), stored under `Model/`.

### 2. Model wrapper `Module/Network/SEARAFTCov/`
- `build_searaft(cfg, enc_dtype, dec_dtype)` mirroring `build_flowformer`.
- Class running SEA-RAFT's loop and, each iteration, feeding `inp = cat[net, context, motion]`
  (384 ch, same width as today's `inp_cat`) into a covariance block derived from
  `CovUpdateBlock`, with its hidden state initialised from `context` (tanh) as in
  `MemoryCovDecoder`. Accumulate log σ additively across iterations as today.
- Return per-iteration `flow` and `cov` lists in training; `(flow_up, flow_raw), (cov_up, cov_raw)`
  in eval, so `Export.py`'s wrapper and the C++ output contract stay unchanged.
- `inference(image1, image2)` returning `flow, exp(2 * cov)`; strict `load_state_dict` with an
  explicit allow-list for unused SEA-RAFT tensors (its MoL head can stay loaded and unused).
- Decoder depth configurable (`decoder_depth`), default 4 to match SEA-RAFT's training.

### 3. Head changes worth making while retraining (ranked)
Items 1 and 2 exist for FlowFormerCov on branch `covhead-retrain`: `cov_entries` in `covhead.py`,
`cov_nll` in `loss.py`, `stereo_target` in `Train/MatchingNet/utils.py`. Reuse them; see
`docs/covhead_retrain_handoff.md` sections 10 and 11.
1. Full 2x2 covariance: predict a Cholesky factor (3 channels) and use the full 2x2 Gaussian
   NLL. `from_partial_cov` and the selector/`MatchCovariance` already carry σ_uv.
2. Stereo supervision: add (L, R) batches with GT disparity f·b/depth from TartanAir
   `depth_left`; `loss.py` has an unused `depth_loss` scaffold. This directly trains the
   disparity σ that drives the 3D covariance.
3. Not β-NLL while the matcher is frozen. β-NLL keeps a jointly learned mean from
   under-fitting hard points. With only σ learned, it merely reweights pixels by σ^(2β),
   towards the uncertain ones MAC-VO discards (`docs/covhead_retrain_handoff.md` section 10).
   Consider it only if the flow is trained too.
4. Lighter covariance branch (e.g. hidden 64): the cov GRU currently costs as much as the
   flow GRU.
5. Rather than a validity channel trained on `flow_mask`, which only knows occlusion, learn an
   inlier probability from the actual errors with a mixture likelihood
   (`docs/covhead_retrain_handoff.md` section 11). It also covers matching failures in
   textureless or repetitive regions.
Keep: additive log-σ recurrence, per-iteration γ weighting, 3D propagation in
`Module/Covariance/Project2to3.py`.

### 4. Trainer
Generalise `train_flowformer.py`: model factory and trainable-parameter selection from
config instead of the hardcoded `build_flowformer` / `memory_decoder.cov_update`. Align the
`iters + 1` flow list with the cov list in `sequence_loss`. Start with `cov` mode (frozen
SEA-RAFT), batch 6 at 480x640, a few tens of thousands of steps; one 24 GB GPU, under a
GPU-day. Data: a 6–8 environment subset of TartanAir v1 Easy+Hard first (~150–300 GB with
`--rgb --flow` via `castacks/tartanair_tools`), extend if validation asks for it.

### 5. Frontend
`SEARAFTCovFrontend` next to `FlowFormerCovFrontend` in `Module/Frontend/Frontend.py`
(same batch layout: `input_A = cat[t2.L, t1.L]`, `input_B = cat[t2.R, t2.L]`), config block
under `frontend:` in a copy of `MACVO_Fast.yaml`.

### 6. Gates, in order
1. Wrapper exactness: flow from the wrapper equals raw SEA-RAFT flow (fp32, random and
   real pairs).
2. `Train/MatchingNet/evaluate.py` on TartanAir v2 test: EPE and NLL vs the FlowFormerCov
   numbers.
3. Full pipeline on TartanAir v2 test (`python MACVO.py --odom <cfg> --data <seq>` per
   sequence, `python -m Evaluation.EvalSeq --spaces ...`): RTE/ROE for both frontends. This
   is the decision gate.
4. Export: `python Export.py --mode fast` (adapt the wrapper import), `python Build.py
   --mode fast --timing-cache runs/timing.cache`, `trtexec --loadEngine ... --shapes=image_1:2x3x704x704,image_2:2x3x704x704`.
   Reference for comparison on the RTX 2000 Ada laptop: FlowFormerCov v4 engine 278 ms
   (batch 2) / 146 ms (batch 1); on a 4090, 56 ms (batch 2).
5. Parity: new fp32 reference with `make_reference.py` for the new model, then
   `compare.py --trt <plan> --pairing b2`.
6. In-house trajectory smoke test only (`Evaluate.py --plan <plan> --skip 1 2 3`).

## Risks

- VO accuracy: FlowFormer is stronger on Sintel final; the keyframe pass (large displacement)
  is where SEA-RAFT with 4 iterations may lose. If so try 8–12 iterations before giving up.
- Backend tuning: `MACVO_TRT` LM/iSAM2 covariance scales and robust thresholds were tuned to
  FlowFormerCov's σ; expect re-tuning.
- Export: `grid_sample` x4 levels per iteration, BatchNorm folding, ConvNeXt LayerNorm over
  channels in NCHW (permutes). Keep opset 18; onnxsim may fail (fallback exists).
- Engines are GPU- and TensorRT-version specific; build on the target machine, keep the
  timing cache with the engine.

## Measured context from the FlowFormerCov work (for calibration of expectations)

- Exact rewrites of the FlowFormerCov graph gave 374 → 278 ms (batch 2) on the laptop at
  equal accuracy; details in `git log f0bd298..HEAD` and `Scripts/Parity`.
- The decoder is not converged at 12 iterations but the residual on selectable pixels is
  0.07 px (stereo/temporal) and 0.35 px (kf); fewer iterations double the error; warm-start
  initialisation hurts (head is out of distribution for non-zero initial flow).
- bf16/fp16 inference deviates ~0.8 px mean EPE from fp32 on all pixels, concentrated in
  low-texture regions.

## Suggested opening prompt for a new session

"Read docs/searaft_cov_handoff.md and follow its plan. Start with step 1 (vendor SEA-RAFT
under Module/Network/SEARAFT) and step 2 (SEARAFTCov wrapper with the covariance block),
verify wrapper exactness against raw SEA-RAFT, then stop and report before touching the
trainer."
