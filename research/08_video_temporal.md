# Research: Video Frame Extraction & Temporal Operations Atoms

## Goal

Find best-in-class, pure-function implementations for video processing
primitives: frame extraction, temporal aggregation, and tracking utilities.
Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (~16 stages)

- Video frame extraction at fixed FPS (DFDC Deepfake, DFL Bundesliga, Great Barrier Reef)
- Uniform frame sampling from video (DFDC Deepfake — 32 frames)
- Frame-level prediction temporal averaging (DFDC Deepfake, DFL Bundesliga)
- Median filter on temporal prediction sequences (DCASE SED)
- Sliding window over video frames (DFL Bundesliga)
- 2.5D frame stacking — 3 consecutive frames as RGB channels (DFL Bundesliga)
- Optical flow estimation concept (video CDGs)
- Multi-object tracking / track association (NFL Helmet Assignment)
- Frame-to-video aggregation — max/mean pooling over frames (DFDC)
- Temporal unrolling — expanding aggregated data back to timesteps (M5 Accuracy)
- DSL functional primitives for grid transforms (ARC Program Synthesis)

## What to research

### 1. Video frame extraction (OpenCV-free where possible)
- Extract frames at fixed interval from video file metadata
- `sample_frame_indices(total_frames: int, target_fps: float, video_fps: float) -> NDArray[int]`
- Note: actual pixel decoding requires OpenCV/ffmpeg — the atom should compute
  WHICH frames to extract (index selection), not do the I/O
- Uniform sampling: `uniform_sample_indices(total_frames: int, n_samples: int) -> NDArray[int]`

### 2. Frame-level prediction aggregation
- Average/max/vote over frame-level predictions for video-level output
- `temporal_mean_pool(frame_predictions: NDArray) -> NDArray`
- `temporal_max_pool(frame_predictions: NDArray) -> NDArray`
- `temporal_attention_pool(frame_predictions: NDArray, weights: NDArray) -> NDArray`

### 3. Temporal median filter
- Median filter on 1D signal of frame predictions
- `temporal_median_filter(predictions: NDArray, kernel_size: int) -> NDArray`
- Source: scipy.ndimage.median_filter (BSD)

### 4. Sliding window
- Extract overlapping windows from temporal sequence
- `sliding_windows(sequence: NDArray, window_size: int, stride: int) -> NDArray`
- Pure numpy strided tricks or explicit loop

### 5. Consecutive frame stacking
- Stack N adjacent frames as channels
- `stack_adjacent_frames(frames: NDArray, center_idx: int, num_adjacent: int) -> NDArray`
- Same concept as 2.5D medical slices but for video

### 6. Track association (Hungarian algorithm)
- Assign detections to existing tracks via IoU cost matrix
- `hungarian_assignment(cost_matrix: NDArray) -> list[tuple[int, int]]`
- Source: scipy.optimize.linear_sum_assignment (BSD)
- Wrapper: `associate_detections_to_tracks(track_boxes, detection_boxes, iou_threshold) -> assignments`

### 7. Temporal unrolling
- Expand aggregated predictions back to original timestep granularity
- `temporal_unroll(aggregated: NDArray, group_sizes: NDArray) -> NDArray`
- np.repeat with group-specific counts

## Research questions

1. For frame sampling: what's the best strategy for uniform sampling?
   (linspace vs random vs keyframe-aware)
2. For temporal pooling: mean vs max vs attention-weighted — which is most common?
3. For tracking: is scipy's linear_sum_assignment sufficient, or do we need
   the SORT/DeepSORT algorithm? (Recommend: just the assignment atom,
   SORT is orchestration)
4. What contracts are natural? (n_samples <= total_frames, window_size > 0,
   kernel_size odd for median filter)
5. Should video I/O (actual pixel decoding) be an atom or external?
   (Recommend: external — atom handles index selection and aggregation)

## Output format

Concept types: `data_extraction` for frame selection, `analysis` for
aggregation, and `signal_filter` for temporal filtering.

For each candidate atom, provide:
```
Name: uniform_frame_indices
Description: Select uniformly spaced frame indices from a video length.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: data_extraction, analysis, or signal_filter
Signature: (num_frames: int, num_samples: int) -> NDArray
Pure function boundary: frame counts, arrays, and explicit parameters in,
  selected indices or aggregated arrays out; no video decoding, file I/O, model
  state, GPU state, or global RNG.
Contracts:
  - require: num_frames > 0
  - require: num_samples > 0
  - ensure: all returned indices are in [0, num_frames)
Witness: 10-frame video and 4 samples; verify deterministic indices and bounds.
Dependencies: numpy only preferred; scipy acceptable for interpolation/filtering
CDG stages covered: dfl_bundesliga/frame_sampling, great_barrier_reef/temporal_nms, ...
```
