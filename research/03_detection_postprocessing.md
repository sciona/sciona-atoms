# Research: Object Detection Post-Processing Atoms

## Goal

Find best-in-class, pure-function implementations for object detection
utilities: NMS, WBF, box encoding, anchor generation, IoU computation.
Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (~28 stages)

- Non-Maximum Suppression / NMS (Great Barrier Reef, DFL Bundesliga, RSNA series)
- Weighted Box Fusion / WBF (Global Wheat, RSNA Pneumonia, Great Barrier Reef)
- Soft-NMS (Sartorius Cell Segmentation)
- IoU / GIoU computation (many detection CDGs)
- Anchor generation (Passenger Screening, DSB2017)
- Box encoding/decoding — delta format (RSNA series)
- Confidence thresholding (BirdCLEF, DFL Bundesliga)
- 1D NMS / peak finding in probability signals (DFL Bundesliga)
- Multi-object tracking association (NFL Helmet Assignment)
- Face detection bounding box extraction (DFDC Deepfake)
- Box ensembling across models (BYU Flagellar Motors)
- Text span Weighted Box Fusion — 1D adaptation (Feedback Prize Writing)
- Bounding box generation from instance masks (Sartorius Cell Segmentation)

## What to research

### 1. Non-Maximum Suppression (NMS)
- Standard greedy NMS: sort by confidence, suppress overlapping boxes
- Pure function: `nms(boxes: NDArray, scores: NDArray, iou_threshold: float) -> NDArray[indices]`
- Source: torchvision (BSD), or the pure numpy reference implementations
- Variants: class-aware NMS (per-class suppression)

### 2. Soft-NMS
- Bodla et al. 2017 — decay scores instead of hard suppression
- `soft_nms(boxes, scores, iou_threshold, sigma, method) -> (boxes, updated_scores)`
- Gaussian and linear decay variants

### 3. Weighted Box Fusion (WBF)
- Solovyev et al. 2021 — merge overlapping boxes by weighted average
- Source: https://github.com/ZFTurbo/Weighted-Boxes-Fusion (MIT)
- `wbf(boxes_list, scores_list, labels_list, iou_threshold, skip_box_thr) -> (boxes, scores, labels)`
- Key: averages coordinates of matched boxes weighted by confidence

### 4. IoU / GIoU computation
- `iou_matrix(boxes_a: NDArray, boxes_b: NDArray) -> NDArray` — pairwise IoU
- `giou(boxes_a, boxes_b) -> NDArray` — Generalized IoU (Rezatofighi et al. 2019)
- Pure numpy: intersection area / union area

### 5. Anchor generation
- Grid-based anchor generation for single-shot detectors
- `generate_anchors(feature_map_size, stride, scales, ratios) -> NDArray[anchors]`
- Source: torchvision FPN anchor generator or Detectron2

### 6. Box encoding/decoding
- Delta encoding: `encode_boxes(anchors, gt_boxes) -> deltas`
- Delta decoding: `decode_boxes(anchors, deltas) -> boxes`
- Standard (tx, ty, tw, th) parameterization from Faster R-CNN

### 7. Confidence thresholding
- `threshold_detections(boxes, scores, threshold) -> (filtered_boxes, filtered_scores)`
- Simple but important as a composable atom

### 8. 1D NMS / peak detection in signals
- `nms_1d(signal: NDArray, min_distance: int, threshold: float) -> NDArray[peak_indices]`
- Adaptation of NMS for temporal signals (action detection in video)

### 9. Mask-to-box conversion
- `masks_to_boxes(binary_masks: NDArray) -> NDArray[boxes]`
- Bounding box from binary mask via nonzero coordinate extrema

## Research questions

1. What is the pure numpy implementation of each operation?
   (No CUDA, no torchvision C++ ops — just array operations)
2. For NMS: what is the O(n^2) reference implementation? Is there a faster numpy path?
3. For WBF: what is the exact merging algorithm? (cluster formation, weighted averaging)
4. What contracts are natural? (boxes in [0,1] or absolute coords, scores in [0,1],
   IoU threshold in (0,1), output boxes valid)
5. What are the edge cases? (no detections, all suppressed, single box, identical boxes)

## Output format

All atoms should use concept_type `analysis` or `signal_filter` depending on context.

For each candidate atom, provide:
```
Name: nms
Description: Greedy non-maximum suppression over boxes and confidence scores.
Source: URL to the best reference implementation or paper
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: analysis or signal_filter
Signature: (boxes: NDArray, scores: NDArray, iou_threshold: float) -> NDArray
Pure function boundary: boxes, scores, and explicit thresholds in, selected
  indices or filtered detections out; no model state, GPU state, global RNG, or
  file I/O.
Contracts:
  - require: boxes.shape == (n, 4)
  - require: scores.shape == (n,)
  - require: 0 <= iou_threshold <= 1
  - ensure: returned indices are valid indices into boxes
Witness: three boxes with one highly overlapping pair; verify the lower-score
  overlapping box is suppressed.
Dependencies: numpy only preferred; scipy acceptable only if justified
CDG stages covered: great_barrier_reef/nms, global_wheat/wbf, ...
```
