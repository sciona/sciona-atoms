# Research: Specialized Loss Function Atoms

## Goal

Find best-in-class, pure-function implementations for loss functions used in
Kaggle competition pipelines that don't exist in the current atom library.
Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (~15 stages)

- Quadratic Weighted Kappa loss (Aptos Blindness, PetFinder)
- CTC loss (ASL Fingerspelling, Bengali Speech)
- Focal loss (Great Barrier Reef, RSNA series)
- Lovasz-Softmax loss (HubMap Kidney, TGS Salt)
- Dice loss / soft Dice (segmentation CDGs)
- CRPS — Continuous Ranked Probability Score (NFL Big Data Bowl)
- Contrastive loss / triplet loss (Shopee, Facebook Image Similarity)
- Label smoothing cross-entropy (MoA Prediction, others)
- Multi-task weighted loss (Bengali Grapheme — 3 heads with different weights)
- Multimodal NLL loss with mode selection (Lyft Motion)
- Weighted BCE with demographic groups (Jigsaw Toxicity)

## What to research

### 1. Quadratic Weighted Kappa (QWK) loss
- Differentiable QWK for ordinal classification
- Source: competition kernels, QWK implementations in sklearn + differentiable variants
- Pure function: `qwk_loss(predictions: NDArray, targets: NDArray, num_classes: int) -> float`
- Note: sklearn has `cohen_kappa_score` but we need the differentiable loss variant

### 2. CTC loss (Connectionist Temporal Classification)
- Standard CTC forward algorithm (Graves et al. 2006)
- Pure numpy implementation of the forward pass (alpha computation)
- Source: PyTorch's CTCLoss reference, warp-ctc, or numpy implementations
- Function: `ctc_loss(log_probs: NDArray, targets: NDArray, input_lengths: NDArray, target_lengths: NDArray) -> float`
- The blank-label handling and log-space computation

### 3. Focal loss
- Lin et al. 2017 (RetinaNet paper)
- `focal_loss(predictions: NDArray, targets: NDArray, alpha: float, gamma: float) -> float`
- Simple: `-alpha * (1-p)^gamma * log(p)` for positives
- Both binary and multiclass variants

### 4. Lovasz-Softmax loss
- Berman et al. 2018 — optimal submodular extension of Jaccard/IoU
- The Lovasz extension + softmax probability inputs
- Pure function: `lovasz_softmax_loss(probabilities: NDArray, targets: NDArray) -> float`
- Source: https://github.com/bermanmaxim/lovasz-softmax (MIT)

### 5. Dice loss / soft Dice
- V-Net paper (Milletari et al. 2016)
- `dice_loss(predictions: NDArray, targets: NDArray, smooth: float) -> float`
- Formula: `1 - (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)`

### 6. CRPS (Continuous Ranked Probability Score)
- NFL Big Data Bowl metric
- `crps_score(cdf_predictions: NDArray, true_values: NDArray) -> float`
- Integral of squared difference between predicted CDF and step function

### 7. Contrastive / Triplet loss
- Contrastive: `contrastive_loss(embedding_a, embedding_b, label, margin) -> float`
- Triplet: `triplet_loss(anchor, positive, negative, margin) -> float`
- Source: PyTorch implementations, FaceNet paper

### 8. Label smoothing cross-entropy
- `label_smoothing_ce(logits: NDArray, targets: NDArray, epsilon: float) -> float`
- Simple: mix one-hot targets with uniform distribution

### 9. Multi-task weighted loss
- `weighted_multitask_loss(losses: list[float], weights: list[float]) -> float`
- Simple weighted sum but important as a composable atom

## Research questions

1. For each loss, what is the pure numpy forward-pass implementation?
   (No autograd, no GPU tensors — just the mathematical computation)
2. What are natural contracts? (e.g., focal loss gamma >= 0, dice smooth > 0,
   CTC input_lengths >= target_lengths)
3. What's the relationship between the loss value and its inputs?
   (monotonicity, bounds, behavior at extremes)
4. Are there numerically stable implementations? (log-sum-exp tricks for CTC,
   clamping for focal loss)
5. What witness values are natural? (perfect predictions -> 0 loss,
   random predictions -> known bound)

## Output format

For each candidate atom, provide:
```
Name: qwk_loss
Description: Differentiable quadratic weighted kappa loss for ordinal
  classification probabilities and integer targets.
Source: URL to the best reference implementation or paper
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: loss_function
Signature: (predictions: NDArray, targets: NDArray, num_classes: int) -> float
Pure function boundary: numeric arrays and explicit parameters in, scalar loss
  out; no autograd requirement, GPU state, global RNG, or file I/O.
Contracts:
  - require: predictions.shape[0] == targets.shape[0]
  - require: predictions.shape[1] == num_classes
  - require: all targets are integers in [0, num_classes)
  - ensure: result is finite
Witness: small probability matrix and target vector with a known expected loss;
  include edge cases such as perfect predictions.
Dependencies: numpy only preferred; scipy/torch acceptable only if justified
CDG stages covered: aptos_blindness/qwk_loss, petfinder/qwk_metric, ...
```
