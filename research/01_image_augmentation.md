# Research: Image Augmentation & Test-Time Augmentation Atoms

## Goal

Find best-in-class, pure-function implementations for image augmentation
primitives used across Kaggle competition pipelines. These will become atoms
in `sciona-atoms-dl`.

## CDG stages this research covers (34 stages)

These stages across 20+ CDGs need augmentation atoms:

- CutMix (Bengali Grapheme, Cassava Leaf, Global Wheat, others)
- MixUp (Bengali Grapheme, Melanoma, others)
- GridMask / Cutout (Bengali Grapheme, others)
- Geometric transforms (rotation, flip, scale, shift — many CDGs)
- Color augmentation (brightness, contrast, hue, saturation)
- Test-Time Augmentation averaging (Alaska2, Cassava, TGS Salt, many CDGs)
- Image resize/pad to target size (many CDGs)
- Image normalization (ImageNet mean/std, per-channel, Ben Graham method)
- Fold ensembling / prediction averaging across CV folds
- Heavy augmentation composition (Albumentations Compose-style)

## What to research

For each augmentation primitive, find:

### 1. CutMix
- Canonical source: the original paper implementation (Yun et al. 2019)
- Pure function: `cutmix(image_a, image_b, label_a, label_b, alpha) -> (mixed_image, mixed_label)`
- The bbox sampling from Beta distribution, pixel replacement, label interpolation

### 2. MixUp
- Canonical source: Zhang et al. 2018
- Pure function: `mixup(image_a, image_b, label_a, label_b, lam) -> (mixed_image, mixed_label)`
- Simple convex combination: `lam * a + (1-lam) * b`

### 3. Cutout / GridMask
- Cutout: DeVries & Taylor 2017 — zero out a random rectangle
- GridMask: Chen 2020 — structured grid-based masking
- Pure functions that take image + mask parameters, return masked image

### 4. Geometric transforms (pure numpy/scipy)
- Horizontal/vertical flip: `np.flip`
- Random crop + resize: crop coordinates + interpolation
- Affine transform: rotation + scale + translation matrix
- Look at Albumentations source for the numpy implementations underneath

### 5. Color augmentation
- Brightness/contrast: linear transform `alpha * pixel + beta`
- Hue/saturation shift: convert to HSV, shift, convert back
- Channel shuffle, grayscale conversion
- Ben Graham's retinal preprocessing (subtract local average color)

### 6. Test-Time Augmentation
- Apply N augmentations to test image, predict each, average predictions
- Pure function: `tta_average(predictions_list) -> averaged_prediction`
- Geometric TTA: flip + predict + unflip + average
- Multi-crop TTA: crop N regions, predict each, average

### 7. Image normalization
- ImageNet normalization: `(pixel - mean) / std` per channel
- Per-image normalization: zero-mean unit-variance per image
- Min-max scaling to [0, 1]

## Research questions

1. What are the pure numpy implementations underneath Albumentations?
   (Albumentations wraps OpenCV — find the equivalent numpy/scipy paths)
2. For each transform, what is the minimal function signature?
   (image ndarray in, image ndarray out, parameters explicit)
3. What natural contracts exist? (e.g., CutMix output has same shape as input,
   MixUp lambda in [0,1], normalized image has zero mean)
4. What are the licenses? (Albumentations = MIT, torchvision = BSD)
5. Are there edge cases to handle? (single-channel vs RGB, float vs uint8)

## Output format

For each atom, provide:
```
Name: cutmix_apply
Description: Apply CutMix augmentation by replacing a random rectangular region
  of image_a with the same region from image_b, interpolating labels by area ratio.
Source: https://github.com/clovaai/CutMix-PyTorch (MIT)
License: MIT
Concept type: sampler
Signature: (image_a: NDArray, image_b: NDArray, label_a: NDArray, label_b: NDArray,
            bbox: tuple[int,int,int,int]) -> tuple[NDArray, NDArray]
Pure function boundary: explicit bbox in, mixed image and mixed label out; no
  global RNG, model state, GPU state, or file I/O.
Contracts:
  - require: image_a.shape == image_b.shape
  - require: 0 <= bbox[0] < bbox[2] <= image_a.shape[0]
  - ensure: result[0].shape == image_a.shape
Witness: 32x32x3 uint8 images, known bbox, verify pixel replacement and label mix
Dependencies: numpy only
CDG stages covered: bengali_grapheme_1st/heavy_augmentation, cassava_leaf_1st/image_augmentation,
  global_wheat_1st/heavy_augmentation, ...
```
