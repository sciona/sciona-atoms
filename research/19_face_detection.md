# Research: Face Detection & Alignment Atoms

## Goal

Find best-in-class implementations for face detection bounding box extraction
and landmark-based face alignment cropping. Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (2 stages)

- `dfdc_deepfake_detection/face_detection`: Run MTCNN/RetinaFace detector on
  frames to find face bounding boxes (PyTorch)
- `dfdc_deepfake_detection/face_alignment_cropping`: Apply margin expansions
  and align facial landmarks to a standard template, cropping to face region

## What to research

### 1. Face detection (opaque vs decomposable?)
- MTCNN: MIT-licensed Python implementation (facenet-pytorch)
- RetinaFace: multiple implementations, check licenses
- Key question: should this be opaque (pretrained model) or decomposable?
- Recommend: opaque wrapper with contract `(B,3,H,W) -> list[boxes, landmarks, scores]`

### 2. Face alignment from landmarks
- Given 5 facial landmarks (eyes, nose, mouth corners), compute similarity transform
- `align_face(image: NDArray, landmarks: NDArray, target_landmarks: NDArray, output_size: tuple) -> NDArray`
- This IS decomposable — it's a similarity transform (rotation + scale + translation)
- Source: skimage.transform.SimilarityTransform or pure numpy least-squares
- Standard target template: FFHQ alignment landmarks

### 3. Margin-expanded face crop
- `crop_face_with_margin(image: NDArray, bbox: NDArray, margin: float) -> NDArray`
- Expand bounding box by margin percentage, clip to image bounds, crop
- Pure numpy indexing

## Research questions

1. Is MTCNN decomposable into atoms, or should it be a single opaque wrapper?
   (Recommend: opaque — it's a cascade of 3 neural networks)
2. For alignment: what is the standard 5-point landmark template?
   (FFHQ uses specific coordinates — document them)
3. What is the pure numpy/scipy similarity transform implementation?
4. Licenses: facenet-pytorch MTCNN (MIT), insightface RetinaFace (MIT)

## Output format

Concept types: `neural_network` for detection (opaque), `geometry` for alignment.

For each candidate atom, provide:
```
Name: face_similarity_align
Description: Align a detected face to a canonical template using a similarity
  transform estimated from five facial landmarks.
Source: URL to the best reference implementation or paper
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: geometry
Signature: (image: NDArray, src_landmarks: NDArray, dst_landmarks: NDArray,
            output_size: tuple[int, int]) -> NDArray
Pure function boundary: image array, landmark coordinates, and target template
  in, cropped/aligned face image out; no model inference, file I/O, GPU state,
  or global RNG.
Contracts:
  - require: src_landmarks.shape == (5, 2)
  - require: dst_landmarks.shape == (5, 2)
  - ensure: result.shape[:2] == output_size
Witness: synthetic image with known landmark positions; verify aligned output
  matches expected transform.
Dependencies: numpy + scipy or skimage for affine transform
CDG stages covered: dfdc_deepfake_detection/face_alignment_cropping
```
