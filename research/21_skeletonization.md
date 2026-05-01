# Research: Morphological Skeletonization Atoms

## Goal

Find best-in-class, pure-function implementation for morphological
skeletonization (thinning) of binary masks. Target repo: `sciona-atoms-dl`
or `sciona-atoms`.

## CDG stages this research covers (1 stage)

- `spacenet3_roads_1st/skeletonization`: Pass the binary road mask through a
  morphological skeletonization algorithm to reduce roads to 1-pixel wide
  continuous lines (skimage.morphology.skeletonize)

## What to research

### 1. Skeletonization (medial axis thinning)
- `skeletonize(mask: NDArray) -> NDArray`
- Zhang-Suen or Lee94 thinning algorithm
- Source: skimage.morphology.skeletonize (BSD-3-Clause)
- Pure implementation: iterative erosion with connectivity preservation

### 2. Medial axis transform (alternative)
- `medial_axis(mask: NDArray) -> tuple[NDArray, NDArray]`
- Returns skeleton + distance transform
- Source: skimage.morphology.medial_axis (BSD-3-Clause)

### 3. Skeleton to graph conversion
- `skeleton_to_graph(skeleton: NDArray) -> tuple[NDArray, NDArray]`
- Identify junction points (degree > 2) and endpoints (degree == 1)
- Build adjacency from connected skeleton segments
- Source: sknw library or custom implementation

## Research questions

1. Is skimage.morphology.skeletonize a pure function?
   (Yes — takes binary array, returns binary array, no side effects)
2. Should we wrap skimage or reimplement?
   (Wrap — the algorithm is complex and well-tested in skimage)
3. What contracts are natural? (input is binary 2D, output same shape,
   output is subset of input foreground)
4. Is skeleton-to-graph a separate atom? (Yes — different concept_type)

## Output format

Concept types: `signal_filter` for skeletonization, `graph_traversal` for
skeleton-to-graph.

For each candidate atom, provide:
```
Name: skeletonize_2d
Description: Reduce a binary mask to a one-pixel-wide skeleton via
  morphological thinning.
Source: URL to the best reference implementation or paper
License: BSD-3-Clause (scikit-image)
Concept type: signal_filter
Signature: (mask: NDArray) -> NDArray
Pure function boundary: binary 2D mask in, thinned binary mask out; no file
  I/O, global state, or GPU operations.
Contracts:
  - require: mask.ndim == 2
  - require: mask values are 0 or 1
  - ensure: result.shape == mask.shape
  - ensure: result is a subset of mask (no new foreground pixels)
Witness: small binary rectangle (10x5 block of ones); verify skeleton is a
  1-pixel-wide line along the medial axis.
Dependencies: scikit-image (BSD-3-Clause)
CDG stages covered: spacenet3_roads_1st/skeletonization
```
