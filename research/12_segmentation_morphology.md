# Research: Segmentation Post-Processing & Morphological Operations Atoms

## Goal

Find best-in-class, pure-function implementations for segmentation mask
post-processing, morphological operations, and CRF refinement.
Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (~16 stages)

- CRF post-processing on segmentation masks (DSTL Satellite, TGS Salt)
- Morphological closing/opening/dilation/erosion (HubMap, Severstal)
- Hole filling in binary masks (HubMap)
- Connected component filtering (Severstal, Sartorius)
- Watershed segmentation (Sartorius Cell)
- Instance segmentation from semantic masks (Sartorius)
- Mask-to-RLE encoding (many segmentation CDGs)
- RLE-to-mask decoding (many segmentation CDGs)
- WKT polygon to raster mask (DSTL Satellite)
- Boundary smoothing via contour approximation (HubMap)
- Area thresholding — remove small masks (Severstal, TGS)
- False color generation from multi-band imagery (Google Contrails)
- Vector rasterization (DSTL Satellite)

## What to research

### 1. Morphological operations (pure scipy/numpy)
- `morphological_close(mask: NDArray, kernel_size: int) -> NDArray`
- `morphological_open(mask: NDArray, kernel_size: int) -> NDArray`
- `dilate_mask(mask: NDArray, iterations: int) -> NDArray`
- `erode_mask(mask: NDArray, iterations: int) -> NDArray`
- Source: scipy.ndimage.binary_closing, etc. (BSD)

### 2. Hole filling
- `fill_holes(mask: NDArray) -> NDArray`
- Source: scipy.ndimage.binary_fill_holes (BSD)

### 3. Connected component filtering
- `filter_components_by_area(mask: NDArray, min_area: int) -> NDArray`
- Label components, measure areas, remove small ones
- Source: scipy.ndimage.label + component filtering

### 4. Dense CRF post-processing
- Apply dense CRF to refine segmentation boundaries
- `dense_crf_2d(image: NDArray, unary: NDArray, sxy: float, srgb: float, compat: float, iterations: int) -> NDArray`
- Source: pydensecrf (MIT) — the Krahenbuhl & Koltun 2011 implementation
- Note: pydensecrf is a compiled C++ library — may need opaque treatment

### 5. Run-Length Encoding (RLE)
- `mask_to_rle(mask: NDArray) -> list[int]` — encode binary mask as run lengths
- `rle_to_mask(rle: list[int], shape: tuple) -> NDArray` — decode back
- Standard Kaggle submission format
- Pure Python/numpy

### 6. Watershed instance segmentation
- Convert semantic segmentation to instance masks via watershed
- `watershed_instances(distance_map: NDArray, mask: NDArray, min_distance: int) -> NDArray`
- Source: scipy.ndimage or skimage.segmentation.watershed

### 7. Contour smoothing / polygon approximation
- `smooth_contour(mask: NDArray, epsilon: float) -> NDArray`
- Douglas-Peucker polygon simplification
- Source: OpenCV cv2.approxPolyDP concept, or pure Python implementation

### 8. WKT polygon to raster mask
- `wkt_to_mask(wkt_string: str, image_shape: tuple, transform: tuple) -> NDArray`
- Convert Well-Known Text polygons to binary raster masks
- Source: shapely (BSD) + rasterio (BSD), or manual scan-line rasterization

### 9. Multi-band false color composition
- `false_color_composite(bands: NDArray, channel_indices: tuple, formula: str) -> NDArray`
- Combine satellite bands using domain formulas (e.g., Ash morphology for contrails)
- Pure numpy band arithmetic

## Research questions

1. For CRF: should we wrap pydensecrf or implement a simplified version?
   (Recommend: opaque wrapper — the C++ bilateral filtering is the core)
2. For morphological ops: scipy.ndimage vs skimage — which is more standard?
   (scipy.ndimage is lighter, skimage has more variants)
3. For RLE: what is the most efficient numpy implementation?
   (np.diff + np.where approach)
4. For watershed: what preprocessing is standard?
   (Distance transform → local maxima → watershed)
5. What contracts are natural? (mask is binary, kernel_size > 0,
   RLE encodes same number of pixels as mask, CRF iterations > 0)

## Output format

Concept types: `signal_filter` for morphological ops, `data_extraction` for
encoding/decoding, and `analysis` for segmentation.

For each candidate atom, provide:
```
Name: binary_opening
Description: Apply erosion followed by dilation to remove small foreground
  artifacts from a binary mask.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: signal_filter, data_extraction, or analysis
Signature: (mask: NDArray, kernel_size: int) -> NDArray
Pure function boundary: masks and explicit parameters in, filtered masks,
  encodings, or scalar metrics out; no image file I/O, global state, GPU state,
  or model inference.
Contracts:
  - require: mask is binary
  - require: kernel_size > 0
  - ensure: result.shape == mask.shape
Witness: small binary mask with a one-pixel artifact; verify the artifact is
  removed while the larger component remains.
Dependencies: numpy/scipy/scikit-image acceptable depending on operation and license
CDG stages covered: tgs_salt/morphology, sartorius/rle_encoding, ...
```
