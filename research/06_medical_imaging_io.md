# Research: Medical Imaging I/O & Preprocessing Atoms

## Goal

Find best-in-class, pure-function implementations for medical image loading,
preprocessing, and domain-specific operations. Target repo: `sciona-atoms-bio`.

## CDG stages this research covers (~16 stages)

- DICOM loading and windowing (RSNA PE, RSNA Mammography, RSNA Cervical Spine)
- 3D volume resampling / interpolation (RSNA Cervical Spine, Passenger Screening)
- Stain normalization — Macenko method (HubMap HPA)
- CycleGAN stain translation concept (HubMap HPA)
- Multi-view 2D projection from 3D volumes (Passenger Screening)
- Body zone segmentation (Passenger Screening)
- 3D volume registration / coregistration (RSNA MICCAI Brain Tumor)
- NIfTI loading (brain tumor CDGs)
- 2.5D slice extraction — stacking adjacent slices as channels (RSNA Cervical, DFL)
- Proprietary format conversion (.a3d mmWave to numpy) (Passenger Screening)
- Brain voxel extraction and mapping (TReNDS Neuroimaging)

## What to research

### 1. DICOM loading and windowing
- Read pixel data from DICOM, apply window center/width
- `dicom_window(pixel_array: NDArray, window_center: float, window_width: float) -> NDArray`
- `dicom_to_hounsfield(pixel_array: NDArray, slope: float, intercept: float) -> NDArray`
- Source: pydicom (MIT) for reading, numpy for windowing math
- Note: the atom should take already-loaded pixel arrays, not do file I/O

### 2. 3D volume resampling
- Resample 3D medical volumes to target spacing/resolution
- `resample_volume(volume: NDArray, current_spacing: tuple, target_spacing: tuple, order: int) -> NDArray`
- Source: scipy.ndimage.zoom or scipy.ndimage.map_coordinates (BSD)

### 3. Stain normalization (Macenko)
- Macenko et al. 2009 — SVD-based stain vector estimation
- Steps: convert to optical density, SVD, project, normalize, reconstruct
- `macenko_stain_vectors(image: NDArray) -> tuple[NDArray, NDArray]`
- `macenko_normalize(image: NDArray, target_vectors: NDArray) -> NDArray`
- Source: StainTools library (MIT), or HistomicsTK

### 4. Multi-view 2D projection
- Max intensity projection (MIP) from 3D volume at multiple angles
- `max_intensity_projection(volume: NDArray, angle: float, axis: int) -> NDArray`
- Rotate volume, project along axis via np.max
- Source: scipy.ndimage.rotate + np.max

### 5. 2.5D slice extraction
- Stack N adjacent slices as channels for pseudo-3D input
- `extract_25d_slices(volume: NDArray, center_idx: int, num_adjacent: int) -> NDArray`
- Pure numpy indexing with boundary handling

### 6. Brain voxel extraction
- Extract 3D regions of interest from NIfTI brain volumes
- Crop to brain mask, resample to standard space
- `crop_to_mask(volume: NDArray, mask: NDArray, margin: int) -> NDArray`

### 7. Connected component filtering for medical masks
- Remove small connected components from binary segmentation masks
- `filter_small_components(mask: NDArray, min_size: int) -> NDArray`
- Source: scipy.ndimage.label + size filtering (BSD)

## Research questions

1. What is the pure numpy/scipy implementation for each operation?
   (pydicom for DICOM parsing is acceptable as a dependency,
    but windowing/resampling should be numpy/scipy only)
2. For stain normalization: what is the exact Macenko algorithm?
   (SVD, percentile thresholding, concentration estimation)
3. For 3D resampling: scipy.ndimage.zoom vs map_coordinates — which is more
   appropriate for medical volumes? (anisotropic spacing considerations)
4. What are natural contracts? (window_width > 0, volume is 3D,
   spacing values positive, Macenko input is RGB uint8)
5. What edge cases exist? (empty DICOM, single-slice volumes,
   monochrome vs multi-channel)

## Output format

Concept types: `data_extraction` for I/O, `signal_transform` for normalization,
and `data_assembly` for volume manipulation.

For each candidate atom, provide:
```
Name: dicom_window
Description: Apply DICOM window center and width to an already-loaded pixel
  array and return clipped/scaled intensities.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: data_extraction, signal_transform, or data_assembly
Signature: (pixel_array: NDArray, window_center: float,
            window_width: float) -> NDArray
Pure function boundary: already-loaded arrays and explicit metadata in,
  transformed arrays out; no DICOM/NIfTI file I/O, global state, GPU state, or
  network access.
Contracts:
  - require: window_width > 0
  - require: pixel_array is numeric
  - ensure: result.shape == pixel_array.shape
Witness: small 2D pixel array with fixed center/width; verify clipping at the
  expected lower and upper bounds.
Dependencies: numpy/scipy preferred; pydicom/nibabel acceptable only for parsing
  wrappers that still expose a clear pure boundary
CDG stages covered: rsna_pe/dicom_windowing, rsna_cervical/volume_resampling, ...
```
