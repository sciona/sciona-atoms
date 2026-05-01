# Research: 3D / Volumetric Augmentation Atoms

## Goal

Find best-in-class, pure-function implementations for 3D medical image
augmentations: random 3D rotation, elastic deformation, volumetric scaling,
Gaussian noise injection. Target repo: `sciona-atoms-bio` or `sciona-atoms-dl`.

## CDG stages this research covers (2 stages)

- `byu_flagellar_motors_4th/medical_augmentation`: Apply domain-specific
  tensor augmentations (rotation, volumetric scaling, Gaussian noise injection)
  tailored specifically to 3D cryo-ET data (MONAI/TorchIO)
- `vesuvius_ink_detection_1st/volumetric_augmentation`: Apply 3D rotations and
  random elastic deformations to CT volumes (TorchIO)

## What to research

### 1. Random 3D rotation
- Rotate a 3D volume by random angles around x/y/z axes
- `random_rotate_3d(volume: NDArray, angles: tuple[float,float,float], order: int) -> NDArray`
- Source: scipy.ndimage.rotate (BSD) for single-axis, compose for multi-axis
- TorchIO RandomAffine (Apache-2.0), MONAI RandRotate (Apache-2.0)

### 2. 3D elastic deformation
- Apply random smooth deformation field to 3D volume
- `elastic_deform_3d(volume: NDArray, sigma: float, alpha: float, seed: int) -> NDArray`
- Source: elasticdeform library (MIT), scipy.ndimage.map_coordinates
- Key: generate random displacement field, smooth with Gaussian, apply

### 3. Gaussian noise injection
- `add_gaussian_noise_3d(volume: NDArray, mean: float, std: float, seed: int) -> NDArray`
- Pure numpy: `volume + rng.normal(mean, std, volume.shape)`

### 4. Volumetric scaling / zoom
- `scale_volume(volume: NDArray, scale_factors: tuple[float,float,float], order: int) -> NDArray`
- Source: scipy.ndimage.zoom (BSD)

## Research questions

1. Can we extract pure numpy/scipy implementations from TorchIO/MONAI without
   requiring torch as a dependency?
2. For elastic deformation: what is the standard sigma/alpha range for medical volumes?
3. What are natural contracts? (volume is 3D or 4D with channel dim,
   rotation angles in radians, scale_factors positive)

## Output format

Concept types: `sampler` for augmentation operations.

For each candidate atom, provide:
```
Name: random_rotate_3d
Description: Rotate a 3D volume by specified angles around each axis using
  trilinear interpolation.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: sampler
Signature: (volume: NDArray, angles_deg: tuple[float, float, float],
            order: int) -> NDArray
Pure function boundary: 3D array and explicit rotation parameters in,
  rotated array out; no file I/O, GPU state, or global RNG — seed must be
  passed explicitly when randomness is needed.
Contracts:
  - require: volume.ndim >= 3
  - require: order in (0, 1, 2, 3)
  - ensure: result.shape == volume.shape (when no cropping)
Witness: small 8x8x8 volume with a single nonzero voxel; verify it moves
  to the expected location after 90-degree rotation.
Dependencies: numpy + scipy.ndimage preferred; torch/MONAI only as reference
CDG stages covered: byu_flagellar_motors_4th/medical_augmentation,
  vesuvius_ink_detection_1st/volumetric_augmentation
```
