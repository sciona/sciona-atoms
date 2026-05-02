# Research: 3D Perception Primitives (LiDAR / BEV / Multi-Modal Fusion)

## Goal

Find best-in-class, pure-function implementations for 3D perception
operations used in autonomous driving competition pipelines. Target repo:
`sciona-atoms-dl`.

## CDG stages this research covers (4 stages)

- `lyft_3d_object_detection_1st/voxelization`: Discretize continuous LiDAR
  point cloud into a structured 3D grid of voxels
- `lyft_3d_object_detection_1st/multi_modal_fusion`: Project 2D camera
  bounding boxes onto 3D LiDAR space to "paint" point cloud with RGB features
- `lyft_motion_1st/bev_rasterization`: Convert vector map data, ego-vehicle
  history, and agent history into a 3D bird's-eye-view image tensor
- `facebook_image_similarity_1st/spatial_verification`: RANSAC + homography
  estimation from matched keypoints for geometric verification

## What to research

### 1. Point cloud voxelization
- Discretize (x, y, z) point cloud into a fixed-resolution 3D grid
- `voxelize_point_cloud(points: NDArray, voxel_size: tuple[float,float,float],
   point_range: tuple[float,...], max_points_per_voxel: int) -> tuple[NDArray, NDArray, NDArray]`
  Returns (voxel_features, voxel_coords, num_points_per_voxel)
- Hard voxelization: random sample or mean of points per voxel
- Source: OpenPCDet (Apache-2.0), mmdetection3d (Apache-2.0), or
  spconv voxelization (Apache-2.0)
- Key: the numpy version without CUDA — iterate points, assign to grid cells,
  aggregate features within each cell

### 2. Camera-to-LiDAR projection (point painting)
- Project 2D image features onto 3D points using calibration matrices
- `project_image_to_points(points_3d: NDArray, image_features: NDArray,
   intrinsic: NDArray, extrinsic: NDArray, image_shape: tuple) -> NDArray`
  Returns points augmented with projected image features
- Steps: transform 3D→camera frame, project to pixel coords, sample features
- Source: PointPainting paper (Vora et al. 2020), nuscenes-devkit (Apache-2.0)
- Pure numpy: matrix multiply for projection, bilinear interpolation for sampling

### 3. Bird's-eye-view rasterization
- Render vector map elements (lanes, crosswalks) and agent trajectories
  as a multi-channel top-down image
- `rasterize_bev(agents: NDArray, map_elements: list[NDArray],
   ego_pose: NDArray, raster_size: tuple[int,int],
   resolution: float) -> NDArray`
  Returns (C, H, W) multi-channel BEV image
- Each channel: one semantic class (ego history, other agents, lane lines, etc.)
- Source: L5Kit (Apache-2.0), nuscenes-devkit (Apache-2.0)
- Pure numpy: rasterize polygons/polylines onto grid using scan-line fill

### 4. RANSAC homography estimation
- Estimate homography matrix from point correspondences with outlier rejection
- `ransac_homography(src_points: NDArray, dst_points: NDArray,
   threshold: float, max_iterations: int, seed: int) -> tuple[NDArray, NDArray]`
  Returns (3x3 homography matrix, inlier mask)
- Steps: sample 4 point pairs, compute homography via DLT, count inliers, repeat
- Source: OpenCV (Apache-2.0) for reference, skimage.transform (BSD) for
  ProjectiveTransform, or pure numpy DLT implementation
- The DLT (Direct Linear Transform) for 4-point homography is ~20 lines of numpy

## Research questions

1. For voxelization: what is the pure numpy implementation without spconv/CUDA?
   (Assign points to grid cells via floor division, aggregate via scatter)
2. For point painting: what is the standard calibration matrix format?
   (4x4 extrinsic, 3x3 intrinsic — standard pinhole camera model)
3. For BEV rasterization: how are polylines rasterized to a grid?
   (Bresenham line drawing or cv2.fillPoly equivalent in numpy)
4. For RANSAC homography: what is the minimal DLT implementation?
   (SVD of the 8xN constraint matrix from 4+ point pairs)
5. What contracts are natural? (voxel_size > 0, points are Nx3 or Nx4,
   intrinsic is 3x3, homography inlier count > 4)

## Output format

Concept types: `data_assembly` for voxelization/rasterization, `geometry`
for projection/homography.

For each candidate atom, provide:
```
Name: voxelize_point_cloud
Description: Discretize a 3D point cloud into a fixed-resolution voxel grid,
  aggregating point features within each occupied voxel.
Source: URL to the best reference implementation or paper
License: Apache-2.0, MIT, or BSD; flag any incompatible license
Concept type: data_assembly or geometry
Signature: (points: NDArray, voxel_size: tuple[float, float, float],
            point_range: tuple[float, ...], max_points_per_voxel: int) -> tuple[NDArray, NDArray, NDArray]
Pure function boundary: point cloud array and explicit grid parameters in,
  voxel arrays out; no GPU state, file I/O, or global RNG.
Contracts:
  - require: points.ndim == 2 and points.shape[1] >= 3
  - require: all voxel_size values > 0
  - ensure: voxel_coords are within the grid dimensions
Witness: small point cloud (20 points) with known voxel assignments; verify
  correct grid cell assignment and feature aggregation.
Dependencies: numpy only preferred; scipy acceptable for spatial operations
CDG stages covered: lyft_3d_object_detection_1st/voxelization
```
