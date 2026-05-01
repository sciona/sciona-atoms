# Research: Geospatial, GNSS & Sensor Fusion Atoms

## Goal

Find best-in-class, pure-function implementations for geospatial processing,
GNSS signal handling, and inertial sensor fusion. Target repo: `sciona-atoms-geo`.

## CDG stages this research covers (~10 stages)

- Raw GNSS pseudorange parsing (Google Smartphone Decimeter)
- Satellite signal quality filtering — C/N0 thresholding (Google Decimeter)
- EKF state estimation for GNSS+IMU fusion (Google Decimeter — partial match exists)
- RTS smoother (Google Decimeter — partial match exists)
- Pedestrian Dead Reckoning from IMU (Indoor Navigation)
- Snap-to-grid / map matching (Indoor Navigation)
- Satellite image resampling / GSD normalization (DSTL Satellite, Google Contrails)
- Multi-resolution image alignment (DSTL Satellite)
- Magic multiplier / bias correction for predictions (M5 Accuracy)
- Coordinate system conversions — ECEF, LLA, ENU (Google Decimeter)

## What to research

### 1. Coordinate conversions
- `ecef_to_lla(x, y, z) -> (lat, lon, alt)` — Earth-Centered to Lat/Lon/Alt
- `lla_to_ecef(lat, lon, alt) -> (x, y, z)` — inverse
- `ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt) -> (east, north, up)` — local frame
- Source: pyproj (MIT) or pure numpy implementation of WGS84 formulas

### 2. GNSS pseudorange corrections
- Clock bias correction: `correct_clock_bias(pseudorange, clock_bias) -> corrected`
- Ionospheric correction (Klobuchar model)
- Tropospheric correction (Saastamoinen model)
- Source: GNSS textbooks, georinex (MIT), gnss-lib (check license)

### 3. Satellite signal quality filtering
- `filter_by_cn0(measurements: NDArray, cn0: NDArray, threshold: float) -> NDArray`
- `filter_multipath(measurements: NDArray, multipath_indicator: NDArray, threshold: float) -> NDArray`
- Simple threshold-based filtering

### 4. Pedestrian Dead Reckoning (PDR)
- Step detection from accelerometer magnitude
- `detect_steps(accel_magnitude: NDArray, threshold: float, min_interval: int) -> NDArray[indices]`
- Step length estimation: `estimate_step_length(accel_magnitude: NDArray, step_indices: NDArray) -> NDArray`
- Heading from gyroscope integration: `integrate_heading(gyro_z: NDArray, dt: float) -> NDArray`
- Position update: `pdr_position_update(step_lengths, headings, initial_position) -> NDArray`
- Source: competition solutions, PDR research papers

### 5. Map matching / snap-to-grid
- Project trajectory points to nearest grid/hallway points
- `snap_to_nearest(points: NDArray, grid_points: NDArray) -> NDArray`
- KD-tree nearest neighbor lookup
- Source: scipy.spatial.KDTree (BSD)

### 6. GSD-aware image operations
- Resample satellite images to common ground sampling distance
- `resample_to_gsd(image: NDArray, current_gsd: float, target_gsd: float) -> NDArray`
- We have GSD atoms in sciona-atoms-geo — check if they cover this

### 7. EKF / RTS smoother
- Check existing coverage: `update_step` and `predict_step` from sciona-atoms-robotics
- Research whether we need GNSS-specific EKF atoms or if the existing
  robotics EKF atoms can be reused with different state models
- RTS smoother: `rts_smooth(filtered_states, filtered_covs, predicted_states, predicted_covs, transition_matrices) -> smoothed_states`

## Research questions

1. For GNSS: what are the standard correction models in pure Python?
   (Klobuchar and Saastamoinen have simple analytical formulas)
2. For PDR: what step detection algorithm is standard?
   (Peak detection on accelerometer magnitude — we have detect_peaks_in_signal)
3. For coordinate conversions: pyproj vs pure numpy WGS84?
   (Pure numpy avoids the C dependency)
4. For map matching: simple nearest-point vs Hidden Markov Model?
   (Start with nearest-point, HMM is orchestration)
5. For EKF: can we reuse robotics atoms or need GNSS-specific?
   (Reuse — the EKF math is the same, state model differs)

## Output format

Concept types: `geometry` for coordinates, `sequential_filter` for EKF/RTS,
`signal_filter` for quality filtering, and `data_extraction` for parsing.

For each candidate atom, provide:
```
Name: haversine_distance
Description: Compute great-circle distance between latitude/longitude points.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: geometry, sequential_filter, signal_filter, or data_extraction
Signature: (lat1: NDArray, lon1: NDArray, lat2: NDArray,
            lon2: NDArray, radius: float) -> NDArray
Pure function boundary: coordinate/sensor arrays and explicit parameters in,
  transformed coordinates, filtered states, or parsed records out; no device I/O,
  hidden clocks, global state, network calls, or file I/O.
Contracts:
  - require: latitude values are in [-90, 90]
  - require: longitude values are in [-180, 180]
  - require: radius > 0
  - ensure: distances are non-negative
Witness: identical coordinate pair returns zero; known city pair returns an
  approximate expected distance within tolerance.
Dependencies: numpy/scipy preferred; pyproj/geographiclib acceptable when
  precision or projection support requires them
CDG stages covered: google_smartphone_decimeter/coordinate_transform,
  indoor_location/sensor_filtering, ...
```
