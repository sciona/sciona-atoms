# Research: Graph Construction & GNN Utility Atoms

## Goal

Find best-in-class, pure-function implementations for graph construction,
graph feature computation, and GNN preprocessing. Target repos:
`sciona-atoms` (general graph) and `sciona-atoms-dl` (GNN-specific).

## CDG stages this research covers (~17 stages)

- Molecular graph construction from 3D coordinates (CHAMPS, Novozymes)
- RNA secondary structure to adjacency matrix (COVID Vaccine mRNA)
- Knowledge graph operations (OGB WikiKG90M)
- Road network graph from skeleton pixels (SpaceNet3 Roads)
- Spatial graph construction — KNN/radius from point clouds (IceCube Neutrinos)
- Team/entity interaction graph (March Mania, Foursquare)
- Connected component extraction (Shopee, Foursquare)
- Graph Laplacian computation (COVID Vaccine, spectral methods)
- Node feature engineering — degree, centrality, PageRank (March Mania)
- GNN message passing preprocessing (IceCube, OGB MAG240M)
- RNA base-pair probability to graph features (Stanford Ribonanza)

## What to research

### 1. KNN graph from point cloud
- Build K-nearest-neighbor graph from coordinate array
- `knn_graph(points: NDArray, k: int, metric: str) -> tuple[NDArray, NDArray]`
  returns (edge_index, edge_distances)
- Source: scipy.spatial.KDTree (BSD), sklearn NearestNeighbors (BSD)

### 2. Radius graph from coordinates
- Connect all points within radius r
- `radius_graph(points: NDArray, radius: float) -> tuple[NDArray, NDArray]`
- Source: scipy.spatial.KDTree.query_ball_point

### 3. Molecular graph from 3D coordinates
- Atoms as nodes, bonds as edges (distance-based or explicit bond table)
- `molecular_distance_graph(coordinates: NDArray, elements: NDArray, cutoff: float) -> tuple[NDArray, NDArray, NDArray]`
  returns (edge_index, edge_distances, edge_features)
- Source: RDKit (BSD) or pure distance computation

### 4. Adjacency matrix to edge list (and vice versa)
- `adjacency_to_edge_list(adj_matrix: NDArray) -> NDArray` — (2, num_edges)
- `edge_list_to_adjacency(edge_list: NDArray, num_nodes: int) -> NDArray`
- Pure numpy operations

### 5. Graph Laplacian
- `graph_laplacian(adjacency: NDArray, normalized: bool) -> NDArray`
- D - A (unnormalized) or I - D^{-1/2} A D^{-1/2} (symmetric normalized)
- Already exists partially — research whether our version covers all CDG needs

### 6. Node degree and centrality features
- `node_degrees(edge_index: NDArray, num_nodes: int) -> NDArray`
- `pagerank(adjacency: NDArray, damping: float, max_iter: int) -> NDArray`
- Power iteration method for PageRank
- Source: NetworkX (BSD) for reference, pure numpy for implementation

### 7. Connected components
- `connected_components(edge_index: NDArray, num_nodes: int) -> NDArray[labels]`
- Union-find algorithm, pure Python/numpy
- Source: scipy.sparse.csgraph.connected_components (BSD)

### 8. Skeleton to graph (road network)
- Convert binary skeleton image to graph of nodes (junctions/endpoints) and edges (road segments)
- `skeleton_to_graph(skeleton: NDArray) -> tuple[NDArray, NDArray]`
- Source: sknw library or skimage skeleton analysis

## Research questions

1. What is the standard KNN graph construction in numpy/scipy?
   (KDTree for low-dim, brute-force for small N)
2. For molecular graphs: should we use distance-only or incorporate
   bond types from RDKit? (distance-only is simpler, more general)
3. For connected components: scipy vs union-find — which is more
   appropriate for sparse graphs? (scipy for sparse matrices)
4. What contracts are natural? (k > 0, radius > 0, adjacency symmetric,
   edge_index has valid node indices)
5. What graph formats should atoms use? Edge list (2, E) is most common
   in PyG. Adjacency matrix is better for spectral methods. Support both?

## Output format

Concept types: `graph_traversal` for graph construction and `analysis` for
feature computation.

For each candidate atom, provide:
```
Name: knn_graph
Description: Build a k-nearest-neighbor graph from point or embedding features.
Source: URL to the best reference implementation, paper, or library source
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Concept type: graph_traversal or analysis
Signature: (features: NDArray, k: int, metric: str = "euclidean") -> NDArray
Pure function boundary: node features and explicit graph parameters in, edge
  list or adjacency matrix out; no training loop, global state, GPU state, or
  file I/O.
Contracts:
  - require: features is a 2D array
  - require: 0 < k < number of nodes
  - ensure: returned edge indices are valid node indices
Witness: four 2D points with k=1; verify nearest-neighbor edges.
Dependencies: numpy/scipy/sklearn acceptable depending on algorithm and license
CDG stages covered: molecular_graph/graph_construction, recommender/knn_graph, ...
```
