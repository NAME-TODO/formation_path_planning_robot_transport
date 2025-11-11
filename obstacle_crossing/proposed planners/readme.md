# Proposed planners

Two Python algorithms for computing collision-free paths for dual robot formations navigating around opposite sides of obstacles.

## Algorithms

### A*-based Planner
**File:** `astar_based_planner.py`

The algorithm limits search to boundary points of the obstacle for maximum computational speed and efficiency.

### Graph-search-based Planner 
**File:** `graph_search_based_planner.py`

The algorithm considers all combinations of positions around the obstacle boundary with an optimization process to minimize the distance between obstacles and side formations. This approach increases computational time but enables traversal of critical passages and provides more maneuvering room around obstacles.

## Dependencies

- **OpenCV** - Image processing and visualization
- **NumPy** - Numerical computations  
- **NetworkX** - Graph algorithms and data structures
- **Shapely** - Geometric operations
- **Matplotlib** - Plotting and visualization
- **scikit-image** - Image processing

## Important Note

⚠️ **Research Use Only**: This code is intended for research and demonstration purposes. Some functionalities are experimental and may not be fully implemented for production use.

