# Map Analysis Framework

A Python-based framework for formation-aware robotic path planning in complex environments. This framework provides tools for map processing, traversability analysis (obstacle/narrow passages detection), graph-based pathfinding, and robot formation visualization.

## Core Components

### `map_analyzer.py`
The main analysis engine containing the `MapAnalyzer` class with the following capabilities:
- **Region classification**: Identifies different navigable areas (loose space, critical space, narrow space)
- **Graph construction**: Builds a navigational graph from processed map data 
- **Global path computation**: Calculates optimal routes between start and goal points

### `obstacle_analyzer.py`
Specialized obstacle detection and analysis module:
- **Obstacle detection**: Identifies obstacles from map images
- **Obstacle preprocessing**: Splits obstacles into a left and right side; finds entry and exit points around obstacles
- **Crossing path computation**: Compute the path for the left and right side-formations around the obstacles, leveragion one of the proposed algorithms

### `graph_path_planner.py`
Implementation of our proposed graph-based search algorithm.

### `main.py`
Testing interface:
- **Step-by-step execution**: Test individual framework components

### `robot_formation.py`
Utilities for robot formation visualization and animation.

## Dependencies

- `OpenCV` - Image processing and visualization
- `NumPy` - Numerical computations
- `NetworkX` - Graph algorithms and data structures
- `Shapely` - Geometric operations
- `Matplotlib` - Plotting and visualization

## Important Note

⚠️ **Research Use Only**: This code is intended for research and demonstration purposes. Some functionalities are experimental and may not be fully implemented for production use.