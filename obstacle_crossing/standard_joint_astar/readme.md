# Standard joint A*

This is a Python implementation of the A* algorithm for two agents moving simultaneously. While moving, the agents must remain within a specified distance threshold from one another.

## Algorithms

### Joint A* 4-way
**File:** `joint_astar_4_way.py`

In this version, each agent can move in 4 directions (horizontal/vertical) or stay still.

### Joint A* 8-way
**File:** `joint_astar_8_way.py`

In this version, each agent can move in 8 directions (horizontal/vertical and diagonal) or stay still.

**Required packages:**
- `PIL` (Pillow) - Image processing and map loading
- `numpy` - Array operations and grid manipulation  
- `matplotlib` - Path visualization and plotting
- `opencv-python` (cv2) - Image dilation and preprocessing
- `typing` - Type hints (built-in for Python 3.5+)

## Important Note

⚠️ **Research Use Only**: This code is intended for research and demonstration purposes. Some functionalities are experimental and may not be fully implemented for production use.

