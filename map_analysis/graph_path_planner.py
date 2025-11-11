"""
Graph-based Path Planning for Dual Robot Formation Navigation

This module provides a complete solution for finding collision-free paths for a formation
of robots navigating around obstacles. The planner uses A* search on a configuration graph
where nodes represent valid robot formation positions.

Author: Generated from graph_search_based_planner.py and navlib.py
"""

import math
import heapq
import time
import numpy as np
import cv2
from itertools import count
from typing import Tuple, List, Optional, Union
import networkx as nx
from shapely.geometry import Polygon, Point
from skimage.morphology import skeletonize


# ===============================
# Geometry Helpers from navlib
# ===============================

def find_longest_path(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """Return the 8-connected diameter path of a binary skeleton image.

    Returns a list of (x, y) coordinates (OpenCV-style ordering) along the longest path.
    """
    skel = skeleton.astype(bool)
    G = nx.Graph()
    rows, cols = np.where(skel)
    for r, c in zip(rows, cols):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                    G.add_edge((r, c), (rr, cc))

    if not G.nodes:
        return []

    # Two BFS passes to approximate the graph diameter
    start = next(iter(G.nodes))
    dist1 = nx.single_source_shortest_path_length(G, start)
    far1 = max(dist1, key=dist1.get)
    dist2 = nx.single_source_shortest_path_length(G, far1)
    far2 = max(dist2, key=dist2.get)

    path_rc = nx.shortest_path(G, far1, far2)  # list of (row, col)
    # Convert to (x, y) = (col, row) for OpenCV consistency
    return [(c, r) for (r, c) in path_rc]


def find_split_point(contour: np.ndarray, point: Tuple[int, int], radius: float) -> Tuple[int, int]:
    """Pick a stable split point on the inflated contour near a center point.

    It intersects the contour with a circle centered at 'point' with the given 'radius',
    collects the points inside the circle (possibly in two segments due to wrapping),
    concatenates them, and returns the middle one.
    """
    circle = Point(point).buffer(radius)

    buf1, buf2 = [], []
    old_i = 0
    buf = buf1
    switched = False

    for i in range(len(contour)):
        if circle.contains(Point(contour[i])):
            if i > 0 and i > old_i + 1 and not switched:
                buf = buf2
                switched = True
            buf.append(contour[i])
            old_i = i

    buf2.extend(buf1)
    if not buf2:
        # Fallback: if nothing found, return the closest point on the contour
        d = np.linalg.norm(contour - np.array(point), axis=1)
        return tuple(contour[int(np.argmin(d))])

    return tuple(buf2[len(buf2) // 2])


# ===============================
# Obstacle Class from navlib
# ===============================

class Obstacle:
    """Loads an obstacle from an image, inflates it, samples boundary, and extracts main skeleton branch."""

    def __init__(self, path: str, inflation_radius: int = 40, step_size: int = 10, scale: float = 1):
        self._path = path
        self._inflation_radius = inflation_radius
        self._step_size = step_size
        self._scale = scale

        self._image_gray: Optional[np.ndarray] = None
        self._debug_image: Optional[np.ndarray] = None
        self._obstacle_polygon_true: Optional[Polygon] = None
        self._obstacle_polygon: Optional[Polygon] = None
        self._sampled: Optional[np.ndarray] = None
        self._main_skeleton_branch: Optional[List[Tuple[int, int]]] = None

        self._load_obstacle_from_image()
        self._sample_polygon()
        self._compute_skeleton()

    def _load_obstacle_from_image(self) -> None:
        img = cv2.imread(self._path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read obstacle image: {self._path}")
        self._image_gray = img

        # Detect external contours and inflate
        _, thresh = cv2.threshold(self._image_gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            raise ValueError("Expected at least 2 contours (background + obstacle)")

        self._obstacle_polygon_true = Polygon(contours[1].squeeze())
        self._obstacle_polygon = self._obstacle_polygon_true.buffer(distance=self._inflation_radius)
        self._debug_image = np.zeros_like(cv2.cvtColor(self._image_gray, cv2.COLOR_GRAY2BGR))

    def _sample_polygon(self) -> None:
        assert self._obstacle_polygon is not None
        boundary = self._obstacle_polygon.boundary
        perimeter_length = boundary.length
        num_points = max(1, int(perimeter_length // self._step_size))
        sampled = [boundary.interpolate(i * self._step_size) for i in range(num_points)]
        self._sampled = np.array([(int(p.x * self._scale), int(p.y * self._scale)) for p in sampled], np.int32)

    def _compute_skeleton(self) -> None:
        assert self._image_gray is not None and self._obstacle_polygon is not None
        mask = np.zeros_like(self._image_gray)
        coords = np.array(list(self._obstacle_polygon.exterior.coords), dtype=np.int32)
        cv2.fillPoly(mask, [coords], color=255)
        skel = skeletonize((mask > 0).astype(np.uint8)).astype(np.uint8)
        self._main_skeleton_branch = find_longest_path(skel)

    def get_polygon(self) -> Polygon:
        assert self._obstacle_polygon is not None
        return self._obstacle_polygon

    def get_samples(self) -> np.ndarray:
        assert self._sampled is not None
        return self._sampled

    def get_raw_image(self) -> np.ndarray:
        assert self._image_gray is not None
        return self._image_gray

    def split_obstacle(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split inflated polygon samples into two ordered chains using skeleton endpoints as anchors."""
        assert self._sampled is not None and self._main_skeleton_branch is not None

        center_start = self._main_skeleton_branch[0]
        center_end = self._main_skeleton_branch[-1]
        radius = self._inflation_radius + 100

        start_point = find_split_point(self._sampled, center_start, radius)
        end_point = find_split_point(self._sampled, center_end, radius)

        temp = [tuple(row) for row in self._sampled]
        start_id = temp.index(tuple(start_point))
        end_id = temp.index(tuple(end_point))

        length = len(self._sampled)
        array = np.zeros_like(self._sampled)
        offset = abs(end_id - start_id)

        array[0:length - start_id] = self._sampled[start_id:]
        array[length - start_id:] = self._sampled[0:start_id]

        array_rx = array[0:offset]
        array_lx = np.flip(array[offset:], axis=0)

        return array_lx, array_rx


# ===============================
# Formation utilities from navlib
# ===============================

def find_start(array_a: np.ndarray, array_b: np.ndarray, BAR: float, TOL: float) -> Tuple[int, Optional[int]]:
    """Find the first index pair (0, j) such that distance between array_a[0] and array_b[j] is ~ BAR within tolerance TOL."""
    id_a = 0
    id_b = None
    for j in range(len(array_b)):
        distance = np.linalg.norm(array_a[id_a] - array_b[j])
        if abs(distance - BAR) < TOL * 1:
            id_b = j
            break
    return id_a, id_b


def get_oobb(A, B, robot_r, dist_deg):
    """Compute two oriented bounding boxes (mirrored) for two 3-robot groups centered at A and B.

    dist_deg is treated as a chord length; converted to an angle using the formation radius.
    Returns (oobb1, oobb2) as arrays of shape (4, 2).
    """
    # Convert inputs to float32 for better performance
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    
    center = (A + B) * 0.5
    formation_r = np.linalg.norm(B - A) * 0.5

    # Inline angle computation for better performance
    if formation_r <= 0 or dist_deg < 0:
        raise ValueError("formation_r must be > 0 and dist_deg >= 0")
    
    ratio = max(0.0, min(1.0, dist_deg / (2.0 * formation_r)))
    dist_deg_angle = 2.0 * math.degrees(math.asin(ratio))

    # Cache trigonometric calculations
    dist_rad = np.radians(dist_deg_angle)
    sin_dist = np.sin(dist_rad)
    cos_dist = np.cos(dist_rad)
    
    orientation = np.degrees(np.arctan2(A[1] - B[1], A[0] - B[0])) - 270

    # Vectorized bounding box corner calculations
    bb_corner_x = formation_r * np.sin(-dist_rad) - robot_r
    bb_corner_y = formation_r * cos_dist - robot_r
    width = 2 * robot_r + 2 * formation_r * sin_dist
    height = 2 * robot_r + formation_r * (1 - cos_dist)

    # Create corners using vectorized operations
    corners_local = np.array([
        [bb_corner_x, -bb_corner_y],
        [bb_corner_x + width, -bb_corner_y],
        [bb_corner_x + width, -(bb_corner_y + height)],
        [bb_corner_x, -(bb_corner_y + height)]
    ], dtype=np.float32)
    
    oobb = corners_local + center

    # Apply rotation if needed
    if orientation != 0:
        angle_rad = np.radians(orientation)
        cos_orient, sin_orient = np.cos(angle_rad), np.sin(angle_rad)
        Rm = np.array([[cos_orient, -sin_orient],
                       [sin_orient,  cos_orient]], dtype=np.float32)
        oobb = (Rm @ (oobb - center).T).T + center

    # Create mirrored OOBB (180-degree rotation is just negation + translation)
    oobb2 = 2 * center - oobb

    return oobb, oobb2


def check_polygon_collision(polygon: np.ndarray, grid: np.ndarray) -> bool:
    """Return True if polygon area overlaps with obstacles (0) in a binary grid (255 free, 0 obstacle)."""
    # Vectorized bounds calculation with early exit checks
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    
    x_min = max(int(np.min(x_coords)), 0)
    x_max = min(int(np.ceil(np.max(x_coords))), grid.shape[1] - 1)
    y_min = max(int(np.min(y_coords)), 0)
    y_max = min(int(np.ceil(np.max(y_coords))), grid.shape[0] - 1)
    
    # Early exit if polygon is completely outside grid bounds
    if x_min >= grid.shape[1] or x_max < 0 or y_min >= grid.shape[0] or y_max < 0:
        return False
    
    # Early exit if bounding box has no area
    if x_min > x_max or y_min > y_max:
        return False

    grid_crop = grid[y_min:y_max + 1, x_min:x_max + 1]
    
    # Early exit if cropped area is all free space (255)
    if np.all(grid_crop == 255):
        return False
    
    poly_crop = polygon - np.array([x_min, y_min], dtype=np.float32)

    mask = np.zeros_like(grid_crop, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_crop.astype(np.int32)], 1)

    return bool(np.any((mask == 1) & (grid_crop == 0)))


def fit_oobb(A, B, robot_r, deg, min_dist, min_decr, map, step: int = 1):
    """Try to minimally move A and then B along AB to fit both mirrored OOBBs collision-free, while keeping |AB| >= min_dist.

    Returns (ok, A_fit, B_fit) with integer coordinates when feasible.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    
    dist = np.linalg.norm(B - A)
    if dist < min_dist:
        return False, None, None

    # Check initial configuration first
    oobb_a, oobb_b = get_oobb(A, B, robot_r, deg)
    coll_a = check_polygon_collision(oobb_a, map)
    coll_b = check_polygon_collision(oobb_b, map)
    
    if not coll_a and not coll_b:
        return True, A.astype(np.int32), B.astype(np.int32)

    # Pre-compute direction and movement parameters
    direction = (B - A) / dist if dist != 0 else np.zeros_like(A, dtype=np.float32)
    room = dist - min_dist
    max_steps = int(room / step)
    
    # Try to fit A first
    fitted_A = A
    if coll_a and max_steps > 0:
        step_vector = direction * step
        for i in range(1, max_steps + 1):
            new_A = A + step_vector * i
            
            # Early termination if we're getting too close to B
            if np.linalg.norm(new_A - B) < min_decr:
                break
                
            oobb_a, _ = get_oobb(new_A, B, robot_r, deg)
            if not check_polygon_collision(oobb_a, map):
                fitted_A = new_A
                break
        else:
            # If no valid A position found, fail
            if coll_a:
                return False, None, None

    # Try to fit B with the fitted A
    fitted_B = B
    new_dist = np.linalg.norm(B - fitted_A)
    room_b = new_dist - min_dist
    max_steps_b = int(room_b / step)
    
    if coll_b and max_steps_b > 0:
        direction_b = (fitted_A - B) / new_dist if new_dist != 0 else np.zeros_like(B, dtype=np.float32)
        step_vector_b = direction_b * step
        
        for i in range(1, max_steps_b + 1):
            new_B = B + step_vector_b * i
            
            # Early termination if we're getting too close to fitted_A
            if np.linalg.norm(new_B - fitted_A) < min_decr:
                break
                
            _, oobb_b = get_oobb(fitted_A, new_B, robot_r, deg)
            if not check_polygon_collision(oobb_b, map):
                fitted_B = new_B
                break
        else:
            # If no valid B position found, fail
            if coll_b:
                return False, None, None

    return True, fitted_A.astype(np.int32), fitted_B.astype(np.int32)


def move_point_to_distance(A, B, target_distance):
    """Move B along AB so that |AB| == target_distance; return integer coordinates."""
    A = np.array(A)
    B = np.array(B)
    direction = B - A
    current = np.linalg.norm(direction)
    if current == 0:
        raise ValueError("A and B coincide; cannot determine direction")
    new_B = A + direction * (target_distance / current)
    return new_B.astype(int)


def move_A_along_AB_until_displacement(A, B, robot_r, deg, map, step, max_displacement):
    """Translate A and B together along AB by step until both OOBBs are collision-free or max shift reached."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    
    diff = B - A
    dist = np.linalg.norm(diff)
    
    if dist == 0:
        return False, None, None
    
    direction = diff / dist
    max_steps = int(max_displacement / step)
    
    # Pre-compute step vector for efficiency
    step_vector = direction * step
    
    for i in range(max_steps + 1):
        offset = step_vector * i
        new_A = A + offset
        new_B = B + offset
        
        # Early exit if we've moved too far
        moved = np.linalg.norm(offset)
        if moved > max_displacement:
            break
        
        oobb_a, oobb_b = get_oobb(new_A, new_B, robot_r=robot_r, dist_deg=deg)
        
        # Check both collisions - early exit on success
        if (not check_polygon_collision(oobb_a, map) and 
            not check_polygon_collision(oobb_b, map)):
            return True, new_A.astype(np.int32), new_B.astype(np.int32)
    
    return False, None, None


def spread_oobb(A, B, robot_r, deg, target_distance, map, step: int = 1, max_shift: int = 100):
    """Increase |AB| up to target_distance, then try shifting along AB to get a feasible OOBB pair."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    
    # Compute new B position efficiently (inline move_point_to_distance)
    direction = B - A
    current_dist = np.linalg.norm(direction)
    
    if current_dist == 0:
        return False, None, None
    
    # Scale direction to target distance
    new_B = A + direction * (target_distance / current_dist)
    
    # Quick collision check for the spread configuration
    _, oobb_b = get_oobb(A, new_B, robot_r, deg)
    if not check_polygon_collision(oobb_b, map):
        return move_A_along_AB_until_displacement(A, new_B, robot_r, deg, map, step=step, max_displacement=max_shift)
    
    return False, None, None


# ===============================
# Path Planning Core Functions
# ===============================

def euclid(p, q):
    """Euclidean distance between two 2D points (x, y)."""
    return math.hypot(p[0]-q[0], p[1]-q[1])


def default_astar_heuristic(u, v):
    """Heuristic for nodes represented as ((Ax,Ay),(Bx,By)) -> sum of Euclidean distances."""
    Au, Bu = u
    Av, Bv = v
    # Optimized distance calculation using hypot
    return math.hypot(Au[0]-Av[0], Au[1]-Av[1]) + math.hypot(Bu[0]-Bv[0], Bu[1]-Bv[1])


def check_neighbors(i, j, k, valid_map, len_a, len_b, array_a, array_b, G):
    """Add edges between neighboring valid configurations in the graph."""
    # Cache current configuration to avoid repeated indexing
    current_config = valid_map[i, j, k]
    Af, Bf = current_config[0], current_config[1]
    current_node = (tuple(Af), tuple(Bf))
    
    # Pre-compute neighbor offsets
    offsets = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
               (-1,  0, -1), (-1,  0, 0), (-1,  0, 1),
               (-1,  1, -1), (-1,  1, 0), (-1,  1, 1),
               ( 0, -1, -1), ( 0, -1, 0), ( 0, -1, 1),
               ( 0,  0, -1),              ( 0,  0, 1),
               ( 0,  1, -1), ( 0,  1, 0), ( 0,  1, 1),
               ( 1, -1, -1), ( 1, -1, 0), ( 1, -1, 1),
               ( 1,  0, -1), ( 1,  0, 0), ( 1,  0, 1),
               ( 1,  1, -1), ( 1,  1, 0), ( 1,  1, 1)]
    
    for di, dj, dk in offsets:
        ni, nj, nk = i + di, j + dj, k + dk
        
        # Bounds checking
        if not (0 <= ni < len_a and 0 <= nj < len_b and 0 <= nk < 3):
            continue
            
        neighbor_config = valid_map[ni, nj, nk]
        if np.any(neighbor_config):
            A, B = neighbor_config[0], neighbor_config[1]
            neighbor_node = (tuple(A), tuple(B))
            
            # Vectorized distance calculation
            diff_A = A - Af
            diff_B = B - Bf
            weight = np.linalg.norm(diff_A) + np.linalg.norm(diff_B)
            
            G.add_edge(current_node, neighbor_node, weight=weight)


def astar_heapq(G, start, goal, heuristic=None, weight='weight'):
    """A* path search on a NetworkX graph using a binary heap.

    Args:
        G: NetworkX graph with G.neighbors(node) and G[node][nbr][weight]
        start, goal: nodes of the form ((Ax,Ay),(Bx,By))
        heuristic: function(u, v) -> float, defaults to sum of Euclidean distances
        weight: edge attribute name for costs
    
    Returns:
        List of nodes from start to goal
    """
    if heuristic is None:
        heuristic = default_astar_heuristic

    open_heap = []
    push = heapq.heappush
    pop = heapq.heappop
    tie = count()

    g = {start: 0.0}
    parent = {}
    closed = set()

    push(open_heap, (heuristic(start, goal), next(tie), start))

    while open_heap:
        f_curr, _, u = pop(open_heap)
        if u in closed:
            continue
        if u == goal:
            # Reconstruct path
            path = [u]
            while u in parent:
                u = parent[u]
                path.append(u)
            return path[::-1]

        closed.add(u)
        for v in G.neighbors(u):
            w = G[u][v].get(weight, 1.0)
            tentative = g[u] + w
            if tentative < g.get(v, float('inf')):
                g[v] = tentative
                parent[v] = u
                push(open_heap, (tentative + heuristic(v, goal), next(tie), v))

    raise RuntimeError('No path found by A*')


# ===============================
# Main Path Planning Function
# ===============================

def find_dual_robot_path(
    array_a: np.ndarray,
    array_b: np.ndarray,
    start_a: Tuple[int, int],
    start_b: Tuple[int, int],
    goal_a: Tuple[int, int],
    goal_b: Tuple[int, int],
    collision_map: np.ndarray,
    robot_radius: int = 15,
    intra_robot_distance: int = 35,
    formation_distance: int = 300,
    formation_tolerance: float = 0.2,
    gamma: int = 5,
    step_size: int = 1,
    verbose: bool = True
) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Find a collision-free path for a dual robot formation navigating around obstacles.
    
    Args:
        array_a: Array of boundary points for the first obstacle side [(x, y), ...]
        array_b: Array of boundary points for the second obstacle side [(x, y), ...]
        start_a: Starting position for robot formation center A (x, y)
        start_b: Starting position for robot formation center B (x, y)
        goal_a: Goal position for robot formation center A (x, y)
        goal_b: Goal position for robot formation center B (x, y)
        collision_map: Binary map where 255=free space, 0=obstacle
        robot_radius: Radius of individual robots in pixels
        intra_robot_distance: Distance between robots within each formation
        formation_distance: Target distance between formation centers A and B
        formation_tolerance: Tolerance for formation distance (as fraction of formation_distance)
        gamma: Safety margin for formations
        step_size: Step size for configuration adjustments
        verbose: Print progress information
    
    Returns:
        List of path nodes as [(point_A, point_B), ...] or None if no path found
        Each point is a tuple (x, y) representing the center of a robot formation
    """
    
    if verbose:
        print("Starting dual robot path planning...")
        start_time = time.time()
    
    # Calculate formation parameters
    BAR = formation_distance
    TOL = formation_tolerance * BAR
    
    MAX_DIST = BAR + TOL
    MIN_DIST = BAR - TOL
    MAX_BOUND = BAR + TOL
    MIN_BOUND = BAR - TOL + gamma
    
    r_side = 50 + gamma
    
    if verbose:
        print(f"Formation parameters:")
        print(f"  Target distance: {BAR}")
        print(f"  Tolerance: {TOL}")
        print(f"  Min/Max distance: {MIN_DIST}/{MAX_DIST}")
        print(f"  Robot radius: {robot_radius}")
        print(f"  Intra-robot distance: {intra_robot_distance}")
    
    # Convert arrays to numpy if needed
    array_a = np.asarray(array_a)
    array_b = np.asarray(array_b)
    
    len_a = len(array_a)
    len_b = len(array_b)
    
    if verbose:
        print(f"Array lengths: A={len_a}, B={len_b}")
    
    # Build configuration graph
    if verbose:
        print("Building configuration graph...")
        graph_start_time = time.time()
    
    # 3D valid map: [i, j, k, point_pair, coordinate]
    # k=0: spread from A, k=1: fit, k=2: spread from B
    valid_map = np.zeros((len_a, len_b, 3, 2, 2), dtype=int)
    G = nx.Graph()
    
    # Pre-compute arrays as float32 for better performance
    array_a_np = np.asarray(array_a, dtype=np.float32)
    array_b_np = np.asarray(array_b, dtype=np.float32)
    
    # Vectorized distance computation for all pairs
    if verbose:
        print("Computing distance matrix...")
    
    diff = array_a_np[:, None, :] - array_b_np[None, :, :]  # Shape: (len_a, len_b, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)  # Shape: (len_a, len_b)
    
    # Create masks for different distance ranges
    too_close_mask = dist_matrix <= MIN_DIST
    too_far_mask = dist_matrix >= MAX_DIST
    valid_range_mask = ~too_close_mask & ~too_far_mask
    
    # Get indices for each category
    too_close_indices = np.where(too_close_mask)
    valid_range_indices = np.where(valid_range_mask)
    
    if verbose:
        print(f"Processing {len(too_close_indices[0])} close pairs and {len(valid_range_indices[0])} valid range pairs")
        print(f"Skipping {np.sum(too_far_mask)} too-far pairs")
    
    # Process pairs that are too close (need spreading)
    for idx in range(len(too_close_indices[0])):
        i, j = too_close_indices[0][idx], too_close_indices[1][idx]
        
        # Try spreading from A to meet minimum distance
        is_ok, A, B = spread_oobb(array_a[i], array_b[j], robot_radius, intra_robot_distance, MIN_BOUND, collision_map, step=step_size, max_shift=r_side)
        if is_ok:
            valid_map[i, j, 0] = [A, B]
            check_neighbors(i, j, 0, valid_map, len_a, len_b, array_a, array_b, G)

        # Try spreading from B to meet minimum distance
        is_ok, B, A = spread_oobb(array_b[j], array_a[i], robot_radius, intra_robot_distance, MIN_BOUND, collision_map, step=step_size, max_shift=r_side)
        if is_ok:
            valid_map[i, j, 2] = [A, B]
            check_neighbors(i, j, 2, valid_map, len_a, len_b, array_a, array_b, G)

    # Process pairs in valid distance range (need fitting)
    for idx in range(len(valid_range_indices[0])):
        i, j = valid_range_indices[0][idx], valid_range_indices[1][idx]
        
        # Distance in valid range, try fitting
        is_ok, A, B = fit_oobb(array_a[i], array_b[j], robot_radius, intra_robot_distance, min_dist=MIN_DIST, min_decr=r_side, map=collision_map, step=step_size)
        if is_ok:
            valid_map[i, j, 1] = [A, B]
            check_neighbors(i, j, 1, valid_map, len_a, len_b, array_a, array_b, G)

    if verbose:
        print(f"Graph nodes: {G.number_of_nodes()}")
        print(f"Graph edges: {G.number_of_edges()}")
        print(f"Graph building time: {time.time() - graph_start_time:.2f}s")
    
    # Run A* search
    if verbose:
        print("Starting A* search...")
        search_start_time = time.time()
    
    try:
        path = astar_heapq(
            G,
            start=(start_a, start_b),
            goal=(goal_a, goal_b),
            heuristic=default_astar_heuristic,
            weight='weight'
        )
        
        if verbose:
            print("Path found successfully!")
            print(f"Search time: {time.time() - search_start_time:.2f}s")
            print(f"Total time: {time.time() - start_time:.2f}s")
            print(f"Path length: {len(path)} waypoints")
        
        return path
        
    except RuntimeError as e:
        if verbose:
            print(f"Path finding failed: {e}")
        return None


def create_obstacle_arrays_from_image(
    obstacle_image_path: str,
    inflation_radius: int = 55,
    step_size: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create obstacle boundary arrays from an image file.
    
    Args:
        obstacle_image_path: Path to the obstacle image file
        inflation_radius: Radius for obstacle inflation
        step_size: Step size for boundary sampling
        
    Returns:
        Tuple of (array_a, array_b) representing the two obstacle boundaries
    """
    obstacle = Obstacle(obstacle_image_path, inflation_radius=inflation_radius, step_size=step_size)
    return obstacle.split_obstacle()


# ===============================
# Example Usage
# ===============================

if __name__ == "__main__":
    # Example usage of the path planning function
    
    # Load obstacle and create boundary arrays
    map_path = "maps/mp_final.png"
    wall_path = "walls/wall_final.png"
    
    # Create obstacle arrays
    array_a, array_b = create_obstacle_arrays_from_image(
        obstacle_image_path=map_path,
        inflation_radius=55,
        step_size=10
    )
    
    # Load collision map
    collision_map = cv2.imread(wall_path, cv2.IMREAD_GRAYSCALE)
    
    # Define formation parameters
    BAR = 300
    TOL = 0.2 * BAR
    gamma = 5
    
    # Find start and goal positions
    id_a_start, id_b_start = find_start(array_a, array_b, BAR, TOL)
    
    # For goal, use reversed arrays to find end positions
    array_a_inv = array_a[::-1]
    array_b_inv = array_b[::-1]
    id_a_end_inv, id_b_end_inv = find_start(array_a_inv, array_b_inv, BAR, TOL)
    
    start_a = tuple(array_a[id_a_start])
    start_b = tuple(array_b[id_b_start])
    goal_a = tuple(array_a_inv[id_a_end_inv])
    goal_b = tuple(array_b_inv[id_b_end_inv])
    
    print(f"Start positions: A={start_a}, B={start_b}")
    print(f"Goal positions: A={goal_a}, B={goal_b}")
    
    # Find path
    path = find_dual_robot_path(
        array_a=array_a,
        array_b=array_b,
        start_a=start_a,
        start_b=start_b,
        goal_a=goal_a,
        goal_b=goal_b,
        collision_map=collision_map,
        robot_radius=15,
        intra_robot_distance=35,
        formation_distance=300,
        formation_tolerance=0.2,
        gamma=5,
        step_size=1,
        verbose=True
    )
    
    if path:
        print(f"Path planning successful! Found path with {len(path)} waypoints.")
        
        # Print first few waypoints
        print("First 5 waypoints:")
        for i, (pt_a, pt_b) in enumerate(path[:5]):
            print(f"  {i+1}: A={pt_a}, B={pt_b}")
    else:
        print("Path planning failed!")