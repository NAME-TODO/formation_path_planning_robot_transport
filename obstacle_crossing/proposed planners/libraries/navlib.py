from __future__ import annotations

import cv2
import math
import numpy as np
from shapely.geometry import Polygon, Point
from skimage.morphology import skeletonize
import networkx as nx


# ===============================
# Geometry helpers
# ===============================
def find_longest_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
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


def find_split_point(contour: np.ndarray, point: tuple[int, int], radius: float) -> tuple[int, int]:
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
# Obstacle handling
# ===============================
class Obstacle:
    """Loads an obstacle from an image, inflates it, samples boundary, and extracts main skeleton branch."""

    def __init__(self, path: str, inflation_radius: int = 40, step_size: int = 10, scale: float = 1):
        self._path = path
        self._inflation_radius = inflation_radius
        self._step_size = step_size
        self._scale = scale

        self._image_gray: np.ndarray | None = None
        self._debug_image: np.ndarray | None = None
        self._obstacle_polygon_true: Polygon | None = None
        self._obstacle_polygon: Polygon | None = None
        self._sampled: np.ndarray | None = None
        self._main_skeleton_branch: list[tuple[int, int]] | None = None

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

    # --- Public API ---
    def get_polygon(self) -> Polygon:
        assert self._obstacle_polygon is not None
        return self._obstacle_polygon

    def get_samples(self) -> np.ndarray:
        assert self._sampled is not None
        return self._sampled

    def get_raw_image(self) -> np.ndarray:
        assert self._image_gray is not None
        return self._image_gray

    def get_debug_image(self) -> np.ndarray:
        assert self._debug_image is not None
        self.show(save=True)
        return self._debug_image.copy()

    def split_obstacle(self) -> tuple[np.ndarray, np.ndarray]:
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

    def show(self, save: bool = False) -> None:
        """Render obstacle visualization. If save=True keep a copy for get_debug_image."""
        assert self._debug_image is not None and self._obstacle_polygon_true is not None and self._sampled is not None
        image = np.ones_like(self._debug_image) * 255

        # Draw original obstacle
        coords = np.array(list(self._obstacle_polygon_true.exterior.coords), dtype=np.int32)
        cv2.fillPoly(image, [coords], color=(60, 60, 60))

        # Draw split samples
        array_lx, array_rx = self.split_obstacle()
        for pt_lx in array_lx:
            cv2.circle(image, tuple(pt_lx), radius=7, color=(255, 60, 60), thickness=-1)
        for pt_rx in array_rx:
            cv2.circle(image, tuple(pt_rx), radius=7, color=(60, 60, 255), thickness=-1)

        # Draw skeleton main branch
        assert self._main_skeleton_branch is not None
        for point in self._main_skeleton_branch:
            cv2.circle(image, tuple(point), radius=4, color=(60, 255, 60), thickness=-1)

        # Mark split points
        center_start = self._main_skeleton_branch[0]
        center_end = self._main_skeleton_branch[-1]
        radius = self._inflation_radius + 100
        start_point = find_split_point(self._sampled, center_start, radius)
        end_point = find_split_point(self._sampled, center_end, radius)
        cv2.circle(image, start_point, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(image, end_point, radius=5, color=(0, 0, 255), thickness=-1)

        if save:
            self._debug_image = image.copy()
        else:
            cv2.imshow('VISUALIZE OBSTACLE', image)
            cv2.waitKey(0)


def find_start(array_a: np.ndarray, array_b: np.ndarray, BAR: float, TOL: float) -> tuple[int, int | None]:
    """Find the first index pair (0, j) such that distance between array_a[0] and array_b[j] is ~ BAR within tolerance TOL."""
    id_a = 0
    id_b = None
    for j in range(len(array_b)):
        distance = np.linalg.norm(array_a[id_a] - array_b[j])
        if abs(distance - BAR) < TOL * 1:
            id_b = j
            break
    return id_a, id_b


# ===============================
# Formation fit/spread utilities
# ===============================
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


def display_fit(A, B, robot_r, deg, image, balck: bool = True):
    """Draw the two mirrored OOBBs and the 3 robots for each group on the given image."""
    img = image.copy()
    rect_fitted, rect2_fitted = get_oobb(A, B, robot_r, deg)
    pts1 = rect_fitted.astype(np.int32).reshape((-1, 1, 2))
    pts2 = rect2_fitted.astype(np.int32).reshape((-1, 1, 2))

    if balck:
        colorA = colorB = (0, 0, 0)
    else:
        colorA = (0, 255, 255)
        colorB = (255, 255, 0)

    cv2.circle(img, (int(A[0]), int(A[1])), 3, colorA, -1)
    cv2.circle(img, (int(B[0]), int(B[1])), 3, colorB, -1)
    cv2.polylines(img, [pts1], True, colorA, 2)
    cv2.polylines(img, [pts2], True, colorB, 2)

    # Draw 3 robots for each group
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)
    C = (A_np + B_np) / 2.0
    r = np.linalg.norm(B_np - A_np) / 2.0
    if r > 1e-9:
        d = float(deg)
        ratio = max(0.0, min(1.0, d / (2.0 * r)))
        alpha = 2.0 * np.arcsin(ratio)
        angA = np.arctan2(A_np[1] - C[1], A_np[0] - C[0])
        angB = angA + np.pi
        A1 = C + r * np.array([np.cos(angA + alpha), np.sin(angA + alpha)])
        A2 = C + r * np.array([np.cos(angA - alpha), np.sin(angA - alpha)])
        B1 = C + r * np.array([np.cos(angB + alpha), np.sin(angB + alpha)])
        B2 = C + r * np.array([np.cos(angB - alpha), np.sin(angB - alpha)])
        rad = max(1, int(robot_r))
        for p in (A_np, A1, A2):
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), rad, colorA, -1)
        for p in (B_np, B1, B2):
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), rad, colorB, -1)

    cv2.imshow('fit', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def muovi_A_lungo_AB_fino_spostamento(A, B, robot_r, deg, map, step, max_spostamento):
    """Translate A and B together along AB by step until both OOBBs are collision-free or max shift reached."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    
    diff = B - A
    dist = np.linalg.norm(diff)
    
    if dist == 0:
        return False, None, None
    
    direction = diff / dist
    max_steps = int(max_spostamento / step)
    
    # Pre-compute step vector for efficiency
    step_vector = direction * step
    
    for i in range(max_steps + 1):
        offset = step_vector * i
        new_A = A + offset
        new_B = B + offset
        
        # Early exit if we've moved too far
        moved = np.linalg.norm(offset)
        if moved > max_spostamento:
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
        return muovi_A_lungo_AB_fino_spostamento(A, new_B, robot_r, deg, map, step=step, max_spostamento=max_shift)
    
    return False, None, None
