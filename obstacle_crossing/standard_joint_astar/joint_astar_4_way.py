from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional

def load_grid_from_image(image_path, threshold=128):
    """Load a binary occupancy grid from an image.

    Returns a numpy array with values: 0 = free, 1 = occupied.
    """
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    grid = (arr > threshold).astype(np.uint8)
    grid = 1 - grid  # 0: free, 1: occupied
    return grid  # keep as numpy array for faster indexing

# Note: agents move in 4 directions (up/down/left/right):
# cost 1 for orthogonal moves and 0 for staying in place (wait).
# Available heuristics do not use precomputation and are evaluated on-the-fly.

def _precompute_neighbors(grid):
    """Precompute 4-connected neighbors (including wait) for every free cell.

    Returns:
        neighbors_by_id: list where index is cell id (x*W+y) and value is list of neighbor ids.
        pos_x, pos_y: lists mapping id -> coordinates for fast access.
    """
    H, W = grid.shape
    def enc(x, y):
        return x * W + y

    # Precompute coordinate lookup for ids
    size = H * W
    pos_x = [i // W for i in range(size)]
    pos_y = [i % W for i in range(size)]

    # 4-neighbors: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    neighbors_by_id = [[] for _ in range(size)]
    for x in range(H):
        for y in range(W):
            if grid[x, y] != 0:
                # occupied -> no neighbors
                continue
            cid = enc(x, y)
            lst = [cid]  # include wait (0,0)
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx, ny] == 0:
                    lst.append(enc(nx, ny))
            neighbors_by_id[cid] = lst
    return neighbors_by_id, pos_x, pos_y

import heapq
def joint_astar(
    grid,
    start1,
    goal1,
    start2,
    goal2,
    min_dist,
    max_dist,
    *,
    heuristic: str = "manhattan",
):
    """Optimized Joint A* for two agents with distance constraints.

    - Movement costs: 1 (orthogonal), 0 (wait). No diagonals (4-neighbors)
    - Uses squared Euclidean distance to avoid sqrt in constraint check for constraints
    - Precomputes neighbors and coordinate lookups
    - Encodes states as integer ids for faster hashing/comparisons
    - Keeps came_from and g-score to avoid storing full paths in the open set

    Args:
        grid: 2D list or numpy array (0=free, 1=occupied)
        start1, goal1, start2, goal2: tuple (x,y)
        min_dist, max_dist: float, allowed euclidean distance between agents (inclusive)
        heuristic: 'manhattan' (recommended, admissible) or 'chebyshev' (admissible but less informative).
    """
    if isinstance(grid, list):
        grid = np.array(grid, dtype=np.uint8)

    H, W = grid.shape
    def enc(p):
        return p[0] * W + p[1]
    def dec(i):
        return (i // W, i % W)

    neighbors_by_id, pos_x, pos_y = _precompute_neighbors(grid)

    start1_id, goal1_id = enc(start1), enc(goal1)
    start2_id, goal2_id = enc(start2), enc(goal2)

    # Early exit if start/goal are blocked
    if grid[start1[0], start1[1]] or grid[start2[0], start2[1]] or \
       grid[goal1[0], goal1[1]] or grid[goal2[0], goal2[1]]:
        return None

    # Heuristic (selectable). Default: manhattan for 4-connected movement
    gx1, gy1 = pos_x[goal1_id], pos_y[goal1_id]
    gx2, gy2 = pos_x[goal2_id], pos_y[goal2_id]

    if heuristic not in ("manhattan", "chebyshev"):
        raise ValueError("heuristic must be one of: 'manhattan', 'chebyshev'")

    if heuristic == "manhattan":
        # Manhattan distance: |dx| + |dy|
        def h_of(i1, i2, _pos_x=pos_x, _pos_y=pos_y, _gx1=gx1, _gy1=gy1, _gx2=gx2, _gy2=gy2):
            dx1 = abs(_pos_x[i1] - _gx1)
            dy1 = abs(_pos_y[i1] - _gy1)
            dx2 = abs(_pos_x[i2] - _gx2)
            dy2 = abs(_pos_y[i2] - _gy2)
            return (dx1 + dy1) + (dx2 + dy2)
    elif heuristic == "chebyshev":
        # Chebyshev distance: max(|dx|, |dy|)
        def h_of(i1, i2, _pos_x=pos_x, _pos_y=pos_y, _gx1=gx1, _gy1=gy1, _gx2=gx2, _gy2=gy2):
            dx1 = abs(_pos_x[i1] - _gx1)
            dy1 = abs(_pos_y[i1] - _gy1)
            dx2 = abs(_pos_x[i2] - _gx2)
            dy2 = abs(_pos_y[i2] - _gy2)
            return max(dx1, dy1) + max(dx2, dy2)

    # Movement cost for 4-directions: 0 for wait, 1 for orthogonal step
    def step_cost(a, b, _px=pos_x, _py=pos_y):
        if a == b:
            return 0.0
        # With only 4-neighbors, dx+dy is always 1 when moving
        return 1.0

    min2 = float(min_dist) * float(min_dist)
    max2 = float(max_dist) * float(max_dist)

    # Priority queue items: (f, g, id1, id2)
    open_set = []
    start_state = (start1_id, start2_id)
    start_h = h_of(start1_id, start2_id)
    heapq.heappush(open_set, (start_h, 0, start1_id, start2_id))

    came_from = {}
    g_score = {start_state: 0}

    # Helper to compute squared distance between two ids
    def dist2(i, j):
        dx = pos_x[i] - pos_x[j]
        dy = pos_y[i] - pos_y[j]
        return dx * dx + dy * dy

    # Check that the initial state respects the constraint
    if not (min2 <= dist2(start1_id, start2_id) <= max2):
        # If the initial positions violate the constraint, no solution
        return None

    goal_state = (goal1_id, goal2_id)

    while open_set:
        f, g, id1, id2 = heapq.heappop(open_set)
        state = (id1, id2)

        # Skip if we have already found a better path to this state
        if g > g_score.get(state, float('inf')):
            continue

        if state == goal_state:
            # Reconstruct path
            path = []
            cur = state
            while cur in came_from:
                a, b = cur
                path.append((dec(a), dec(b)))
                cur = came_from[cur]
            # include the start
            a, b = cur
            path.append((dec(a), dec(b)))
            path.reverse()
            return path

        nb1 = neighbors_by_id[id1]
        nb2 = neighbors_by_id[id2]

        # Localize for speed
        get_g = g_score.get
        push = heapq.heappush
        h_of_local = h_of

        for n1 in nb1:
            for n2 in nb2:
                if n1 == n2:
                    # same cell conflict
                    continue
                d2 = dist2(n1, n2)
                if not (min2 <= d2 <= max2):
                    continue
                ns = (n1, n2)
                # Compute joint step cost as sum of per-agent movement costs
                c = step_cost(id1, n1) + step_cost(id2, n2)
                g_next = g + c
                if g_next < get_g(ns, float('inf')):
                    g_score[ns] = g_next
                    came_from[ns] = state
                    h = h_of_local(n1, n2)
                    push(open_set, (g_next + h, g_next, n1, n2))
    return None

def plot_paths(
    grid,
    path,
    start1,
    goal1,
    start2,
    goal2,
    *,
    rotate_ccw_90: bool = False,
    show_legend: bool = True,
    y_min: Optional[int] = None,
    y_max: Optional[int] = None,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
):
    arr = np.array(grid)
    H0, W0 = arr.shape

    # Vertical crop (rows)
    if y_min is not None or y_max is not None:
        crop_y0 = 0 if y_min is None else max(0, int(y_min))
        crop_y1 = H0 if y_max is None else min(H0, int(y_max))
        if crop_y0 >= crop_y1:
            raise ValueError("y_min must be < y_max and within image bounds")
        arr = arr[crop_y0:crop_y1, :]
    else:
        crop_y0, crop_y1 = 0, H0

    # Horizontal crop (columns)
    if x_min is not None or x_max is not None:
        crop_x0 = 0 if x_min is None else max(0, int(x_min))
        crop_x1 = W0 if x_max is None else min(W0, int(x_max))
        if crop_x0 >= crop_x1:
            raise ValueError("x_min must be < x_max and within image bounds")
        arr = arr[:, crop_x0:crop_x1]
    else:
        crop_x0, crop_x1 = 0, W0

    # Dimensions after crop
    H, W = arr.shape

    # Rotation
    arr_to_show = np.rot90(arr) if rotate_ccw_90 else arr

    plt.figure(figsize=(8,8))
    plt.imshow(arr_to_show, cmap='gray_r')

    def rot_xy(x, y):
        # Map old (row=x, col=y) to new after 90° CCW rotation
        return (W - 1 - y, x)

    def adj_point(x, y):
        # If inside crop, return relative coordinates, otherwise None
        if (crop_y0 <= x < crop_y1) and (crop_x0 <= y < crop_x1):
            return (x - crop_y0, y - crop_x0)
        return None

    def adj_series(xs, ys):
        # Apply crop on rows and columns; out of range -> NaN to break lines
        out_x, out_y = [], []
        for x, y in zip(xs, ys):
            if (crop_y0 <= x < crop_y1) and (crop_x0 <= y < crop_x1):
                out_x.append(x - crop_y0)
                out_y.append(y - crop_x0)
            else:
                out_x.append(np.nan)
                out_y.append(np.nan)
        return out_x, out_y

    # Agent 1 path
    a1_x = [p[0][0] for p in path]
    a1_y = [p[0][1] for p in path]
    a1_x, a1_y = adj_series(a1_x, a1_y)
    if rotate_ccw_90:
        rot = [rot_xy(x, y) if not (np.isnan(x) or np.isnan(y)) else (np.nan, np.nan) for x, y in zip(a1_x, a1_y)]
        a1_x, a1_y = zip(*rot) if rot else ([], [])
    plt.plot(a1_y, a1_x, 'r-', label='Agent 1 Path')

    # Agent 1 markers
    pnt = adj_point(*start1)
    if pnt is not None:
        sx1, sy1 = pnt
        if rotate_ccw_90:
            sx1, sy1 = rot_xy(sx1, sy1)
        plt.plot(sy1, sx1, 'r*', label='Agent 1 Start', markersize=10)
    pnt = adj_point(*goal1)
    if pnt is not None:
        gx1, gy1 = pnt
        if rotate_ccw_90:
            gx1, gy1 = rot_xy(gx1, gy1)
        plt.plot(gy1, gx1, 'ro', label='Agent 1 Goal')

    # Agent 2 path
    a2_x = [p[1][0] for p in path]
    a2_y = [p[1][1] for p in path]
    a2_x, a2_y = adj_series(a2_x, a2_y)
    if rotate_ccw_90:
        rot = [rot_xy(x, y) if not (np.isnan(x) or np.isnan(y)) else (np.nan, np.nan) for x, y in zip(a2_x, a2_y)]
        a2_x, a2_y = zip(*rot) if rot else ([], [])
    plt.plot(a2_y, a2_x, 'b-', label='Agent 2 Path')

    # Agent 2 markers
    pnt = adj_point(*start2)
    if pnt is not None:
        sx2, sy2 = pnt
        if rotate_ccw_90:
            sx2, sy2 = rot_xy(sx2, sy2)
        plt.plot(sy2, sx2, 'b*', label='Agent 2 Start', markersize=10)
    pnt = adj_point(*goal2)
    if pnt is not None:
        gx2, gy2 = pnt
        if rotate_ccw_90:
            gx2, gy2 = rot_xy(gx2, gy2)
        plt.plot(gy2, gx2, 'bo', label='Agent 2 Goal')

    if show_legend:
        plt.legend()
    ttl = "Joint A* (4-way) Paths with Distance Constraint"
    plt.title(ttl)
    plt.show()

if __name__ == "__main__":
    # Load grid from image
    grid = load_grid_from_image("maps_dilated/map_slim.png")  # <-- Replace with your image path

    # Set start/goal positions
    
    # map_slim.png
    goal1 = (3, 30)
    start1 = (196, 25)
    goal2 = (3, 60)
    start2 = (196, 55)

    """# map_large.png
    goal1 = (3, 5)
    start1 = (198, 85)
    goal2 = (3, 35)
    start2 = (198, 115)"""

    min_dist = 24.0
    max_dist = 36.0

    # Heuristic selection: 'manhattan' or 'chebyshev'
    heuristic = "manhattan"  # or "chebyshev"
    print(f"Heuristic: {heuristic}")

    # Plan
    start_time = time.time()
    path = joint_astar(
        grid,
        start1,
        goal1,
        start2,
        goal2,
        min_dist,
        max_dist,
        heuristic=heuristic,
    )

    # Visualize
    rotate_ccw_90 = True   # set to True to rotate 90° counterclockwise
    show_legend = False    # set to False to hide the legend
    crop_y_min = None      # e.g. 30
    crop_y_max = None      # e.g. 120
    crop_x_min = None      # e.g. 10
    crop_x_max = None      # e.g. 200
    if path:
        print("Elapsed time (s): %.6f" % (time.time() - start_time))
        plot_paths(
            grid,
            path,
            start1,
            goal1,
            start2,
            goal2,
            rotate_ccw_90=rotate_ccw_90,
            show_legend=show_legend,
            y_min=crop_y_min,
            y_max=crop_y_max,
            x_min=crop_x_min,
            x_max=crop_x_max,
        )
    else:
        print("No path found.")