import math, heapq
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import count
from libraries.navlib import *
from robot_formation import draw_robot_formation

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

# --- Main execution ---
if __name__ == "__main__":

    start_time = time.time()

    # Map parameters
    map_path = "maps/mp_slim.png"
    wall_path = "maps/mp_slim.png"
    #wall_path = "walls/wall_slim.png"

    # Visualization parameters
    save_to_file = True
    show = False

    # Formation parameters
    BAR = 300
    TOL = 0.2* BAR
    step_size = 10
    gamma = 5

    MAX_DIST = BAR + TOL
    MIN_DIST = BAR - TOL

    MAX_BOUND = BAR + TOL
    MIN_BOUND = BAR - TOL + gamma

    print("BAR:", BAR, "TOL:", TOL)
    print("Min bound:", MIN_BOUND, "Max bound:", MAX_BOUND)
    print("Min dist:", MIN_DIST, "Max dist:", MAX_DIST)

    robot_r = 15
    intra_robot_dist = 35
    r_side = 50
    print("Side-formation radius:", r_side)

    map = cv2.imread(wall_path, cv2.IMREAD_GRAYSCALE)
    map_rgb = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)

    obstacle = Obstacle(map_path, inflation_radius=r_side + gamma, step_size=step_size)
    # obstacle.show()
    image = obstacle.get_debug_image()

    # Split obstacle into two point arrays
    array_a, array_b = obstacle.split_obstacle()
    array_a_inv = array_a[::-1]
    array_b_inv = array_b[::-1]

    length_a = len(array_a)
    length_b = len(array_b)

    # Find start/end indices
    id_a_start, id_b_start = find_start(array_a, array_b, BAR, TOL)
    id_b_start_2, id_a_start_2 = find_start(array_b, array_a, BAR, TOL)
    id_a_end_inv, id_b_end_inv = find_start(array_a_inv, array_b_inv, BAR, TOL)
    id_b_end_2_inv, id_a_end_2_inv = find_start(array_b_inv, array_a_inv, BAR, TOL)
    id_a_end = length_a - 1 - id_a_end_inv
    id_b_end = length_b - 1 - id_b_end_inv
    id_a_end_2 = length_a - 1 - id_a_end_2_inv
    id_b_end_2 = length_b - 1 - id_b_end_2_inv

    # Start/end coordinates
    a_start = tuple(array_a[id_a_start])
    b_start = tuple(array_b[id_b_start])
    a_start_2 = tuple(array_a[id_a_start_2])
    b_start_2 = tuple(array_b[id_b_start_2])
    a_end = tuple(array_a_inv[id_a_end_inv])
    b_end = tuple(array_b_inv[id_b_end_inv])
    a_end_2 = tuple(array_a_inv[id_a_end_2_inv])
    b_end_2 = tuple(array_b_inv[id_b_end_2_inv])

    len_a = length_a
    len_b = length_b
    
    start = (id_a_start, id_b_start)
    # Goal specified as indices on both curves
    goal_indices = (id_a_end, id_b_end)
    
    print("Length A:", len_a, "Length B:", len_b)
    print("Start indices:", start, "Goal indices:", goal_indices)
    print("Euclid dist start:", euclid(a_start, b_start))
    print("Euclid dist goal target:", euclid(a_end, b_end))

    start_time_graph_build = time.time()

    # 3D valid map: [i, j, k, point_pair, coordinate]
    # k=0: spread from A, k=1: fit, k=2: spread from B
    valid_map = np.zeros((len_a, len_b, 3, 2, 2), dtype=int)

    print("Time to preprocess (s):", time.time() - start_time)

    print("Starting graph building...")

    G = nx.Graph()

    # Pre-compute arrays as float32 for better performance
    array_a_np = np.asarray(array_a, dtype=np.float32)
    array_b_np = np.asarray(array_b, dtype=np.float32)
    
    # Vectorized distance computation for all pairs
    print("Computing distance matrix...")
    # Broadcasting: array_a_np[:, None, :] shape (len_a, 1, 2), array_b_np[None, :, :] shape (1, len_b, 2)
    diff = array_a_np[:, None, :] - array_b_np[None, :, :]  # Shape: (len_a, len_b, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)  # Shape: (len_a, len_b)
    
    # Pre-filter valid pairs to reduce processing
    valid_pairs = []
    too_close_pairs = []
    
    # Create masks for different distance ranges
    too_close_mask = dist_matrix <= MIN_DIST
    too_far_mask = dist_matrix >= MAX_DIST
    valid_range_mask = ~too_close_mask & ~too_far_mask
    
    # Get indices for each category
    too_close_indices = np.where(too_close_mask)
    valid_range_indices = np.where(valid_range_mask)
    
    print(f"Processing {len(too_close_indices[0])} close pairs and {len(valid_range_indices[0])} valid range pairs")
    print(f"Skipping {np.sum(too_far_mask)} too-far pairs")

    # Process pairs that are too close (need spreading)
    for idx in range(len(too_close_indices[0])):
        i, j = too_close_indices[0][idx], too_close_indices[1][idx]
        
        # Try spreading from A to meet minimum distance
        is_ok, A, B = spread_oobb(array_a[i], array_b[j], robot_r, intra_robot_dist, MIN_BOUND, map, step=1, max_shift=r_side)
        if is_ok:
            valid_map[i, j, 0] = [A, B]
            check_neighbors(i, j, 0, valid_map, len_a, len_b, array_a, array_b, G)

        # Try spreading from B to meet minimum distance
        is_ok, B, A = spread_oobb(array_b[j], array_a[i], robot_r, intra_robot_dist, MIN_BOUND, map, step=1, max_shift=r_side)
        if is_ok:
            valid_map[i, j, 2] = [A, B]
            check_neighbors(i, j, 2, valid_map, len_a, len_b, array_a, array_b, G)

    # Process pairs in valid distance range (need fitting)
    for idx in range(len(valid_range_indices[0])):
        i, j = valid_range_indices[0][idx], valid_range_indices[1][idx]
        
        # Distance in valid range, try fitting
        is_ok, A, B = fit_oobb(array_a[i], array_b[j], robot_r, intra_robot_dist, min_dist=MIN_DIST, min_decr=r_side, map=map, step=1)
        if is_ok:
            valid_map[i, j, 1] = [A, B]
            check_neighbors(i, j, 1, valid_map, len_a, len_b, array_a, array_b, G)

    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())
    print("Time to build graph (s):", time.time() - start_time_graph_build)

    start_time_path_search = time.time()
    print("Starting graph search...")

    # Run A* search
    try:
        path = astar_heapq(
            G,
            start=(a_start, b_start),
            goal=(a_end, b_end),
            heuristic=default_astar_heuristic,
            weight='weight'
        )
        print("Path found successfully!")
        print("Time to search path (s):", time.time() - start_time_path_search)
        print("Total execution time (s):", time.time() - start_time)

        # Visualize results
        img = map_rgb.copy()

        # Mark start and end points
        cv2.circle(img, a_end, 24, (255, 0, 0), -1)
        cv2.circle(img, b_end, 24, (0, 0, 255), -1)
        cv2.drawMarker(img, a_start, (255, 0, 0), markerType=cv2.MARKER_STAR, markerSize=45, thickness=6)
        cv2.drawMarker(img, b_start, (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=45, thickness=6)

        # Draw obstacle boundary points
        for pt in array_a:
            cv2.circle(img, tuple(pt), 3, (255, 0, 0), -1)
        for pt in array_b:
            cv2.circle(img, tuple(pt), 3, (0, 0, 255), -1)
        
        # Draw path
        for i in range(len(path) - 1):
            pts_from = path[i]
            pts_to = path[i+1]
            cv2.arrowedLine(img, pts_from[0], pts_to[0], color=(255, 0, 0), thickness=6)
            cv2.arrowedLine(img, pts_from[1], pts_to[1], color=(0, 0, 255), thickness=6)

        # Crop and display image
        img_bk = img.copy()
        img = img[400:2600, 350:1170]  # slim view
        #img = img[400:2600, 500:2500]  # large view
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plt.figure(figsize=(8,8))
        plt.imshow(img)
        plt.show()

        # Step-by-step visualization of robot formations
        for pt in path:
            img_temp = img_bk.copy()
            #display_fit(pt[0], pt[1], robot_r, intra_robot_dist, img_temp)
            img = draw_robot_formation(image=img_temp, point_A=pt[0], point_B=pt[1], r_rob=robot_r, d_rob=intra_robot_dist)
            cv2.imshow("Formation", img)
            cv2.waitKey(0)    

    except Exception as e:
        print(f"Path finding failed: {e}")
        path = None

