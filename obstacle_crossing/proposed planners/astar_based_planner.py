import math, heapq
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from libraries.navlib import *

def euclid(p, q):
    """Euclidean distance between two 2D points (x, y)."""
    return math.hypot(p[0]-q[0], p[1]-q[1])

def build_valid_matrix(A, B, BAR, TOL_percent, return_numpy=True):
    """Build a boolean matrix valid[j,i] where distance(A[i], B[j]) is within BAR±TOL%."""
    A_arr = np.asarray(A, dtype=np.float32)
    B_arr = np.asarray(B, dtype=np.float32)
    low = BAR * (1 - TOL_percent/100.0)
    high = BAR * (1 + TOL_percent/100.0)
    low2 = low*low; high2 = high*high
    diff = B_arr[:, None, :] - A_arr[None, :, :]
    dist2 = (diff[...,0]**2 + diff[...,1]**2)
    valid = (dist2 >= low2) & (dist2 <= high2)
    return valid if return_numpy else valid.tolist()

def astar_product(
    A, B, BAR, TOL_percent, start, goal,
    use_numpy_state=True,
    local_precompute=True
):
    """A* on the Cartesian product of two closed curves A and B (indices wrap-around).

    - valid is computed vectorized as a numpy bool array
    - g_score and closed are arrays for O(1) access
    - parents stored via a flattened state index
    - heuristic is the minimal arc length distance along each curve to the goal indices
    - local step distances (±1 per curve) optionally precomputed for speed
    Goal is a pair of indices (gi, gj).
    """
    A_arr = np.asarray(A, dtype=np.float32)
    B_arr = np.asarray(B, dtype=np.float32)
    n = len(A_arr); m = len(B_arr)

    # Build valid matrix (vectorized)
    low = BAR * (1 - TOL_percent/100.0)
    high = BAR * (1 + TOL_percent/100.0)
    if use_numpy_state:
        diff = B_arr[:, None, :] - A_arr[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        valid = (dist >= low) & (dist <= high)
    else:
        valid = build_valid_matrix(A, B, BAR, TOL_percent)

    si, sj = start

    # Goal (indices only) & heuristic precompute
    gi, gj = goal
    if not (0 <= gi < n and 0 <= gj < m):
        return None, math.inf
    if use_numpy_state:
        if not valid[gj, gi]:
            return None, math.inf
    else:
        if not valid[gj][gi]:
            return None, math.inf
    # Heuristic based on minimal arc-length on closed curves
    # Precompute consecutive edge lengths and cumulative sums
    A_edges = np.linalg.norm(A_arr[(np.arange(n)+1) % n] - A_arr, axis=1)
    B_edges = np.linalg.norm(B_arr[(np.arange(m)+1) % m] - B_arr, axis=1)
    A_cum = np.concatenate(([0.0], np.cumsum(A_edges)))
    B_cum = np.concatenate(([0.0], np.cumsum(B_edges)))
    A_total = float(A_cum[-1])
    B_total = float(B_cum[-1])

    def arc_len_A(i0, i1):
        if i1 >= i0:
            forward = A_cum[i1] - A_cum[i0]
        else:
            forward = (A_total - A_cum[i0]) + A_cum[i1]
        backward = A_total - forward
        return forward if forward <= backward else backward

    def arc_len_B(j0, j1):
        if j1 >= j0:
            forward = B_cum[j1] - B_cum[j0]
        else:
            forward = (B_total - B_cum[j0]) + B_cum[j1]
        backward = B_total - forward
        return forward if forward <= backward else backward

    def h_func(i, j):
        return arc_len_A(i, gi) + arc_len_B(j, gj)
    def is_goal(i, j):
        return (i, j) == (gi, gj)

    # Validate start
    if not (0 <= si < n and 0 <= sj < m):
        return None, math.inf
    if use_numpy_state:
        if not valid[sj, si]:
            return None, math.inf
    else:
        if not valid[sj][si]:
            return None, math.inf

    # Local distance precompute for offsets -1,+1 (wrap) for A and B
    if local_precompute:
        A_prev = np.linalg.norm(A_arr - A_arr[np.arange(n)-1], axis=1)
        A_next = np.linalg.norm(A_arr - A_arr[(np.arange(n)+1) % n], axis=1)
        B_prev = np.linalg.norm(B_arr - B_arr[np.arange(m)-1], axis=1)
        B_next = np.linalg.norm(B_arr - B_arr[(np.arange(m)+1) % m], axis=1)
    else:
        A_prev = A_next = B_prev = B_next = None

    def local_edge_cost(i, j, ni, nj):
        costA = 0.0; costB = 0.0
        if i != ni:
            di = (ni - i)
            if di == 1 or di == 1 - n:
                costA = A_next[i]
            elif di == -1 or di == n-1:
                costA = A_prev[i]
            else:
                costA = euclid(A[i], A[ni])
        if j != nj:
            dj = (nj - j)
            if dj == 1 or dj == 1 - m:
                costB = B_next[j]
            elif dj == -1 or dj == m-1:
                costB = B_prev[j]
            else:
                costB = euclid(B[j], B[nj])
        return costA + costB

    # A* structures
    total_states = n * m
    g_score = np.full(total_states, np.inf, dtype=np.float32)
    parent = np.full(total_states, -1, dtype=np.int32)
    closed = np.zeros(total_states, dtype=bool)

    def sid(i, j):
        return i * m + j
    def ij_from_sid(s):
        return divmod(s, m)

    start_id = sid(si, sj)
    g_score[start_id] = 0.0
    h0 = h_func(si, sj)
    openq = []
    heapq.heappush(openq, (h0, start_id))

    # Precompute neighbor offsets (8-directional)
    neighbor_offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if not (di == 0 and dj == 0) and (abs(di)+abs(dj) in (1,2))]

    # Main loop
    while openq:
        f, s_cur = heapq.heappop(openq)
        if closed[s_cur]:
            continue
        i, j = ij_from_sid(s_cur)
        if is_goal(i, j):
            # Reconstruct path
            path_idx = []
            cid = s_cur
            while cid != -1:
                ii, jj = ij_from_sid(cid)
                path_idx.append((ii, jj))
                cid = parent[cid]
            path_idx.reverse()
            return [(A[ii], B[jj]) for ii, jj in path_idx], g_score[s_cur]
        closed[s_cur] = True

        # Local neighbors
        for di, dj in neighbor_offsets:
            ni = (i + di) % n
            nj = (j + dj) % m
            if use_numpy_state:
                if not valid[nj, ni]:
                    continue
            else:
                if not valid[nj][ni]:
                    continue
            nsid = sid(ni, nj)
            if closed[nsid]:
                continue
            c = local_edge_cost(i, j, ni, nj)
            tentative = g_score[s_cur] + c
            if tentative < g_score[nsid]:
                g_score[nsid] = tentative
                parent[nsid] = s_cur
                heapq.heappush(openq, (tentative + h_func(ni, nj), nsid))

    return None, math.inf

# --- Main execution ---
if __name__ == "__main__":

    start_time = time.time()

    # Map parameters
    map_file = "maps/mp_slim.png"

    # Visualization parameters
    show = True

    # Formation parameters
    BAR = 300
    TOL_percent = 20
    step_size = 10
    MAX_DIST = BAR + TOL_percent*BAR/100
    MIN_DIST = BAR - TOL_percent*BAR/100

    print("BAR:", BAR, "TOL %:", TOL_percent)
    print("Min dist:", MIN_DIST, "Max dist:", MAX_DIST)

    robot_r = 15
    r_side = 50
    print("Side-formation radius:", r_side)

    mappa = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
    map_rgb = cv2.cvtColor(mappa, cv2.COLOR_GRAY2BGR)
    
    obstacle = Obstacle(map_file, inflation_radius=r_side, step_size=step_size)
    # obstacle.show()
    image = obstacle.get_debug_image()
    image = np.zeros_like(image)
    cv2.fillPoly(image, [obstacle.get_samples()], color=(255, 255, 255))
    
    # Split obstacle into two point arrays
    array_a, array_b = obstacle.split_obstacle()
    array_a_inv = array_a[::-1]
    array_b_inv = array_b[::-1]

    length_a = len(array_a)
    length_b = len(array_b)

    # Find start/end indices

    id_a_start, id_b_start = find_start(array_a, array_b, BAR, BAR*TOL_percent/100)
    id_b_start_2, id_a_start_2 = find_start(array_b, array_a, BAR, BAR*TOL_percent/100)
    id_a_end_inv, id_b_end_inv = find_start(array_a_inv, array_b_inv, BAR, BAR*TOL_percent/100)
    id_b_end_2_inv, id_a_end_2_inv = find_start(array_b_inv, array_a_inv, BAR, BAR*TOL_percent/100)
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

    A = list(map(tuple, array_a))
    B = list(map(tuple, array_b))
    
    start = (id_a_start, id_b_start)
    # Goal specified as indices on both curves
    goal_indices = (id_a_end, id_b_end)
    
    print("Length A:", len(A), "Length B:", len(B))
    print("Start indices:", start, "Goal indices:", goal_indices)
    print("Euclid dist start:", euclid(A[start[0]], B[start[1]]))
    print("Euclid dist goal target:", euclid(A[goal_indices[0]], B[goal_indices[1]]))
    
    cv2.arrowedLine(image, a_start, b_start, color=(0, 0, 255), thickness=4)
    cv2.line(image, a_end, b_end, color=(0, 0, 255), thickness=4)

    print("Time to preprocess (s):", time.time() - start_time)

    # Run A* search - lean version: local neighbors only
    print("Starting A* (lean version, local neighbors only)...")
    
    start_time_astar = time.time()
    path, cost = astar_product(
        A, B, BAR, TOL_percent, start, goal_indices
    )
    print("Cost:", cost)

    if path:
        print("Path found successfully!")
        print("Elapsed time A* (s): {:.4f}".format(time.time() - start_time_astar))
        print("Elapsed time TOTAL (s): {:.4f}".format(time.time() - start_time))
        
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

        # Step-by-step visualization
        for pair in path:
            img_temp = image.copy()
            cv2.arrowedLine(img_temp, pair[0], pair[1], color=(0, 0, 255), thickness=10)
            cv2.circle(img_temp, pair[0], radius=15, color=(255, 255, 255), thickness=1)
            cv2.circle(img_temp, pair[1], radius=15, color=(255, 255, 255), thickness=1)
            distance = euclid(pair[0], pair[1])
            cv2.putText(img_temp, f"Distance: {distance:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imshow('VISUALIZE OBSTACLE', img_temp)
            cv2.waitKey(0)

    else:
        print("No path found")
