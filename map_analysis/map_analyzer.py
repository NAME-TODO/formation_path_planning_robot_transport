import cv2 
import numpy as np
import networkx as nx

from obstacle_analyzer import ObstacleAnalyzer
from robot_formation import generate_discrete_robot_sequence, generate_robot_formation_animation, animate_d_rob_transition, create_video_from_frames

class MapAnalyzer:
    def __init__(self,
                 map_path, 
                 skeleton_path):
        
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        self.skeleton = cv2.imread(skeleton_path, cv2.IMREAD_COLOR_BGR)

        skel_height, skel_width = self.skeleton.shape[:2]
        map_height, map_width = self.map.shape[:2]
        if skel_height != map_height or skel_width != map_width:
            raise ValueError("Skeleton and map dimensions do not match.") 
        self.height = skel_height
        self.width = skel_width

        self.color_map = None
        self.color_high = None
        self.color_medium = None
        self.color_low = None
        self.skeleton_graph = None
        self.map_graph = None

    def colorize(self, 
                color_high = (254,254,254), 
                color_medium = (139 ,139, 0), 
                color_low = (0, 69, 255),
                size_kernel_high = (135*2, 135*2),
                size_kernel_medium = (55*2, 55*2),
                size_kernel_low = (15*2, 15*2),
                show = False,
                save_color_map_path = None):
        
        self.color_high = color_high
        self.color_medium = color_medium
        self.color_low = color_low
        
        kernel_high = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size_kernel_high)
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size_kernel_medium)
        kernel_low = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size_kernel_low)

        map_inverted = cv2.bitwise_not(self.map)

        high_space = cv2.bitwise_not(cv2.dilate(map_inverted, kernel_high, iterations=1))
        medium_space = cv2.bitwise_not(cv2.dilate(map_inverted, kernel_medium, iterations=1))
        low_space = cv2.bitwise_not(cv2.dilate(map_inverted, kernel_low, iterations=1))

        self.color_map = cv2.cvtColor(self.map, cv2.COLOR_GRAY2BGR)

        self.color_map[low_space == 255] = color_low
        self.color_map[medium_space == 255] = color_medium
        self.color_map[high_space == 255] = color_high

        white_pixels = np.all(self.color_map == [255, 255, 255], axis=-1)
        self.color_map[white_pixels] = (100,100,100)

        if show or save_color_map_path:
            if show:
                cv2.imshow('color_map', self.color_map)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if save_color_map_path:
                cv2.imwrite(save_color_map_path, self.color_map)

    def build_skeleton_graph(self, 
                             centroid_radius=2,
                             show=False, 
                             save_debug_path=None):
        if self.color_map is None:
            raise ValueError("Color map not generated. Please run colorize() first.")
        if self.skeleton is None:
            raise ValueError("Skeleton image not loaded.")

        height, width = self.height, self.width
        skel_bgr = self.skeleton
        color_map = self.color_map

        debug = np.zeros_like(skel_bgr)
        debug_gray = cv2.cvtColor(debug, cv2.COLOR_BGR2GRAY)

        # Skeleton mask (solid red)
        skeleton = np.all(skel_bgr == [0, 0, 255], axis=-1)
        rows, cols = np.where(skeleton)
        debug[skeleton] = [255, 255, 255]

        # Endpoint and junction
        for r, c in zip(rows, cols):
            connections = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and skeleton[rr, cc]:
                        connections += 1
            if connections > 2 or connections == 1:
                debug_gray[r, c] = 255
                debug[r, c] = (255, 0, 255)

        # Connected components for centroids
        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(debug_gray, connectivity=8)

        skeleton_disconnected = np.all(debug == [255, 255, 255], axis=-1)
        rows, cols = np.where(skeleton_disconnected)

        def traverse_edge(r, c):
            path = [(r, c)]
            visited = set(path)
            while True:
                found_next = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < height and 0 <= cc < width:
                            if skeleton_disconnected[rr, cc] and (rr, cc) not in visited:
                                path.append((rr, cc))
                                visited.add((rr, cc))
                                r, c = rr, cc
                                found_next = True
                                break
                if not found_next:
                    break
            return path
        
        rough_paths = []
        for r, c in zip(rows, cols):
            connections = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and skeleton_disconnected[rr, cc]:
                        connections += 1
            if connections == 1 or connections == 0:
                path = traverse_edge(r, c)
                if not path:
                    continue
                rough_paths.append(path)
        
        # De-duplicate paths (same points, order irrelevant)
        unique_rough_paths = []
        seen = set()
        for path in rough_paths:
            key = frozenset(path)
            if key not in seen:
                unique_rough_paths.append(path)
                seen.add(key)

        def bridge_gap_1px(p0, p1):
            if p0 == p1:
                return [p0]
            if max(abs(p0[0]-p1[0]), abs(p0[1]-p1[1])) == 1:
                return [p0, p1]
            else:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = p0[0] + dr, p0[1] + dc
                        if max(abs(rr - p1[0]), abs(cc - p1[1])) == 1:
                            return [p0, (rr, cc), p1]
            return None

        paths = []
        for path in unique_rough_paths:

            start = path[0]
            end = path[-1]

            # Bridge start -> nearest cluster
            r, c = start[0], start[1]
            cluster_start = None
            stop = False
            for dr in [-1, 0, 1]:
                if stop: break
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and \
                          not skeleton_disconnected[rr, cc] and \
                          not label_ids[rr, cc] == 0:
                        cluster_start = label_ids[rr, cc]
                        stop = True
                        break
            if cluster_start is not None:
                cluster_start_point = (int(centroid[cluster_start][1]), int(centroid[cluster_start][0]))
                bridge = bridge_gap_1px(cluster_start_point, (rr, cc))
                if bridge is not None:
                    path = bridge + path
                else:
                    print(f"Warning: could not bridge start {start} to cluster {cluster_start}")
                #path.insert(0, (rr, cc))

            # Bridge end -> nearest cluster (different from start)
            r, c = end[0], end[1]
            cluster_end = None
            stop = False
            for dr in [-1, 0, 1]:
                if stop: break
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and \
                          not skeleton_disconnected[rr, cc] and \
                          not label_ids[rr, cc] == 0 and \
                          not label_ids[rr, cc] == cluster_start:
                        cluster_end = label_ids[rr, cc]
                        stop = True
                        break
            if cluster_end is not None:
                cluster_end_point = (int(centroid[cluster_end][1]), int(centroid[cluster_end][0]))
                bridge = bridge_gap_1px((rr, cc), cluster_end_point)
                if bridge is not None:
                    path = path + bridge
                else:
                    print(f"Warning: could not bridge end {end} to cluster {cluster_end}")
                #path.append((rr, cc))

            paths.append(path)

        # Build graph by segmenting by color
        G = nx.Graph()
        for path in paths:
            color = color_map[path[0]]
            start_id = 0
            for i in range(1, len(path)):
                if not (i == 0 or i == 1 or i == 2) and not (i == len(path)-1 or i == len(path)-2 or i == len(path)-3) and \
                   not np.array_equal(color, color_map[path[i]]) and \
                   not np.array_equal(color, color_map[path[np.min([i+1, len(path)-1])]]) and \
                   not np.array_equal(color, color_map[path[np.min([i+2, len(path)-1])]]):
                    G.add_edge(path[start_id], path[i], color=color, path=path[start_id:i])
                    start_id = i
                    color = color_map[path[i]]
            #if len(path[start_id:-1]) > 1:
            G.add_edge(path[start_id], path[-1], color=color, path=path[start_id:-1])

            """cluster_start = label_ids[path[0][0], path[0][1]]
            cluster_end = label_ids[path[-1][0], path[-1][1]]
            pt_start = (int(centroid[cluster_start][1]), int(centroid[cluster_start][0]))
            pt_end = (int(centroid[cluster_end][1]), int(centroid[cluster_end][0]))
            G.add_edge(pt_start, pt_end, color=[255,0,0], path=path[0:-1])"""
            #G.add_edge(path[0], path[-1], color=color_map[path[0]], path=path)

        if not nx.is_connected(G):
            print("The skeleton graph is not connected.")
            print(f"Connected components: {nx.number_connected_components(G)}")

        # Optional debug
        showcase = np.zeros_like(debug)
        if show or save_debug_path:
            for _, _, data in G.edges(data=True):
                for pt in data['path']:
                    showcase[pt] = data['color']
                if show:
                    cv2.imshow('skeleton_graph_debug', showcase)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            for u, v in G.edges():
                showcase[u] = [255, 0, 255]
                showcase[v] = [255, 0, 255]
            
            if save_debug_path:
                cv2.imwrite(save_debug_path, showcase)

        self.skeleton_graph = G

    def build_map_graph(self, white_threshold=250, w_ortho=1.0, w_diag=np.sqrt(2)):
        """
        Create a graph from 'white' pixels of self.color_map:
        - Node for each pixel with B,G,R >= white_threshold
        - 8-way edges: weight w_ortho for horizontal/vertical, w_diag for diagonal
        """
        if self.color_map is None:
            raise ValueError("Color map not available. Run colorize() first.")
        if self.skeleton_graph is None:
            raise ValueError("Skeleton graph not available. Run build_skeleton_graph() first.")

        img = self.color_map
        h, w = img.shape[:2]

        # Mask of "white" pixels (near white, e.g. 254,254,254)
        b, g, r = cv2.split(img)
        white_mask = (b >= white_threshold) & (g >= white_threshold) & (r >= white_threshold)

        ys, xs = np.where(white_mask)
        G = nx.Graph()

        # Add nodes
        for y, x in zip(ys, xs):
            G.add_node((y, x))

        # Connect 8-neighbors without duplicating edges (half-neighborhood)
        directions = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for dy, dx in directions:
            y2 = ys + dy
            x2 = xs + dx
            valid = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)

            y1v = ys[valid]
            x1v = xs[valid]
            y2v = y2[valid]
            x2v = x2[valid]

            nb_white = white_mask[y2v, x2v]
            if not np.any(nb_white):
                continue

            y1v = y1v[nb_white]
            x1v = x1v[nb_white]
            y2v = y2v[nb_white]
            x2v = x2v[nb_white]

            w_edge = w_diag if (abs(dy) + abs(dx) == 2) else w_ortho
            for a, b_ in zip(zip(y1v, x1v), zip(y2v, x2v)):
                G.add_edge(a, b_, weight=w_edge)

        # Add "bridge" edges from self.skeleton_graph segments
        for edge in self.skeleton_graph.edges(data=True):
            p0, p1, data = edge
            if np.all(data['color'] == self.color_medium):
                anchors = [None, None]
                id = 0
                for pt in [p0, p1]:
                    r, c = pt[0], pt[1]
                    for dr in [-1, 0, 1]: 
                        for dc in [-1, 0, 1]:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w:
                                if np.all(self.color_map[rr, cc] == self.color_high):
                                    anchors[id] = (rr, cc)
                    id += 1
                if anchors[0] is not None and anchors[1] is not None:
                    G.add_edge(anchors[0], anchors[1], weight=len(data['path']), path_info={})
                    print(f"Bridge added between {anchors[0]} and {anchors[1]} with weight {len(data['path'])}")
                            
        if not nx.is_connected(G):
            print("The map graph is not connected.")

        self.map_graph = G

    def find_and_show_path(self, start, end, path_color=(255, 0, 255), thickness=1,
                           ppf=10.0, fps=30, r_rob=15, d_rob=35,
                           show=True, save_path=None):
        """
        Find a path between start and end in self.map_graph and draw it on a copy of self.color_map.
        - start, end: tuple (r, c) that MUST be graph nodes
        - path_color: BGR color of the path
        - thickness: line thickness in pixels
        - show: show a window with the result
        - save_path: if not None, save the resulting image
        Returns: (path, img_vis, path_info_list) where:
        - path is the list of (r,c) nodes
        - img_vis is the BGR image with the drawn path
        - path_info_list is a list of dict with path_info for each edge
        """
        if self.map_graph is None:
            raise ValueError("map_graph not available. Call build_map_graph() first.")
        if self.color_map is None:
            raise ValueError("color_map not available. Call colorize() first.")
        if start not in self.map_graph:
            raise ValueError(f"The start node {start} is not present in map_graph.")
        if end not in self.map_graph:
            raise ValueError(f"The end node {end} is not present in map_graph.")

        try:
            path = nx.shortest_path(self.map_graph, source=start, target=end, weight="weight")
            
            # Extract path_info for each edge of the path
            path_info_list = []
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.map_graph[node1][node2]
                path_info = edge_data.get('path_info', {})
                path_info_list.append({
                    'from': node1,
                    'to': node2,
                    'path_info': path_info,
                    'weight': edge_data.get('weight', 1),
                    'edge_data': edge_data  # All edge data if needed
                })

            full_points_A = []
            full_points_B = []

            full_frames_start = []
            dtr_frames = []
            rot_frames = []
            ray_frames = []
            path_frames = []
            ray_frames_end = []
            rot_frames_end = []
            dtr_frames_end = []
            full_frames_end = []

            map = cv2.cvtColor(self.map.copy(), cv2.COLOR_GRAY2BGR)
            
            for info in path_info_list:
                if len(info['path_info']) > 0:

                    full_frames_start = generate_discrete_robot_sequence(
                            image=map,
                            point_A_list=full_points_A,
                            point_B_list=full_points_B,
                            d_rob=0,
                            r_rob=r_rob,
                            pixels_per_frame=ppf,
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False
                        )
                    
                    full_points_A = []
                    full_points_B = []

                    start_full = info['path_info'].get('start_full', None)
                    start_side = info['path_info'].get('start_side', None)
                    start_side_connection = info['path_info'].get('start_side_connection', None)
                    
                    dtr_frames = animate_d_rob_transition(
                            image=map,
                            point_A=(start_full[0] + 120, start_full[1]),
                            point_B=(start_full[0] - 120, start_full[1]),
                            d_rob_start=0,
                            d_rob_end=d_rob,
                            r_rob=r_rob,
                            num_frames=fps,
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False
                        )
                    
                    rot_frames = generate_robot_formation_animation(
                            image=map,
                            point_A_start=(start_full[0] + 120, start_full[1]),
                            point_B_start=(start_full[0] - 120, start_full[1]),
                            point_A_end=start_side[0],
                            point_B_end=start_side[1],
                            d_rob=d_rob,
                            r_rob=r_rob,
                            pixels_per_frame=ppf,
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False)
                    
                    ray_frames = generate_robot_formation_animation(
                            image=map,
                            point_A_start=start_side[0],
                            point_B_start=start_side[1],
                            point_A_end=start_side_connection[0],
                            point_B_end=start_side_connection[1],
                            d_rob=d_rob,
                            r_rob=r_rob,
                            pixels_per_frame=ppf,
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False)
                    
                    path_points_A = []
                    path_points_B = []
                    path_between = info['path_info'].get('path_between', None)

                    for i, (pt_a, pt_b) in enumerate(path_between[:-1]):
                        path_points_A.append(pt_a)
                        path_points_B.append(pt_b)

                    path_frames = generate_discrete_robot_sequence(
                            image=map,
                            point_A_list=path_points_A,
                            point_B_list=path_points_B,
                            d_rob=d_rob,
                            r_rob=r_rob,
                            pixels_per_frame=ppf,
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False
                        )
                    
                    end_side_connection = info['path_info'].get('end_side_connection', None)
                    end_side = info['path_info'].get('end_side', None)
                    end_full = info['path_info'].get('end_full', None)

                    ray_frames_end = generate_robot_formation_animation(
                            image=map,
                            point_A_start=end_side_connection[0],
                            point_B_start=end_side_connection[1],
                            point_A_end=end_side[0],
                            point_B_end=end_side[1],
                            d_rob=d_rob,
                            r_rob=r_rob,
                            output_dir="animation_test",
                            filename_prefix="test_frame",
                            pixels_per_frame=ppf,
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False)
                    
                    rot_frames_end = generate_robot_formation_animation(
                            image=map,
                            point_A_start=end_side[0],
                            point_B_start=end_side[1],
                            point_A_end=(end_full[0] + 120, end_full[1]),
                            point_B_end=(end_full[0] - 120, end_full[1]),
                            d_rob=d_rob,
                            r_rob=r_rob,
                            output_dir="animation_test",
                            filename_prefix="test_frame",
                            pixels_per_frame=ppf,
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False)
                    
                    dtr_frames_end = animate_d_rob_transition(
                            image=map,
                            point_A=(end_full[0] + 120, end_full[1]),
                            point_B=(end_full[0] - 120, end_full[1]),
                            d_rob_start=d_rob,
                            d_rob_end=0,
                            r_rob=r_rob,
                            num_frames=fps,
                            output_dir="d_rob_transition_test",
                            filename_prefix="d_rob_frame",
                            circumference_color=(255, 0, 0),  # Blue
                            robot_color=(0, 255, 0),          # Verde
                            thickness=2,
                            save_images=False
                        )

                else:
                    point_A = (info['from'][1] + 120, info['from'][0])
                    point_B = (info['from'][1] - 120, info['from'][0])
                    
                    full_points_A.append(point_A)
                    full_points_B.append(point_B)

            full_frames_end = generate_discrete_robot_sequence(
                    image=map,
                    point_A_list=full_points_A,
                    point_B_list=full_points_B,
                    d_rob=0,
                    r_rob=r_rob,
                    pixels_per_frame=ppf,
                    circumference_color=(255, 0, 0),  # Blue
                    robot_color=(0, 255, 0),          # Verde
                    thickness=2,
                    save_images=False
                )

            """for frame in full_frames_start + dtr_frames + rot_frames + ray_frames + path_frames + ray_frames_end + rot_frames_end + dtr_frames_end + full_frames_end:
                    cv2.imshow("Robot Formation Animation", frame)
                    cv2.waitKey(0)"""

            create_video_from_frames(
                frames=full_frames_start + dtr_frames + rot_frames + ray_frames + path_frames + ray_frames_end + rot_frames_end + dtr_frames_end + full_frames_end,
                output_video_path="robot_formation_path.mp4",
                fps=15.0,
                loop_animation=False
            )

            cv2.destroyAllWindows()

        except nx.NetworkXNoPath:
            print(f"No path found between {start} and {end}.")
            return None, self.color_map.copy(), []

        img_vis = self.color_map.copy()

        # Draw the path as polylines between consecutive pixels
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
            cv2.line(img_vis, (c0, r0), (c1, r1), path_color, thickness)

        if show:
            cv2.imshow("map_path", img_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path:
            cv2.imwrite(save_path, img_vis)

        return path, img_vis, path_info_list
    
    def extract_path_info(self, path):
        """
        Extract path_info for each edge of a given path.
        
        Args:
            path: list of nodes (tuples) that form the path
            
        Returns:
            list of dict with path_info for each edge
        """
        if self.map_graph is None:
            raise ValueError("map_graph not available. Call build_map_graph() first.")
        
        path_info_list = []
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            edge_data = self.map_graph[node1][node2]
            path_info_list.append({
                'from': node1,
                'to': node2,
                'path_info': edge_data.get('path_info', {}),
                'weight': edge_data.get('weight', 1),
                'edge_data': edge_data
            })
        return path_info_list
    
    def add_crossings(self, oa: ObstacleAnalyzer, use_dummy=False):

        oa.add_crossings(self.map_graph, use_dummy=use_dummy)