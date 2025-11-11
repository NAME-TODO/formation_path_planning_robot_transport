import cv2
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, Point
from skimage.morphology import skeletonize
from graph_path_planner import find_dual_robot_path

class ObstacleAnalyzer:
    def __init__(self,
                 map_path,
                 color_map_path,
                 bar,
                 tol,
                 min_obstacle_area=10,
                 inflation_radius=5,
                 sampling_step=5,
                 r_rob=15,
                 d_rob=35):
        
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        self.color_map = cv2.imread(color_map_path, cv2.IMREAD_COLOR)
        self.height, self.width = self.map.shape
        self.min_obstacle_area = min_obstacle_area
        self.inflation_radius = inflation_radius
        self.sampling_step = sampling_step
        self.bar = bar
        self.tol = tol
        self.r_rob = r_rob
        self.d_rob = d_rob

        self.obstacles, self.inflated_obstacles = self.find_obstacles()
        self.sampled_obstacles = self.sample_obstacle_boundaries()
        self.main_paths = self.compute_skeletons()
        self.splits = self.split_obstacles()
        self.connections = self.compute_connections()


    def find_obstacles(self):
        
        _, thresh = cv2.threshold(self.map, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) <= 1:
            raise ValueError("No obstacles found in the map.")
        
        print(f"Found {len(contours)} contours.")

        contours = contours[1:]  # Ignore the outer contour

        obstacles = []
        inflated_obstacles = []

        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_obstacle_area:  # Filter small areas
                poly = Polygon(cnt.squeeze())
                inflated_poly = poly.buffer(self.inflation_radius)
                obstacles.append(poly)
                inflated_obstacles.append(inflated_poly)
            
        if len(obstacles) != len(inflated_obstacles):
            raise ValueError("Mismatch in number of obstacles and inflated obstacles.")

        return obstacles, inflated_obstacles
    
    def sample_obstacle_boundaries(self):
        sampled_obstacles = []

        for poly in self.inflated_obstacles:
            boundary = poly.exterior
            perimeter_length = boundary.length

            num_samples = int(perimeter_length / self.sampling_step)
            sampled_points = [boundary.interpolate(i * self.sampling_step) for i in range(num_samples)]
            sampled_points = np.array([(int(p.x), int(p.y)) for p in sampled_points], np.int32)

            sampled_obstacles.append(sampled_points)

        if len(sampled_obstacles) != len(self.inflated_obstacles):
            raise ValueError("Mismatch in number of sampled obstacles and inflated obstacles.")

        return sampled_obstacles
    
    def compute_skeletons(self):

        def find_longest_path(skeleton):
            """
            Given a binary 2D numpy array (skeleton), returns the list of (row, col) coordinates
            representing the longest path (diameter) of the skeleton, using 8-connectivity.

            Args:
                skeleton (np.ndarray): 2D binary array (dtype: bool, 0/1, or 0/255)
            Returns:
                List[Tuple[int, int]]: List of (row, col) coordinates for the longest path
            """
            # Ensure skeleton is boolean
            skeleton = skeleton.astype(bool)
            G = nx.Graph()
            rows, cols = np.where(skeleton)
            for r, c in zip(rows, cols):
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < skeleton.shape[0] and 0 <= cc < skeleton.shape[1]:
                            if skeleton[rr, cc]:
                                G.add_edge((r, c), (rr, cc))

            if len(G.nodes) == 0:
                return []

            # Step 1: pick arbitrary node
            node = next(iter(G.nodes))
            # Step 2: find farthest from node
            lengths = nx.single_source_shortest_path_length(G, node)
            farthest = max(lengths, key=lengths.get)
            # Step 3: find farthest from farthest
            lengths2 = nx.single_source_shortest_path_length(G, farthest)
            farthest2 = max(lengths2, key=lengths2.get)
            # Step 4: get path
            path = nx.shortest_path(G, farthest, farthest2)
            path = [(x, y) for (y, x) in path]
            
            return path
        
        main_paths = []

        for poly in self.inflated_obstacles:
            canvas = np.zeros((self.height, self.width), dtype=np.uint8)
            coords = np.array(poly.exterior.coords, np.int32)
            cv2.fillPoly(canvas, [coords], 255)

            binary_canvas = (canvas > 0).astype(np.uint8)
            skeleton = skeletonize(binary_canvas).astype(np.uint8)

            main_paths.append(find_longest_path(skeleton))

        if len(main_paths) != len(self.inflated_obstacles):
            raise ValueError("Mismatch in number of main paths and inflated obstacles.")

        return main_paths
    
    
    def split_obstacles(self):

        def find_split_point(contour, point, radius):
            circle = Point(point).buffer(radius)

            buf1 = []
            buf2 = []
            old_i = 0
            buf = buf1
            flag = False

            for i in range(len(contour)):
                if circle.contains(Point(contour[i])):
                    if i > 0 and i > old_i + 1 and flag==False:
                        buf = buf2
                        flag = True

                    buf.append(contour[i])

                    old_i = i

            buf2.extend(buf1)

            return buf2[int(len(buf2)/2)]

        splits = []

        for i in range(len(self.sampled_obstacles)):
            contour = self.sampled_obstacles[i]
            main_path = self.main_paths[i]
            radius = int(self.inflation_radius +  100)

            center_start = main_path[0]
            center_end = main_path[-1]

            start_point = find_split_point(contour, center_start, radius)
            end_point = find_split_point(contour, center_end, radius)

            temp = [tuple(row) for row in contour]
        
            start_id = temp.index(tuple(start_point))
            end_id = temp.index(tuple(end_point))

            length = len(contour)
            array = np.zeros_like(contour)
            
            offset = np.abs(end_id-start_id)

            array[0:length-start_id] = contour[start_id:]
            array[length-start_id:] = contour[0:start_id]

            array_rx = array[0:offset]
            array_lx = np.flip(array[offset:], axis=0)  

            splits.append([list(array_rx), list(array_lx)])

        if len(splits) != len(self.sampled_obstacles):
            raise ValueError("Mismatch in number of splits and sampled obstacles.")

        return splits
    
    def compute_connections(self):

        def find_start(array_a, array_b, BAR, TOL):
            id_a = 0
            id_b = None
            for i in range(len(array_b)):
                distance = np.linalg.norm(np.asanyarray(array_a[id_a]) - np.asanyarray(array_b[i]))
                if np.abs(distance - BAR) < TOL:
                    id_b = i
                    break
            
            return id_b
        
        def _scan_parallel_rays_local(
            p0,
            p1,
            angle_step=np.deg2rad(10),
            max_steps=100,
            step_size=2.0,
            bounds_check=True,
            debug_image=self.color_map
        ):
            """
            Local variant: samples angles in [0, pi) and, for each angle, moves two points
            along parallel rays (same direction) until `test_fn(p0_t, p1_t, mid)` is True
            or exits map / reaches max_steps. Collects ALL successes.

            Returns a list of triplets [(p0_i, p1_i, mid_i), ...] for each angle that has
            found a successful step.
            """
            def test_fn(p0_i, p1_i, mid_i):
                go_on = False
                found = False
                """cv2.imshow("Debug Connections", self.color_map)
                cv2.waitKey(1)"""
                
                if (np.all(self.color_map[p0_i[1], p0_i[0]] == [254, 254, 254]) or np.all(self.color_map[p0_i[1], p0_i[0]] == [139, 139, 0])) and \
                   (np.all(self.color_map[p1_i[1], p1_i[0]] == [254, 254, 254]) or np.all(self.color_map[p1_i[1], p1_i[0]] == [139, 139, 0])):
                    go_on = True
                if np.all(self.color_map[mid_i[1], mid_i[0]] == [254, 254, 254]):
                    found = True

                #print(self.color_map[p0_i[1], p0_i[0]])

                return go_on, found

            p0_bk, p1_bk = p0, p1
            p0 = np.asarray(p0, dtype=float)
            p1 = np.asarray(p1, dtype=float)

            results = []
            angles = np.arange(0.0, 2*np.pi, angle_step)

            def in_bounds(pt):
                if not bounds_check:
                    return True
                x, y = int(round(pt[0])), int(round(pt[1]))
                return 0 <= x < self.width and 0 <= y < self.height

            for theta in angles:
                dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                found = False
                for step in range(1, max_steps + 1):
                    delta = step * step_size * dir_vec
                    p0_t = p0 + delta
                    p1_t = p1 + delta

                    if bounds_check and (not in_bounds(p0_t) or not in_bounds(p1_t)):
                        # questo angolo esce dai limiti: passa al prossimo angolo
                        break

                    mid = (p0_t + p1_t) / 2.0
                    p0_i = (int(round(p0_t[0])), int(round(p0_t[1])))
                    p1_i = (int(round(p1_t[0])), int(round(p1_t[1])))
                    mid_i = (int(round(mid[0])), int(round(mid[1])))

                    go_on, found = test_fn(p0_i, p1_i, mid_i)
                    #print(go_on)
                    if found and go_on:
                        results.append((p0_i, p1_i, mid_i))
                        """img = debug_image.copy()
                        cv2.line(img, tuple(p0_bk), tuple(p0_i), (255, 0, 0), 3)
                        cv2.line(img, tuple(p1_bk), tuple(p1_i), (255, 0, 0), 3)
                        cv2.circle(img, tuple(mid_i), 6, (255, 0, 0), -1)
                        cv2.imshow("Debug Connections", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()"""


                        break  # move to next angle after first success for this pair of rays
                    if not go_on:
                        break # if nothing found for this angle, simply continue with the next

            return results
        
        connections = []

        for split_pair in self.splits:
            connections.append({
                "starts": [],
                "start_connections": [],
                "ends": [],
                "end_connections": []
            })
            array_a = split_pair[0]
            array_b = split_pair[1]
            """for pt in array_a:
                cv2.circle(self.color_map, tuple(pt), 5, (0, 255, 0), -1)
            for pt in array_b:
                cv2.circle(self.color_map, tuple(pt), 5, (0, 0, 255), -1)
            cv2.imshow("Debug Connections", self.color_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
            id_b = find_start(array_a, array_b, self.bar, self.tol)
            #print("id_b:", id_b)
            if id_b is not None:
                connections[-1]["starts"].append((array_a[0], array_b[id_b]))
                connections[-1]["start_connections"].append(_scan_parallel_rays_local(array_a[0], array_b[id_b]))
            id_a = find_start(array_b, array_a, self.bar, self.tol)
            #print("id_a:", id_a)
            if id_a is not None:
                connections[-1]["starts"].append((array_a[id_a], array_b[0]))
                connections[-1]["start_connections"].append(_scan_parallel_rays_local(array_a[id_a], array_b[0]))
            array_a_inv = array_a[::-1]
            array_b_inv = array_b[::-1]
            id_b_inv = find_start(array_a_inv, array_b_inv, self.bar, self.tol)
            #print("id_b_inv:", id_b_inv)
            if id_b_inv is not None:
                connections[-1]["ends"].append((array_a_inv[0], array_b_inv[id_b_inv]))
                connections[-1]["end_connections"].append(_scan_parallel_rays_local(array_a_inv[0], array_b_inv[id_b_inv]))
            id_a_inv = find_start(array_b_inv, array_a_inv, self.bar, self.tol)
            #print("id_a_inv:", id_a_inv)
            if id_a_inv is not None:
                connections[-1]["ends"].append((array_a_inv[id_a_inv], array_b_inv[0]))
                connections[-1]["end_connections"].append(_scan_parallel_rays_local(array_a_inv[id_a_inv], array_b_inv[0]))

        return connections

    def add_crossings(self, G, use_dummy=False):

        def dummy_crossings():
            return True, 100
        
        print(len(self.connections), "obstacles with connections")

        stop = False

        for i in range(len(self.connections)):
            if stop:
                break
            for sid, start in enumerate(self.connections[i]["starts"]):
                if stop:
                    break
                for eid, end in enumerate(self.connections[i]["ends"]):
                    if stop:
                        break
                    start_connections = self.connections[i]["start_connections"][sid]
                    end_connections = self.connections[i]["end_connections"][eid]
                    #print(start_connections, end_connections)
                    if start_connections and end_connections:
                        if use_dummy:
                            success, weight = dummy_crossings()
                            path = []
                        else:
                            array_a, array_b =self.splits[i]
                            path = find_dual_robot_path(
                                array_a,
                                array_b,
                                start_a=tuple(start[0]),
                                start_b=tuple(start[1]),
                                goal_a=tuple(end[0]),
                                goal_b=tuple(end[1]),
                                collision_map=self.map,
                                robot_radius=self.r_rob,
                                intra_robot_distance=self.d_rob,
                                formation_distance=self.bar,
                                formation_tolerance=self.tol/100
                            )
                            success = False
                            weight = None
                            if path:
                                success = True
                                weight = len(path)
                                print(f"Path planning successful! Found path with {len(path)} waypoints.")
                        
                        if success:
                            #stop = True
                            print("Adding crossing between {} and {}".format(start, end))
                            for sc in start_connections:
                                for ec in end_connections:
                                    mid_start = sc[2]
                                    mid_end = ec[2]
                                    d_start_ray = np.linalg.norm(np.asanyarray(start[0]) - np.asanyarray(sc[0]))
                                    d_end_ray = np.linalg.norm(np.asanyarray(end[0]) - np.asanyarray(ec[0]))
                                    G.add_edge((mid_start[1], mid_start[0]), (mid_end[1], mid_end[0]), weight=weight + d_start_ray + d_end_ray,
                                               path_info={
                                                   "start_full": mid_start,
                                                   "start_side": (sc[0], sc[1]),
                                                   "start_side_connection": start,
                                                   "path_between": path,
                                                   "end_side_connection": end,
                                                   "end_side": (ec[0], ec[1]),
                                                   "end_full": mid_end
                                               })
                                    
                                    """img = self.color_map.copy()
                                    cv2.line(img, tuple(start[0]), tuple(sc[0]), (255, 0, 0), 2)
                                    cv2.line(img, tuple(start[1]), tuple(sc[1]), (255, 0, 0), 2)
                                    cv2.circle(img, tuple(mid_start), 4, (255, 0, 255), -1)

                                    cv2.line(img, tuple(end[0]), tuple(ec[0]), (0, 255, 0), 4)
                                    cv2.line(img, tuple(end[1]), tuple(ec[1]), (0, 0, 255), 4)
                                    cv2.circle(img, tuple(mid_end), 4, (255, 0, 255), -1)
                                    cv2.imshow("Debug Crossings", img)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()"""
                                

    def visualize_sampled_obstacles(self, save_path=None):
        img_vis = cv2.cvtColor(self.map, cv2.COLOR_GRAY2BGR)

        for i in range(len(self.sampled_obstacles)):
            img_vis_temp = img_vis.copy()
            """for point in self.sampled_obstacles[i]:
                cv2.circle(img_vis_temp, tuple(point), 1, (0, 0, 255), -1)"""
            for point in self.main_paths[i]:
                cv2.circle(img_vis_temp, tuple(point), 1, (0, 255, 0), -1)
            for point in self.splits[i][0]:
                cv2.circle(img_vis_temp, tuple(point), 1, (255, 0, 0), -1)
            for point in self.splits[i][1]:
                cv2.circle(img_vis_temp, tuple(point), 3, (0, 0, 255), -1)

            cv2.imshow("Sampled Obstacles", img_vis_temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path:
            cv2.imwrite(save_path, img_vis)