from map_analyzer import MapAnalyzer
from obstacle_analyzer import ObstacleAnalyzer
import time

start_total = time.perf_counter()

t = time.perf_counter()
analyzer = MapAnalyzer('maps/map_final.png', 'skeletons/skeleton_final.png')
print(f"[1] MapAnalyzer(...)    : {(time.perf_counter()-t)*1:.2f} s")

t = time.perf_counter()
analyzer.colorize(show=False, save_color_map_path='color_map.png',
                  size_kernel_high=(135*2, 135*2), size_kernel_medium=(55*2, 55*2), size_kernel_low=(15*2, 15*2))
print(f"[2] colorize()           : {(time.perf_counter()-t)*1:.2f} s")

t = time.perf_counter()
analyzer.build_skeleton_graph(show=False, centroid_radius=3, save_debug_path='debug.png')
print(f"[3] build_skeleton_graph(): {(time.perf_counter()-t)*1:.2f} s")

t = time.perf_counter()
analyzer.build_map_graph()
print(f"[4] build_map_graph()    : {(time.perf_counter()-t)*1:.2f} s")

t = time.perf_counter()
oa = ObstacleAnalyzer('maps/map_final.png', 'color_map.png', 
                      bar=300, tol=20, min_obstacle_area=10, inflation_radius=55, sampling_step=10)
analyzer.add_crossings(oa, use_dummy=False)
print(f"[4.1] add_crossings()      : {(time.perf_counter()-t)*1:.2f} s")

t = time.perf_counter()
analyzer.find_and_show_path(start=(2600,540), end=(320,990), thickness=5, show=False, save_path='path.png')
print(f"[5] find_and_show_path() : {(time.perf_counter()-t)*1:.2f} s")

print(f"Total time               : {(time.perf_counter()-start_total):.3f} s")
