import cv2
import numpy as np
import math
from typing import Tuple, Optional

def draw_robot_formation(image: np.ndarray, 
                        point_A: Tuple[int, int], 
                        point_B: Tuple[int, int], 
                        d_rob: float, 
                        r_rob: int,
                        circumference_color: Tuple[int, int, int] = (0, 255, 0),
                        robot_color: Tuple[int, int, int] = (255, 0, 0),
                        thickness: int = 2,
                        distance_range: Tuple[int, int] = (240, 360),
                        text_scale: float = 2.4) -> np.ndarray:
    """
    Draw a formation of 6 robots on an image.
    
    Args:
        image: Image to draw on
        point_A: First point of the diameter (x, y)
        point_B: Second point of the diameter (x, y)
        d_rob: Euclidean distance between main and additional robots
        r_rob: Radius of robot circles
        circumference_color: Color of the circumference (B, G, R)
        robot_color: Color of robot circles (B, G, R)
        thickness: Line thickness
    
    Returns:
        Modified image with robot formation
    """
    # Copy the image to avoid modifying the original
    result_img = image.copy()
    
    # Calculate the center and radius of the circumference
    center_x = (point_A[0] + point_B[0]) // 2
    center_y = (point_A[1] + point_B[1]) // 2
    center = (center_x, center_y)
    
    # Calculate the radius (half of the AB distance)
    radius = int(math.sqrt((point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2) / 2)
    
    # Draw the circumference
    cv2.circle(result_img, center, radius, circumference_color, thickness)
    
    # Calculate the angle of diameter AB with respect to the x-axis
    angle_AB = math.atan2(point_B[1] - point_A[1], point_B[0] - point_A[0])
    
    # Positions of main robots (robA and robB at diameter endpoints)
    robA_pos = point_A
    robB_pos = point_B
    
    if d_rob == 0:
        # Special case: 6 equidistant robots along the circumference
        positions = []
        for i in range(6):
            angle = i * (2 * math.pi / 6)  # Equidistant angles
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            positions.append((x, y))
        
        # Draw all 6 equidistant robots
        for i, pos in enumerate(positions):
            cv2.circle(result_img, pos, r_rob, robot_color, -1)
    
    else:
        # Normal case: position robots based on d_rob
        
        # Draw robA and robB
        cv2.circle(result_img, robA_pos, r_rob, robot_color, -1)
        cv2.circle(result_img, robB_pos, r_rob, robot_color, -1)
        
        # Calculate positions of additional robots
        additional_robots = []
        
        # For robA: find points on circumference at distance d_rob
        robA_additional = _find_additional_robots(center, radius, robA_pos, d_rob)
        additional_robots.extend(robA_additional)
        
        # For robB: find points on circumference at distance d_rob
        robB_additional = _find_additional_robots(center, radius, robB_pos, d_rob)
        additional_robots.extend(robB_additional)
        
        # Draw additional robots
        for i, pos in enumerate(additional_robots):
            cv2.circle(result_img, pos, r_rob, robot_color, -1)
    
    # Draw black box with distance information
    # Scale dimensions based on text size (more generously)
    box_width = int(250 * text_scale)  # Wider to contain text
    box_height = int(60 * text_scale)  # Taller to contain text
    cv2.rectangle(result_img, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    
    # Determine text color based on range
    distance = radius * 2  # Diameter of the circumference
    if distance_range[0] <= distance <= distance_range[1]:
        text_color = (0, 255, 0)  # Green
    else:
        text_color = (0, 0, 255)  # Red
    
    # Draw distance text
    text = f"DISTANCE: {distance:.0f}"
    text_thickness = max(1, int(2 * text_scale))
    text_x_offset = int(20 * text_scale)  # Proportional left margin
    text_y_offset = int(45 * text_scale)  # Centered vertical position
    cv2.putText(result_img, text, (text_x_offset, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
    
    return result_img


def _find_additional_robots(center: Tuple[int, int], 
                          radius: int, 
                          main_robot_pos: Tuple[int, int], 
                          d_rob: float) -> list[Tuple[int, int]]:
    """
    Find positions of two additional robots on the circumference
    at distance d_rob from the main robot.
    
    Args:
        center: Center of the circumference
        radius: Radius of the circumference
        main_robot_pos: Position of the main robot
        d_rob: Desired distance
    
    Returns:
        List with positions of the two additional robots
    """
    cx, cy = center
    mx, my = main_robot_pos
    
    # Calculate the angle of the main robot with respect to center
    main_angle = math.atan2(my - cy, mx - cx)
    
    # Calculate angle corresponding to distance d_rob on the circumference
    # Using formula: d = 2 * R * sin(θ/2), so θ = 2 * arcsin(d/(2*R))
    if d_rob > 2 * radius:
        # If d_rob is too large, position robots at antipodes
        delta_angle = math.pi / 2
    else:
        # Calculate exact angle
        delta_angle = 2 * math.asin(d_rob / (2 * radius))
    
    # Positions of the two additional robots
    angle1 = main_angle + delta_angle
    angle2 = main_angle - delta_angle
    
    pos1 = (
        int(cx + radius * math.cos(angle1)),
        int(cy + radius * math.sin(angle1))
    )
    pos2 = (
        int(cx + radius * math.cos(angle2)),
        int(cy + radius * math.sin(angle2))
    )
    
    return [pos1, pos2]


def create_test_image(width: int = 800, height: int = 600) -> np.ndarray:
    """
    Create a test image with white background.
    
    Args:
        width: Image width
        height: Image height
    
    Returns:
        Empty image
    """
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def calculate_animation_frames(point_A_start: Tuple[int, int], 
                             point_B_start: Tuple[int, int],
                             point_A_end: Tuple[int, int], 
                             point_B_end: Tuple[int, int],
                             min_frames: int = 5,
                             max_frames: int = 50,
                             pixels_per_frame: float = 10.0) -> int:
    """
    Calculate the number of frames needed for animation based on distance
    between initial and final configurations.
    
    Args:
        point_A_start: Initial point A
        point_B_start: Initial point B  
        point_A_end: Final point A
        point_B_end: Final point B
        min_frames: Minimum number of frames
        max_frames: Maximum number of frames
        pixels_per_frame: Pixels of movement per frame (controls speed)
    
    Returns:
        Calculated number of frames
    """
    # Calculate movement distance for both points
    dist_A = math.sqrt((point_A_end[0] - point_A_start[0])**2 + 
                       (point_A_end[1] - point_A_start[1])**2)
    dist_B = math.sqrt((point_B_end[0] - point_B_start[0])**2 + 
                       (point_B_end[1] - point_B_start[1])**2)
    
    # Use the greater distance to determine frames
    max_distance = max(dist_A, dist_B)
    
    # Calculate needed frames
    calculated_frames = int(max_distance / pixels_per_frame)
    
    # Apply min/max limits
    return max(min_frames, min(max_frames, calculated_frames))


def interpolate_points(start: Tuple[int, int], 
                      end: Tuple[int, int], 
                      t: float) -> Tuple[int, int]:
    """
    Linearly interpolate between two points.
    
    Args:
        start: Initial point
        end: Final point
        t: Interpolation parameter (0.0 = start, 1.0 = end)
    
    Returns:
        Interpolated point
    """
    x = int(start[0] + t * (end[0] - start[0]))
    y = int(start[1] + t * (end[1] - start[1]))
    return (x, y)


def interpolate_circle_preserving_radius(point_A_start: Tuple[int, int],
                                        point_B_start: Tuple[int, int],
                                        point_A_end: Tuple[int, int],
                                        point_B_end: Tuple[int, int],
                                        t: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Interpolate between two circumference configurations minimizing radius changes.
    When circumferences have the same center, use pure rotation.
    
    Args:
        point_A_start: Initial point A
        point_B_start: Initial point B
        point_A_end: Final point A
        point_B_end: Final point B
        t: Interpolation parameter (0.0 = start, 1.0 = end)
    
    Returns:
        Tuple containing interpolated points A and B
    """
    # Calculate circumference centers
    center_start = (
        (point_A_start[0] + point_B_start[0]) / 2,
        (point_A_start[1] + point_B_start[1]) / 2
    )
    center_end = (
        (point_A_end[0] + point_B_end[0]) / 2,
        (point_A_end[1] + point_B_end[1]) / 2
    )
    
    # Calculate circumference radii
    radius_start = math.sqrt((point_B_start[0] - point_A_start[0])**2 + 
                            (point_B_start[1] - point_A_start[1])**2) / 2
    radius_end = math.sqrt((point_B_end[0] - point_A_end[0])**2 + 
                          (point_B_end[1] - point_A_end[1])**2) / 2
    
    # Threshold to consider centers "equal" (in pixels)
    center_tolerance = 5.0
    center_distance = math.sqrt((center_end[0] - center_start[0])**2 + 
                               (center_end[1] - center_start[1])**2)
    
    if center_distance <= center_tolerance:
        # Case 1: Equal centers -> Pure rotation with constant radius
        # Use average radius to keep circumference as stable as possible
        avg_radius = (radius_start + radius_end) / 2
        
        # Calculate initial and final angles
        angle_A_start = math.atan2(point_A_start[1] - center_start[1], 
                                  point_A_start[0] - center_start[0])
        angle_A_end = math.atan2(point_A_end[1] - center_end[1], 
                                point_A_end[0] - center_end[0])
        
        # Interpolate angle (handling wraparound at 2π)
        angle_diff = angle_A_end - angle_A_start
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        current_angle_A = angle_A_start + t * angle_diff
        current_angle_B = current_angle_A + math.pi  # B is always opposite to A
        
        # Calculate positions with constant radius
        current_center = center_start  # Fixed center
        
        current_A = (
            int(current_center[0] + avg_radius * math.cos(current_angle_A)),
            int(current_center[1] + avg_radius * math.sin(current_angle_A))
        )
        current_B = (
            int(current_center[0] + avg_radius * math.cos(current_angle_B)),
            int(current_center[1] + avg_radius * math.sin(current_angle_B))
        )
        
    else:
        # Case 2: Different centers -> Combination of translation and rotation/scaling
        
        # Interpolate center
        current_center = (
            center_start[0] + t * (center_end[0] - center_start[0]),
            center_start[1] + t * (center_end[1] - center_start[1])
        )
        
        # Interpolate radius gradually
        current_radius = radius_start + t * (radius_end - radius_start)
        
        # Calculate angles and interpolate them as well
        angle_A_start = math.atan2(point_A_start[1] - center_start[1], 
                                  point_A_start[0] - center_start[0])
        angle_A_end = math.atan2(point_A_end[1] - center_end[1], 
                                point_A_end[0] - center_end[0])
        
        # Interpolate angle
        angle_diff = angle_A_end - angle_A_start
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        current_angle_A = angle_A_start + t * angle_diff
        current_angle_B = current_angle_A + math.pi
        
        # Calculate final positions
        current_A = (
            int(current_center[0] + current_radius * math.cos(current_angle_A)),
            int(current_center[1] + current_radius * math.sin(current_angle_A))
        )
        current_B = (
            int(current_center[0] + current_radius * math.cos(current_angle_B)),
            int(current_center[1] + current_radius * math.sin(current_angle_B))
        )
    
    return current_A, current_B


def generate_robot_formation_animation(image: np.ndarray,
                                     point_A_start: Tuple[int, int],
                                     point_B_start: Tuple[int, int], 
                                     point_A_end: Tuple[int, int],
                                     point_B_end: Tuple[int, int],
                                     d_rob: float,
                                     r_rob: int,
                                     output_dir: str = "animation_frames",
                                     filename_prefix: str = "frame",
                                     circumference_color: Tuple[int, int, int] = (0, 255, 0),
                                     robot_color: Tuple[int, int, int] = (255, 0, 0),
                                     thickness: int = 2,
                                     min_frames: int = 5,
                                     max_frames: int = 50,
                                     pixels_per_frame: float = 10.0,
                                     save_images: bool = True,
                                     distance_range: Tuple[int, int] = (240, 360),
                                     text_scale: float = 2.4) -> list[np.ndarray]:
    """
    Generate an animation of robot formation moving from an initial 
    to a final configuration.
    
    Args:
        image: Base image to draw on
        point_A_start: Initial point A
        point_B_start: Initial point B
        point_A_end: Final point A  
        point_B_end: Final point B
        d_rob: Euclidean distance between main and additional robots
        r_rob: Radius of robot circles
        output_dir: Directory to save frames
        filename_prefix: Prefix for file names
        circumference_color: Color of circumference (B, G, R)
        robot_color: Color of robot circles (B, G, R)
        thickness: Line thickness
        min_frames: Minimum number of frames
        max_frames: Maximum number of frames  
        pixels_per_frame: Pixels of movement per frame
        save_images: If True, save images to disk
    
    Returns:
        List of images (animation frames)
    """
    import os
    
    # Calculate number of frames needed
    num_frames = calculate_animation_frames(
        point_A_start, point_B_start, point_A_end, point_B_end,
        min_frames, max_frames, pixels_per_frame
    )
    
    print(f"Generating animation with {num_frames} frames...")
    
    # Create output directory if necessary
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frames = []
    
    for i in range(num_frames + 1):  # +1 to include final frame
        # Interpolation parameter (0.0 to 1.0)
        t = i / num_frames if num_frames > 0 else 1.0
        
        # Interpolate positions of points A and B using optimized interpolation
        current_A, current_B = interpolate_circle_preserving_radius(
            point_A_start, point_B_start, point_A_end, point_B_end, t
        )
        
        # Generate current frame
        frame = draw_robot_formation(
            image.copy(),
            current_A,
            current_B, 
            d_rob,
            r_rob,
            circumference_color,
            robot_color,
            thickness,
            distance_range,
            text_scale
        )
        
        frames.append(frame)
        
        # Save frame if requested
        if save_images:
            filename = f"{filename_prefix}_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            
        # Progress feedback
        if (i + 1) % 10 == 0 or i == num_frames:
            print(f"Generated frame {i + 1}/{num_frames + 1}")
    
    print(f"Animation completed! {len(frames)} frames generated.")
    if save_images:
        print(f"Frames saved in: {output_dir}/")
    
    return frames


def create_video_from_frames(frames: list[np.ndarray],
                           output_video_path: str = "robot_animation.mp4",
                           fps: float = 30.0,
                           loop_animation: bool = True) -> bool:
    """
    Create an MP4 video from animation frames.
    
    Args:
        frames: List of frames (numpy images)
        output_video_path: Output video file path
        fps: Frames per second of the video
        loop_animation: If True, add frames in reverse for smooth loop
    
    Returns:
        True if video was created successfully
    """
    if not frames:
        print("Error: No frames provided")
        return False
    
    try:
        # Get dimensions from first frame
        height, width = frames[0].shape[:2]
        
        # Configure video codec for maximum compatibility
        # Try different codecs in order of preference for compatibility
        codecs_to_try = [
            ('H264', cv2.VideoWriter_fourcc(*'H264')),  # Best for Telegram
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Universal compatibility
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Fallback
        ]
        
        video_writer = None
        used_codec = None
        
        for codec_name, fourcc in codecs_to_try:
            try:
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                if video_writer.isOpened():
                    used_codec = codec_name
                    print(f"Using codec: {codec_name}")
                    break
                else:
                    video_writer.release()
                    video_writer = None
            except:
                if video_writer:
                    video_writer.release()
                    video_writer = None
                continue
        
        if video_writer is None:
            print("Error: No codec available")
            return False
        
        # Prepare frames for video
        video_frames = frames.copy()
        
        # Add frames in reverse for smooth loop (excluding first and last)
        if loop_animation and len(frames) > 2:
            reverse_frames = frames[-2:0:-1]  # From second-to-last to second
            video_frames.extend(reverse_frames)
        
        # Write frames to video
        for frame in video_frames:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video created: {output_video_path} (codec: {used_codec})")
        print(f"Total frames in video: {len(video_frames)}")
        print(f"Duration: {len(video_frames) / fps:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def calculate_robot_positions(point_A: Tuple[int, int], 
                            point_B: Tuple[int, int], 
                            d_rob: float) -> list[Tuple[int, int]]:
    """
    Calculate positions of 6 robots for a given configuration.
    
    Args:
        point_A: First point of diameter
        point_B: Second point of diameter
        d_rob: Distance between main and additional robots
    
    Returns:
        List of the 6 robot positions
    """
    center_x = (point_A[0] + point_B[0]) / 2
    center_y = (point_A[1] + point_B[1]) / 2
    center = (center_x, center_y)
    radius = math.sqrt((point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2) / 2
    
    if d_rob == 0:
        # Equidistant robots along the circumference
        positions = []
        for i in range(6):
            angle = i * (2 * math.pi / 6)
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            positions.append((x, y))
        return positions
    
    else:
        # Normal configuration: robA, robB + 4 additional
        positions = [point_A, point_B]  # robA and robB
        
        # Find additional robots for robA
        robA_additional = _find_additional_robots(center, radius, point_A, d_rob)
        positions.extend(robA_additional)
        
        # Find additional robots for robB  
        robB_additional = _find_additional_robots(center, radius, point_B, d_rob)
        positions.extend(robB_additional)
        
        return positions


def position_to_angle(pos: Tuple[int, int], center: Tuple[float, float]) -> float:
    """
    Convert a cartesian position to angle on the circumference.
    
    Args:
        pos: Position (x, y)
        center: Center of circumference (cx, cy)
    
    Returns:
        Angle in radians [0, 2π)
    """
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    angle = math.atan2(dy, dx)
    # Normalize to [0, 2π)
    if angle < 0:
        angle += 2 * math.pi
    return angle


def angle_to_position(angle: float, center: Tuple[float, float], radius: float) -> Tuple[int, int]:
    """
    Convert an angle to cartesian position on the circumference.
    
    Args:
        angle: Angle in radians
        center: Center of circumference
        radius: Radius of circumference
    
    Returns:
        Position (x, y)
    """
    x = int(center[0] + radius * math.cos(angle))
    y = int(center[1] + radius * math.sin(angle))
    return (x, y)


def calculate_shortest_arc_distance(angle1: float, angle2: float) -> float:
    """
    Calculate the shortest distance along the circumference between two angles.
    
    Args:
        angle1: First angle [0, 2π)
        angle2: Second angle [0, 2π)
    
    Returns:
        Shortest angular distance (always positive)
    """
    diff = abs(angle2 - angle1)
    return min(diff, 2 * math.pi - diff)


def calculate_arc_direction(angle_start: float, angle_end: float) -> int:
    """
    Determine the shortest direction along the circumference.
    
    Args:
        angle_start: Initial angle
        angle_end: Final angle
    
    Returns:
        1 for counterclockwise, -1 for clockwise
    """
    diff = angle_end - angle_start
    if diff > math.pi:
        return -1  # Orario
    elif diff < -math.pi:
        return 1   # Antiorario
    else:
        return 1 if diff >= 0 else -1


def optimize_robot_assignment(angles_start: list[float], 
                            angles_end: list[float]) -> list[int]:
    """
    Optimize robot-destination assignment to minimize movements
    and avoid crossings along the circumference.
    
    Args:
        angles_start: Initial robot angles
        angles_end: Final position angles
    
    Returns:
        List of indices that maps robot_i → position_assignment[i]
    """
    n = len(angles_start)
    if n != len(angles_end):
        raise ValueError("The number of robots and positions must be equal")
    
    # Ordina robot e posizioni per angolo
    start_sorted = sorted(enumerate(angles_start), key=lambda x: x[1])
    end_sorted = sorted(enumerate(angles_end), key=lambda x: x[1])
    
    # Assignment that maintains circular order
    assignment = [0] * n
    for i, (start_idx, _) in enumerate(start_sorted):
        end_idx = end_sorted[i][0]
        assignment[start_idx] = end_idx
    
    return assignment


def interpolate_along_arc(angle_start: float, angle_end: float, t: float) -> float:
    """
    Interpolate along the shortest arc of the circumference.
    
    Args:
        angle_start: Initial angle
        angle_end: Final angle  
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated angle
    """
    # Calculate the shortest direction
    direction = calculate_arc_direction(angle_start, angle_end)
    
    # Calculate the shortest angular distance
    diff = angle_end - angle_start
    if direction == -1 and diff > 0:
        diff -= 2 * math.pi
    elif direction == 1 and diff < 0:
        diff += 2 * math.pi
    
    # Interpola
    current_angle = angle_start + t * diff
    
    # Normalizza a [0, 2π)
    current_angle = current_angle % (2 * math.pi)
    
    return current_angle


def interpolate_robot_positions_on_circle(positions_start: list[Tuple[int, int]], 
                                        positions_end: list[Tuple[int, int]], 
                                        center: Tuple[float, float],
                                        radius: float,
                                        t: float) -> list[Tuple[int, int]]:
    """
    Interpolate robot positions along the circumference avoiding crossings.
    
    Args:
        positions_start: Initial robot positions
        positions_end: Final robot positions
        center: Center of circumference
        radius: Radius of circumference
        t: Interpolation parameter (0.0 = start, 1.0 = end)
    
    Returns:
        Interpolated robot positions
    """
    if len(positions_start) != len(positions_end):
        raise ValueError("The lists must have the same number of positions")
    
    # Convert positions to angles
    angles_start = [position_to_angle(pos, center) for pos in positions_start]
    angles_end = [position_to_angle(pos, center) for pos in positions_end]
    
    # Optimize assignment to avoid crossings
    assignment = optimize_robot_assignment(angles_start, angles_end)
    
    # Interpolate along arcs
    interpolated_positions = []
    for i, start_angle in enumerate(angles_start):
        end_angle = angles_end[assignment[i]]
        current_angle = interpolate_along_arc(start_angle, end_angle, t)
        current_pos = angle_to_position(current_angle, center, radius)
        interpolated_positions.append(current_pos)
    
    return interpolated_positions


def interpolate_robot_positions(positions_start: list[Tuple[int, int]], 
                              positions_end: list[Tuple[int, int]], 
                              t: float) -> list[Tuple[int, int]]:
    """
    Interpolate robot positions between two configurations.
    DEPRECATED: Use interpolate_robot_positions_on_circle for movements along circumference.
    
    Args:
        positions_start: Initial positions of 6 robots
        positions_end: Final positions of 6 robots
        t: Interpolation parameter (0.0 = start, 1.0 = end)
    
    Returns:
        Interpolated robot positions
    """
    if len(positions_start) != len(positions_end):
        raise ValueError("The lists must have the same number of positions")
    
    interpolated = []
    for start_pos, end_pos in zip(positions_start, positions_end):
        x = int(start_pos[0] + t * (end_pos[0] - start_pos[0]))
        y = int(start_pos[1] + t * (end_pos[1] - start_pos[1]))
        interpolated.append((x, y))
    
    return interpolated


def generate_discrete_robot_sequence(image: np.ndarray,
                                   point_A_list: list[Tuple[int, int]],
                                   point_B_list: list[Tuple[int, int]],
                                   d_rob: float,
                                   r_rob: int,
                                   pixels_per_frame: float = 10.0,
                                   output_dir: str = "discrete_sequence",
                                   filename_prefix: str = "discrete_frame",
                                   circumference_color: Tuple[int, int, int] = (0, 255, 0),
                                   robot_color: Tuple[int, int, int] = (255, 0, 0),
                                   thickness: int = 2,
                                   save_images: bool = True,
                                   distance_range: Tuple[int, int] = (240, 360),
                                   text_scale: float = 2.4) -> list[np.ndarray]:
    """
    Generate a sequence of frames from lists of A and B positions without interpolation.
    Filter positions based on pixels_per_frame parameter.
    
    Args:
        image: Base image to draw on
        point_A_list: List of point A positions
        point_B_list: List of point B positions
        d_rob: Euclidean distance between main and additional robots
        r_rob: Radius of robot circles
        pixels_per_frame: Minimum movement threshold in pixels to include a frame
        output_dir: Directory to save frames
        filename_prefix: Prefix for file names
        circumference_color: Color of circumference (B, G, R)
        robot_color: Color of robot circles (B, G, R)
        thickness: Line thickness
        save_images: If True, save images to disk
    
    Returns:
        List of images (sequence frames)
    """
    import os
    
    if len(point_A_list) != len(point_B_list):
        raise ValueError("The point_A and point_B lists must have the same length")
    
    if not point_A_list:
        return []
    
    # Filter positions based on minimum movement
    filtered_indices = [0]  # Includi sempre il primo frame
    
    for i in range(1, len(point_A_list)):
        # Calculate movement distance for both points
        dist_A = math.sqrt((point_A_list[i][0] - point_A_list[filtered_indices[-1]][0])**2 + 
                          (point_A_list[i][1] - point_A_list[filtered_indices[-1]][1])**2)
        dist_B = math.sqrt((point_B_list[i][0] - point_B_list[filtered_indices[-1]][0])**2 + 
                          (point_B_list[i][1] - point_B_list[filtered_indices[-1]][1])**2)
        
        # Usa la distanza maggiore
        max_movement = max(dist_A, dist_B)
        
        # Include frame if movement exceeds threshold
        if max_movement >= pixels_per_frame:
            filtered_indices.append(i)
    
    # Always include last frame if not already included
    if filtered_indices[-1] != len(point_A_list) - 1:
        filtered_indices.append(len(point_A_list) - 1)
    
    print(f"Generating discrete sequence: {len(filtered_indices)} frames from {len(point_A_list)} original positions")
    
    # Create output directory if necessary
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frames = []
    
    for frame_idx, pos_idx in enumerate(filtered_indices):
        # Generate frame for current position
        frame = draw_robot_formation(
            image.copy(),
            point_A_list[pos_idx],
            point_B_list[pos_idx],
            d_rob,
            r_rob,
            circumference_color,
            robot_color,
            thickness,
            distance_range,
            text_scale
        )
        
        frames.append(frame)
        
        # Save frame if requested
        if save_images:
            filename = f"{filename_prefix}_{frame_idx:04d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
        
        # Progress feedback
        if (frame_idx + 1) % 10 == 0 or frame_idx == len(filtered_indices) - 1:
            print(f"Generated frame {frame_idx + 1}/{len(filtered_indices)}")
    
    print(f"Discrete sequence completed! {len(frames)} frames generated.")
    if save_images:
        print(f"Frames saved in: {output_dir}/")
    
    return frames


def animate_d_rob_transition(image: np.ndarray,
                           point_A: Tuple[int, int],
                           point_B: Tuple[int, int],
                           d_rob_start: float,
                           d_rob_end: float,
                           r_rob: int,
                           num_frames: int = 30,
                           output_dir: str = "d_rob_transition",
                           filename_prefix: str = "transition_frame",
                           circumference_color: Tuple[int, int, int] = (0, 255, 0),
                           robot_color: Tuple[int, int, int] = (255, 0, 0),
                           thickness: int = 2,
                           save_images: bool = True,
                           distance_range: Tuple[int, int] = (240, 360),
                           text_scale: float = 2.4) -> list[np.ndarray]:
    """
    Animate the transition between two d_rob values keeping points A and B fixed.
    Robots move along the circumference.
    
    Args:
        image: Base image to draw on
        point_A: Fixed point A of diameter
        point_B: Fixed point B of diameter
        d_rob_start: Initial d_rob value
        d_rob_end: Final d_rob value
        r_rob: Robot circle radius
        num_frames: Number of animation frames
        output_dir: Directory to save frames
        filename_prefix: Prefix for file names
        circumference_color: Circumference color (B, G, R)
        robot_color: Robot circle color (B, G, R)
        thickness: Line thickness
        save_images: If True, save images to disk
    
    Returns:
        List of images (animation frames)
    """
    import os
    
    print(f"Generating d_rob transition: {d_rob_start} → {d_rob_end} with {num_frames} frames...")
    
    # Crea la directory di output se necessario
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate robot positions for initial and final configurations
    positions_start = calculate_robot_positions(point_A, point_B, d_rob_start)
    positions_end = calculate_robot_positions(point_A, point_B, d_rob_end)
    
    print(f"Initial configuration (d_rob={d_rob_start}): {len(positions_start)} robots")
    print(f"Final configuration (d_rob={d_rob_end}): {len(positions_end)} robots")
    
    frames = []
    
    # Calculate circumference parameters (fixed)
    center_x = (point_A[0] + point_B[0]) // 2
    center_y = (point_A[1] + point_B[1]) // 2
    center = (center_x, center_y)
    radius = int(math.sqrt((point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2) / 2)
    
    for i in range(num_frames + 1):
        # Interpolation parameter
        t = i / num_frames if num_frames > 0 else 1.0
        
        # Also interpolate d_rob value for correct labels
        current_d_rob = d_rob_start + t * (d_rob_end - d_rob_start)
        
        # Interpolate robot positions ALONG THE CIRCUMFERENCE
        current_positions = interpolate_robot_positions_on_circle(
            positions_start, positions_end, center, radius, t
        )
        
        # Create frame
        frame = image.copy()
        
        # Draw circumference (always the same)
        cv2.circle(frame, center, radius, circumference_color, thickness)
        
        # Draw robots at interpolated positions
        for j, pos in enumerate(current_positions):
            cv2.circle(frame, pos, r_rob, robot_color, -1)
        
        # Draw black box with distance information
        distance = radius * 2  # Diameter of circumference
        # Scale dimensions based on text size (more generously)
        box_width = int(250 * text_scale)  # Wider to contain text
        box_height = int(60 * text_scale)  # Taller to contain text
        cv2.rectangle(frame, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
        
        # Determine text color based on range
        if distance_range[0] <= distance <= distance_range[1]:
            text_color = (0, 255, 0)  # Green
        else:
            text_color = (0, 0, 255)  # Red
        
        # Draw distance text
        text = f"DISTANCE: {distance:.0f}"
        text_thickness = max(1, int(2 * text_scale))
        text_x_offset = int(20 * text_scale)  # Proportional left margin
        text_y_offset = int(45 * text_scale)  # Centered vertical position
        cv2.putText(frame, text, (text_x_offset, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
        
        frames.append(frame)
        
        # Save frame if requested
        if save_images:
            filename = f"{filename_prefix}_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
        
        # Progress feedback
        if (i + 1) % 10 == 0 or i == num_frames:
            print(f"Generated frame {i + 1}/{num_frames + 1}")
    
    print(f"d_rob transition completed! {len(frames)} frames generated.")
    if save_images:
        print(f"Frames saved in: {output_dir}/")
    
    return frames


def create_d_rob_transition_video(frames: list[np.ndarray],
                                output_video_path: str = "d_rob_transition.mp4",
                                fps: float = 20.0,
                                loop_back: bool = True) -> bool:
    """
    Create a video from d_rob transition.
    
    Args:
        frames: List of animation frames
        output_video_path: Video file path
        fps: Frames per second
        loop_back: If True, add animation in reverse for smooth loop
    
    Returns:
        True if video was created successfully
    """
    if not frames:
        print("Error: No frames provided")
        return False
    
    try:
        height, width = frames[0].shape[:2]
        
        # Configure video codec for maximum compatibility
        codecs_to_try = [
            ('H264', cv2.VideoWriter_fourcc(*'H264')),  # Best for Telegram
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Universal compatibility
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Fallback
        ]
        
        video_writer = None
        used_codec = None
        
        for codec_name, fourcc in codecs_to_try:
            try:
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                if video_writer.isOpened():
                    used_codec = codec_name
                    print(f"Using codec: {codec_name}")
                    break
                else:
                    video_writer.release()
                    video_writer = None
            except:
                if video_writer:
                    video_writer.release()
                    video_writer = None
                continue
        
        if video_writer is None:
            print("Error: No codec available")
            return False
        
        video_frames = frames.copy()
        
        # Add frames in reverse for smooth loop
        if loop_back and len(frames) > 2:
            reverse_frames = frames[-2:0:-1]
            video_frames.extend(reverse_frames)
        
        for frame in video_frames:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video created: {output_video_path} (codec: {used_codec})")
        print(f"Total frames in video: {len(video_frames)}")
        print(f"Duration: {len(video_frames) / fps:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


if __name__ == "__main__":
    # Function test
    
    # Crea un'immagine di test
    test_img = cv2.imread('maps/map_final.png')
    
    # Definisce i punti A e B
    point_A = (1044 + 120, 2140)
    point_B = (1044 - 120, 2140)
    point_A_end = (1044, 2140 + 120)
    point_B_end = (1044, 2140 - 120)

    """frames =generate_robot_formation_animation(
        image=test_img,
        point_A_start=point_A,
        point_B_start=point_B,
        point_A_end=point_A_end,
        point_B_end=point_B_end,
        d_rob=35,
        r_rob=15,
        output_dir="animation_test",
        filename_prefix="test_frame",
        pixels_per_frame=10.0,
        circumference_color=(255, 0, 0),  # Blue
        robot_color=(0, 255, 0),          # Verde
        thickness=2,
        save_images=False
    )

    for frame in frames:
        cv2.imshow("Robot Formation Animation", frame)
        cv2.waitKey(0)  # Mostra ogni frame
    cv2.destroyAllWindows()"""
    
    """# Test 1: Normal case with d_rob > 0
    print("Test 1: Normal case (d_rob = 100)")
    result1 = draw_robot_formation(test_img, point_A, point_B, d_rob=35, r_rob=15)
    cv2.imwrite('test_formation_normal.png', result1)"""

    frames = animate_d_rob_transition(
        image=test_img,
        point_A=point_A,
        point_B=point_B,
        d_rob_start=35,
        d_rob_end=0,
        r_rob=15,
        num_frames=30,
        output_dir="d_rob_transition_test",
        filename_prefix="d_rob_frame",
        circumference_color=(255, 0, 0),  # Blue
        robot_color=(0, 255, 0),          # Verde
        thickness=2,
        save_images=False
    )
    for frame in frames:
        cv2.imshow("d_rob Transition Animation", frame)
        cv2.waitKey(0)  # Mostra ogni frame
    cv2.destroyAllWindows()
    
    """# Test 2: Special case with d_rob = 0
    print("Test 2: Special case (d_rob = 0)")
    result2 = draw_robot_formation(test_img, point_A, point_B, d_rob=0, r_rob=15)
    cv2.imwrite('test_formation_equidistant.png', result2)
    
    # Test 3: Case with diagonal points
    print("Test 3: Diagonal points")
    point_C = (150, 150)
    point_D = (650, 450)
    result3 = draw_robot_formation(test_img, point_C, point_D, d_rob=80, r_rob=12)
    cv2.imwrite('test_formation_diagonal.png', result3)
    
    print("Tests completed! Check the files:")
    print("- test_formation_normal.png")
    print("- test_formation_equidistant.png") 
    print("- test_formation_diagonal.png")"""