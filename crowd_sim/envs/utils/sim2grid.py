from math import sqrt
from crowd_nav.utils.theta_star import Grid
import numpy as np

# calculate the intersection point
def first_circle_line_intersection(robot_pos, line_points, start_radius=1, max_radius=10, step=1):
    cx, cy = robot_pos
    (x1, y1), (x2, y2) = line_points
    gx, gy = line_points[1]

    for radius in np.arange(start_radius, max_radius + step, step):
        dx = x2 - x1
        dy = y2 - y1
        
        # Quadratic coefficients
        a = dx**2 + dy**2
        b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        c = (x1 - cx)**2 + (y1 - cy)**2 - radius**2
        
        discriminant = b**2 - 4 * a * c
        
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            t_candidates = [(-b + sqrt_disc) / (2 * a),
                            (-b - sqrt_disc) / (2 * a)]
            
            # Filter intersections within segment
            intersections = []
            for t in t_candidates:
                if 0 <= t <= 1:
                    ix = x1 + t * dx
                    iy = y1 + t * dy
                    intersections.append((ix, iy))
            
            if intersections:
                # If multiple, pick nearest to goal
                best_point = min(intersections, key=lambda p: (p[0]-gx)**2 + (p[1]-gy)**2)
                return best_point, radius
    
    return None, None  # No intersection found

# Function to transfer CrowdSim to A* Grid
def drl_to_grid(crowd_sim, cell_size=0.5, boundary_buf=2.0):
    """
    Discretize CrowdSim's continuous map into a Grid for A* planning.
    
    Parameters:
        crowd_sim (CrowdSim): The DRL environment instance.
        cell_size (float): Size of each grid cell in continuous units (default 0.5).
    
    Returns:
        grid (Grid): The discretized grid with obstacles.
        min_x (float): Minimum x for coordinate conversion.
        min_y (float): Minimum y for coordinate conversion.
        cell_size (float): The cell size used.
    """
    # Compute bounds from robot, humans, and static obstacles
    min_x = min(crowd_sim.robot.px, crowd_sim.robot.gx)
    max_x = max(crowd_sim.robot.px, crowd_sim.robot.gx)
    min_y = min(crowd_sim.robot.py, crowd_sim.robot.gy)
    max_y = max(crowd_sim.robot.py, crowd_sim.robot.gy)
    for human in crowd_sim.humans:
        min_x = min(min_x, human.px, human.gx)
        max_x = max(max_x, human.px, human.gx)
        min_y = min(min_y, human.py, human.gy)
        max_y = max(max_y, human.py, human.gy)
    for obs in crowd_sim.static_obstacles:
        if obs['type'] == 'circle':
            cx, cy, r = obs['cx'], obs['cy'], obs['r']
            min_x = min(min_x, cx - r)
            max_x = max(max_x, cx + r)
            min_y = min(min_y, cy - r)
            max_y = max(max_y, cy + r)
        elif obs['type'] == 'rect':
            cx, cy, w, h = obs['cx'], obs['cy'], obs['w'], obs['h']
            min_x = min(min_x, cx - w/2)
            max_x = max(max_x, cx + w/2)
            min_y = min(min_y, cy - h/2)
            max_y = max(max_y, cy + h/2)
    
    # Add buffer for safety
    min_x -= boundary_buf
    max_x += boundary_buf
    min_y -= boundary_buf
    max_y += boundary_buf
    
    x_range = int((max_x - min_x) / cell_size) + 1
    y_range = int((max_y - min_y) / cell_size) + 1
    
    grid = Grid(x_range, y_range, 10)
    obstacles = set()
    
    # Mark cells intersecting static obstacles
    for i in range(x_range):
        for j in range(y_range):
            cell_center_x = min_x + (i + 0.5) * cell_size
            cell_center_y = min_y + (j + 0.5) * cell_size
            half_diag = cell_size * sqrt(2) / 2  # Half diagonal for intersection check
            
            for obs in crowd_sim.static_obstacles:
                if obs['type'] == 'circle':
                    cx, cy, r = obs['cx'], obs['cy'], obs['r']
                    if np.hypot(cell_center_x - cx, cell_center_y - cy) < r + half_diag:
                        obstacles.add((i, j))
                        break
                elif obs['type'] == 'rect':
                    cx, cy, w, h = obs['cx'], obs['cy'], obs['w'], obs['h']
                    # print('static grid obst:', (cx, cy, w, h))
                    left, right = cx - w/2, cx + w/2
                    bottom, top = cy - h/2, cy + h/2
                    cell_left, cell_right = cell_center_x - cell_size/2, cell_center_x + cell_size/2
                    cell_bottom, cell_top = cell_center_y - cell_size/2, cell_center_y + cell_size/2
                    if not (cell_right < left or cell_left > right or cell_top < bottom or cell_bottom > top):
                        obstacles.add((i, j))
                        break
            
            # If static sim, add humans as obstacles
            if crowd_sim.test_sim == 'static':
                for human in crowd_sim.humans:
                    cx, cy, r = human.px, human.py, human.radius
                    if np.hypot(cell_center_x - cx, cell_center_y - cy) < r + half_diag:
                        obstacles.add((i, j))
                        break
    
    grid.update(obstacles.union(grid.obstacles))  # Add to existing boundaries
    return grid, min_x, min_y, cell_size