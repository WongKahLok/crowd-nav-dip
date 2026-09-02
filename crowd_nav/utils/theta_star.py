import heapq
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from math import sqrt
from scipy.spatial import cKDTree
from abc import abstractmethod, ABC

class Node(object):
    """
    Class for searching nodes.

    Parameters:
        current (tuple): current coordinate
        parent (tuple): coordinate of parent node
        g (float): path cost
        h (float): heuristic cost

    Examples:
        >>> from env import Node
        >>> node1 = Node((1, 0), (2, 3), 1, 2)
        >>> node2 = Node((1, 0), (2, 5), 2, 8)
        >>> node3 = Node((2, 0), (1, 6), 3, 1)
        ...
        >>> node1 + node2
        >>> Node((2, 0), (2, 3), 3, 2)
        ...
        >>> node1 == node2
        >>> True
        ...
        >>> node1 != node3
        >>> True
    """
    def __init__(self, current: tuple, parent: tuple = None, g: float = 0, h: float = 0) -> None:
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
    
    def __add__(self, node):
        assert isinstance(node, Node)
        return Node((self.x + node.x, self.y + node.y), self.parent, self.g + node.g, self.h)

    def __eq__(self, node) -> bool:
        if not isinstance(node, Node):
            return False
        return self.current == node.current
    
    def __ne__(self, node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        assert isinstance(node, Node)
        return self.g + self.h < node.g + node.h or \
                (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.current)

    def __str__(self) -> str:
        return "Node({}, {}, {}, {})".format(self.current, self.parent, self.g, self.h)

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def x(self) -> float:
        return self.current[0]
    
    @property
    def y(self) -> float:
        return self.current[1]

    @property
    def px(self) -> float:
        if self.parent:
            return self.parent[0]
        else:
            return None

    @property
    def py(self) -> float:
        if self.parent:
            return self.parent[1]
        else:
            return None
        
class Env(ABC):
    """
    Class for building 2-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        eps (float): tolerance for float comparison

    Examples:
        >>> from python_motion_planning.utils import Env
        >>> env = Env(30, 40)
    """
    def __init__(self, x_range: int, y_range: int, eps: float = 1e-6) -> None:
        # size of environment
        self.x_range = x_range  
        self.y_range = y_range
        self.eps = eps

    @property
    def grid_map(self) -> set:
        return {(i, j) for i in range(self.x_range) for j in range(self.y_range)}

    @abstractmethod
    def init(self) -> None:
        pass

class Grid(Env):
    """
    Class for discrete 2-d grid map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
    """
    def __init__(self, x_range: int, y_range: int, safety_distance=0) -> None:
        super().__init__(x_range, y_range)
        # allowed motions
        self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1),  None, sqrt(2), None),
                        Node((0, 1),  None, 1, None), Node((1, 1),   None, sqrt(2), None),
                        Node((1, 0),  None, 1, None), Node((1, -1),  None, sqrt(2), None),
                        Node((0, -1), None, 1, None), Node((-1, -1), None, sqrt(2), None)]
        # obstacles
        self.obstacles = None
        self.obstacles_tree = None
        self.safety_distance = safety_distance
        self.init()
    
    def init(self) -> None:
        """
        Initialize grid map.
        """
        x, y = self.x_range, self.y_range
        obstacles = set()

        # boundary of environment
        for i in range(x):
            obstacles.add((i, 0))
            obstacles.add((i, y - 1))
        for i in range(y):
            obstacles.add((0, i))
            obstacles.add((x - 1, i))

        self.update(obstacles)

    def update(self, obstacles):
        self.obstacles = obstacles 
        self.obstacles_tree = cKDTree(np.array(list(obstacles)))

        inflated_obstacles = set()
        for ox, oy in obstacles:
            for dx in range(-self.safety_distance, self.safety_distance + 1):
                for dy in range(-self.safety_distance, self.safety_distance + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < self.x_range and 0 <= ny < self.y_range:
                        if math.hypot(dx, dy) <= self.safety_distance:
                            inflated_obstacles.add((nx, ny))
        self.obstacles = inflated_obstacles


class Map(Env):
    """
    Class for continuous 2-d map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
    """
    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        self.boundary = None
        self.obs_circ = None
        self.obs_rect = None
        self.init()

    def init(self):
        """
        Initialize map.
        """
        x, y = self.x_range, self.y_range

        # boundary of environment
        self.boundary = [
            [0, 0, 1, y],
            [0, y, x, 1],
            [1, 0, x, 1],
            [x, 1, 1, y]
        ]
        self.obs_rect = []
        self.obs_circ = []

    def update(self, boundary=None, obs_circ=None, obs_rect=None):
        self.boundary = boundary if boundary else self.boundary
        self.obs_circ = obs_circ if obs_circ else self.obs_circ
        self.obs_rect = obs_rect if obs_rect else self.obs_rect

class Planner(ABC):
    def __init__(self, start: tuple, goal: tuple, env: Env) -> None:
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env

    def dist(self, node1: Node, node2: Node) -> float:
        return math.hypot(node2.x - node1.x, node2.y - node1.y)
    
    def angle(self, node1: Node, node2: Node) -> float:
        return math.atan2(node2.y - node1.y, node2.x - node1.x)

    @abstractmethod
    def plan(self):
        '''
        Interface for planning.
        '''
        pass

    @abstractmethod
    def run(self):
        '''
        Interface for running both plannig and animation.
        '''
        pass

class GraphSearcher(Planner):
    """
    Base class for planner based on graph searching.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment
        heuristic_type (str): heuristic function type
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str="euclidean") -> None:
        super().__init__(start, goal, env)
        # heuristic type
        self.heuristic_type = heuristic_type
        # allowed motions
        self.motions = self.env.motions
        # obstacles
        self.obstacles = self.env.obstacles

    def h(self, node: Node, goal: Node) -> float:
        """
        Calculate heuristic.

        Parameters:
            node (Node): current node
            goal (Node): goal node

        Returns:
            h (float): heuristic function value of node
        """
        if self.heuristic_type == "manhattan":
            return abs(goal.x - node.x) + abs(goal.y - node.y)
        elif self.heuristic_type == "euclidean":
            return math.hypot(goal.x - node.x, goal.y - node.y)

    def cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost for this motion.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            cost (float): cost of this motion
        """
        if self.isCollision(node1, node2):
            return float("inf")
        return self.dist(node1, node2)

    def isCollision(self, node1: Node, node2: Node) -> bool:
        """
        Judge collision when moving from node1 to node2.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            collision (bool): True if collision exists else False
        """
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return True

        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

        if x1 != x2 and y1 != y2:
            if x2 - x1 == y1 - y2:
                s1 = (min(x1, x2), min(y1, y2))
                s2 = (max(x1, x2), max(y1, y2))
            else:
                s1 = (min(x1, x2), max(y1, y2))
                s2 = (max(x1, x2), min(y1, y2))
            if s1 in self.obstacles or s2 in self.obstacles:
                return True
        return False

class HybridEnv(Grid):
    def __init__(self, x_range: int, y_range: int, resolution: float = 0.3, safety_distance=0):
        self.resolution = resolution
        self.safety_distance = safety_distance
        super().__init__(x_range, y_range, safety_distance)
        
    def load_from_image(self, img_path: str, threshold: int = 128):
        static_map = Image.open(img_path).convert('L')
        static_map = static_map.resize((self.x_range, self.y_range))
        static_map_array = np.array(static_map)
        obstacles = set()
        for i in range(self.x_range):
            for j in range(self.y_range):
                if static_map_array[i, j] < threshold:
                    obstacles.add((j, self.x_range-1-i))
        for i in range(self.x_range):
            obstacles.add((i, 0))                      
            obstacles.add((i, self.y_range-1))       
        for j in range(self.y_range):
            obstacles.add((0, j))                      
            obstacles.add((self.x_range-1, j))  
        self.update(obstacles)
        self.x_min = 0
        self.x_max = self.x_range*self.resolution
        self.y_min = 0
        self.y_max = self.y_range*self.resolution

    def get_static_obstacles(self):
        static_obs_list = []
        for (obs_idx_x, obs_idx_y) in self.obstacles:
            cx = obs_idx_x * self.resolution + self.resolution/2
            cy = obs_idx_y * self.resolution + self.resolution/2
            
            static_obs_list.append({
                'type': 'rect',
                'cx': cx, 
                'cy': cy, 
                'w': self.resolution,
                'h': self.resolution 
            })
        return static_obs_list
    
    def get_env_para(self):
        env_para = {'x_min':self.x_min, 
                    'x_max':self.x_max,
                    'y_min':self.y_min,
                    'y_max':self.y_max,
                    'grid_res': self.resolution,
                    'nearby_dist': 5.0,
                    }
        
        return env_para
    
    def grid_2_drl(self, grid_path):
        drl_path = []
        for (x,y) in grid_path:
            drl_path.append((x+self.resolution/2, y+self.resolution/2))
        return drl_path

    def world_to_grid(self, x: float, y: float) -> tuple:
        return (int(round(x / self.resolution)), int(round(y / self.resolution)))

    def grid_to_world(self, i: int, j: int) -> tuple:
        return (i * self.resolution, j * self.resolution)
    
    def is_far_enough(self, pos, forb_list, min_dist=1.0):
        """Spawn-validity test: pos must be in free space and >= min_dist from every forbidden point.

        Called by generate_dynamic_obstacles() but never defined, so that generator raised
        AttributeError the moment it was used; this restores it.

        Args:
            pos (tuple): candidate (x, y) in world coordinates
            forb_list (list): forbidden (x, y) world points, e.g. the robot's start and goal
            min_dist (float): required clearance from each forbidden point
        """
        if self.world_to_grid(*pos) in self.obstacles:  # inside land (or its inflation margin)
            return False
        for forb in forb_list:
            if math.dist(pos, (forb[0], forb[1])) < min_dist:
                return False
        return True

    def generate_dynamic_obstacles(self, n_obstacles, forb_list, min_dist=1.0):
        obstacles = []
        while len(obstacles) < n_obstacles:
            valid_start = False
            while not valid_start:
                start_pos = (np.random.uniform(self.x_min, self.x_max), np.random.uniform(self.y_min, self.y_max))
                if self.is_far_enough(start_pos, forb_list, min_dist):
                    valid_start = True
            valid_goal = False
            while not valid_goal:
                goal_pos = (np.random.uniform(self.x_min, self.x_max), np.random.uniform(self.y_min, self.y_max))
                if self.is_far_enough(goal_pos, forb_list, min_dist) and \
                    np.linalg.norm(np.array(goal_pos) - np.array(start_pos)) > min_dist:
                    valid_goal = True
            
            obstacles.append({
                'px': start_pos[0],
                'py': start_pos[1],
                'gx': goal_pos[0],
                'gy': goal_pos[1]
            })
            
        return obstacles
    
class AStar(GraphSearcher):
    """
    Class for A* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.AStar((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, expand = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] A Formal Basis for the heuristic Determination of Minimum Cost Paths
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "A*"

    def plan(self) -> tuple:
        """
        A* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue
                
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node
        return [], [], []

    def getNeighbor(self, node: Node) -> list:
        """
        Find neighbors of node.

        Parameters:
            node (Node): current node

        Returns:
            neighbors (list): neighbors of current node
        """
        return [node + motion for motion in self.motions
                if not self.isCollision(node, node + motion)]

    def extractPath(self, closed_list: dict) -> tuple:
        """
        Extract the path based on the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns:
            cost (float): the cost of planned path
            path (list): the planning path
        """
        cost = 0
        node = closed_list[self.goal.current]
        path = [node.current]
        while node != self.start:
            node_parent = closed_list[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, path

    def run(self):
        """
        Running both planning and animation.
        """
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

class ThetaStar(AStar):
    """
    Class for Theta* motion planning.

    Parameters:
        start (tuple):
            start point coordinate
        goal (tuple):
            goal point coordinate
        env (Grid):
            environment
        heuristic_type (str):
            heuristic function type

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.ThetaStar((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, expand = planner.plan()
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] Theta*: Any-Angle Path Planning on Grids
        [2] Any-angle path planning on non-uniform costmaps
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "Theta*"

    def plan(self) -> tuple:
        """
        Theta* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue
                
                # path1
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)

                node_p = CLOSED.get(node.parent)

                if node_p:
                    self.updateVertex(node_p, node_n)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node
        return [], [], []

    def updateVertex(self, node_p: Node, node_c: Node) -> None:
        """
        Update extend node information with current node's parent node.

        Parameters:
            node_p (Node): parent node
            node_c (Node): current node
        """
        if self.lineOfSight(node_c, node_p):
            # path 2
            if node_p.g + self.dist(node_c, node_p) <= node_c.g:
                node_c.g = node_p.g + self.dist(node_c, node_p)
                node_c.parent = node_p.current

    def lineOfSight(self, node1: Node, node2: Node) -> bool:
        """
        Judge collision when moving from node1 to node2 using Bresenham.

        Parameters:
            node1 (Node): start node
            node2 (Node): end node

        Returns:
            line_of_sight (bool): True if line of sight exists ( no collision ) else False
        """
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return False
        
        x1, y1 = node1.current
        x2, y2 = node2.current

        if x1 < 0 or x1 >= self.env.x_range or y1 < 0 or y1 >= self.env.y_range:
            return False
        if x2 < 0 or x2 >= self.env.x_range or y2 < 0 or y2 >= self.env.y_range:
            return False

        d_x = abs(x2 - x1)
        d_y = abs(y2 - y1)
        s_x = 0 if (x2 - x1) == 0 else (x2 - x1) / d_x
        s_y = 0 if (y2 - y1) == 0 else (y2 - y1) / d_y
        x, y, e = x1, y1, 0

        # check if any obstacle exists between node1 and node2
        if d_x > d_y:
            tau = (d_y - d_x) / 2
            while not x == x2:
                if e > tau:
                    x = x + s_x
                    e = e - d_y
                elif e < tau:
                    y = y + s_y
                    e = e + d_x
                else:
                    x = x + s_x
                    y = y + s_y
                    e = e + d_x - d_y
                if (x, y) in self.obstacles:
                    return False
        # swap x and y
        else:
            tau = (d_x - d_y) / 2
            while not y == y2:
                if e > tau:
                    y = y + s_y
                    e = e - d_x
                elif e < tau:
                    x = x + s_x
                    e = e + d_y
                else:
                    x = x + s_x
                    y = y + s_y
                    e = e + d_y - d_x
                if (x, y) in self.obstacles:
                    return False
        
        return True

def get_local_goal(ship_state, obstacle_states, dense_path, goal_dist=2.0, monitor_dist=5.0):
    num_points = len(dense_path)
    ship_pos = (ship_state.px, ship_state.py)

    obst_monitored_states, obst_monitored_dist = [], []
    for one_obst_state in obstacle_states:
        if abs(one_obst_state.vx) + abs(one_obst_state.vy) < 1e-10:
                continue
        
        obst_pos = (one_obst_state.px, one_obst_state.py)
        obst_dist = math.dist(obst_pos, ship_pos)

        if obst_dist >= monitor_dist:
            continue
        
        obst_monitored_states.append(one_obst_state)
        obst_monitored_dist.append(obst_dist)

    distances = [math.dist(ship_pos, pt) for pt in dense_path]
    nearest_idx = np.argmin(distances)
    look_ahead_dist = monitor_dist if len(obst_monitored_states)>0 else goal_dist
    for i in range(nearest_idx, num_points):
        if math.dist(ship_pos, dense_path[i]) >= look_ahead_dist:
            return dense_path[i]
        
    return dense_path[-1]

def interpolate_path(path, step_size=0.2):
    new_path = []
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        segment_vector = p2 - p1
        distance = np.linalg.norm(segment_vector)
        
        if distance == 0: continue
        
        num_points = max(int(distance / step_size), 1)
        for t in np.linspace(0, 1, num_points, endpoint=False):
            point = p1 + t * segment_vector
            new_path.append(tuple(point))
            
    new_path.append(path[-1])
    return new_path

def main():
    env = Grid(51, 31)

if __name__ == '__main__':
    main()