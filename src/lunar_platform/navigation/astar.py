import numpy as np
import heapq
from typing import List, Tuple

def heuristic(a: Tuple[int, int], b: Tuple[int, int], weight: float = 1.0) -> float:
    # Diagonal distance (Chebyshev) or Euclidean
    # Using Euclidean for smoother paths
    return weight * np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def plan_path_astar(cost_map: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], config: dict = {}) -> List[Tuple[int, int]]:
    """
    Finds the lowest-cost path from start to goal using A*.
    cost_map: 2D array where np.inf means impassable.
    start/goal: (x, y) tuples.
    Returns a list of (x, y) coordinates representing the path.
    """
    height, width = cost_map.shape
    
    if not (0 <= start[0] < width and 0 <= start[1] < height):
        raise ValueError("Start out of bounds")
    if not (0 <= goal[0] < width and 0 <= goal[1] < height):
        raise ValueError("Goal out of bounds")
        
    if np.isinf(cost_map[start[1], start[0]]):
        raise ValueError("Start is on an impassable cell")
    if np.isinf(cost_map[goal[1], goal[0]]):
        raise ValueError("Goal is on an impassable cell")

    h_weight = config.get("heuristic_weight", 1.0)
    
    # Priority queue: (f_score, (x, y))
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    
    came_from = {}
    
    # Cost from start to current node
    g_score = {start: 0.0}
    
    # Default to infinity for unexplored nodes
    # For performance, we won't initialize the whole grid, just use get(node, inf)
    
    # 8-connected grid
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
            
        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                neighbor = (nx, ny)
                step_cost = cost_map[ny, nx]
                
                if np.isinf(step_cost):
                    continue
                    
                # Diagonal moves should cost more in grid terms
                distance_factor = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + (step_cost * distance_factor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal, h_weight)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return [] # Path not found
