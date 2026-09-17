import pytest
import numpy as np
from lunar_platform.navigation.astar import plan_path_astar

def test_astar_straight_line():
    # 10x10 map of low cost
    cost_map = np.ones((10, 10))
    start = (0, 0)
    goal = (9, 9)
    
    path = plan_path_astar(cost_map, start, goal)
    
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal
    # A straight diagonal path should be roughly 10 points
    assert len(path) == 10

def test_astar_obstacle():
    # 10x10 map
    cost_map = np.ones((10, 10))
    
    # Create an impassable wall in the middle
    cost_map[0:8, 5] = np.inf
    
    start = (0, 5)
    goal = (9, 5)
    
    path = plan_path_astar(cost_map, start, goal)
    
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal
    
    # Path must go around the wall (y >= 8)
    for x, y in path:
        if x == 5:
            assert y >= 8

def test_astar_no_path():
    cost_map = np.ones((10, 10))
    
    # Complete wall
    cost_map[:, 5] = np.inf
    
    start = (0, 5)
    goal = (9, 5)
    
    path = plan_path_astar(cost_map, start, goal)
    
    assert len(path) == 0
