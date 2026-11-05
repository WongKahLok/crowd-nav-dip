from conftest import write_map
from test_session import plan
from crowd_nav.utils.theta_star import AStar, ThetaStar, Grid
import math

def test_open_grid_routes():
    grid = Grid(20, 20)
    a, ap, _ = AStar((2, 2), (12, 7), grid).plan()
    t, tp, _ = ThetaStar((2, 2), (12, 7), grid).plan()
    assert math.isclose(a, 5 + 5 * math.sqrt(2))
    assert math.isclose(t, math.hypot(10, 5))
    assert ap[0] == tp[0] == (12, 7)
    assert ap[-1] == tp[-1] == (2, 2)

def test_blocked_route():
    grid = Grid(12, 12)
    grid.update(grid.obstacles | {(6, y) for y in range(12)})
    for planner in (AStar, ThetaStar):
        assert not planner((2, 2), (9, 9), grid).plan()[1]
