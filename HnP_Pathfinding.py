import heapq


# ------------------- Pathfinding ------------------- #
def cell_cost_hunter(x, y, walls, grid_size):
    """Calculates the cost of moving to (x, y). Lower == better

    Args:
        x (int): X coordinate of the cell
        y (int): Y coordinate of the cell
        walls (list ): A list of walls coordinates
        grid_size (int): The grid_size*grid_size size of the playground

    Returns:
        int: Cost for the cell
    """
    base_cost = 1.0
    patch_radius = 1
    free_count = 0
    count = 0
    for dx in range(-patch_radius, patch_radius + 1):
        for dy in range(-patch_radius, patch_radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                count += 1
                if walls[nx][ny] != "w":
                    free_count += 1
    view_quality = free_count / count  # number for free tiles around (max 1.0)
    beta = 0.5  # penalty multiplier
    multiplier = 1 - beta * view_quality
    cost = base_cost * multiplier
    return cost

def a_star_distance_for_hunter(walls, start, goal, grid_size):
    """Modified A* for Hunter, prefers to go through cells with the largest FOV

    Args:
        walls (list): A list of walls coordinates
        start (list): Start coordinates
        goal (list): End coordinates
        grid_size (int): The grid_size*grid_size size of the playground

    Returns:
        int: A* distance
    """
    if start == goal:
        return 0
    (sx, sy) = start
    (gx, gy) = goal
    if walls[sx][sy] == "w" or walls[gx][gy] == "w":
        return None
    open_set = []
    heapq.heappush(open_set, (0, sx, sy))
    cost_so_far = {(sx, sy): 0}
    
    def heuristic(ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)
    
    while open_set:
        priority, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == (gx, gy):
            return cost_so_far[(cx, cy)]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and walls[nx][ny] == ".":
                new_cost = cost_so_far[(cx, cy)] + cell_cost_hunter(nx, ny, walls, grid_size)
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    prio = new_cost + heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_set, (prio, nx, ny))
    return None

def a_star_distance_modified(walls, start, goal, grid_size):
    """Modified A* for Prey with embeded cost calculation function, prefers higher cost

    Args:
        walls (list): A list of walls coordinates
        start (list): Start coordinates
        goal (list): End coordinates
        grid_size (int): The grid_size*grid_size size of the playground

    Returns:
        int: A* distance
    """
    if start == goal:
        return 0
    (sx, sy) = start
    (gx, gy) = goal
    if walls[sx][sy] == "w" or walls[gx][gy] == "w":
        return None
    open_set = []
    heapq.heappush(open_set, (0, sx, sy))
    cost_so_far = {(sx, sy): 0}
    
    def heuristic(ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)
    
    def cell_cost(x, y):
        cost = 1.0
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if walls[nx][ny] == "w":
                    cost = 0.8
                    break
        return cost
    
    while open_set:
        priority, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == (gx, gy):
            return cost_so_far[(cx, cy)]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and walls[nx][ny] == ".":
                new_cost = cost_so_far[(cx, cy)] + cell_cost(nx, ny)
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    prio = new_cost + heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_set, (prio, nx, ny))
    return None