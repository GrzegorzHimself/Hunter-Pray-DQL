# ------------------- Bresenham Supercover ------------------- #
def bresenham(x1, y1, x2, y2):
    """A 'supercover' version of Bresenham's line algorithm that ensures all
    cells touched by the line are included.

    Args:
        x1 (int): X coordinated of the Agent
        y1 (int): Y coordinate of the Agent
        x2 (int): X coordinate of the target cell
        y2 (int): Y coordinate of the etarget cell

    Returns:
        list: A list of "visible" cells
    """
    points = []
    # Record the first point
    points.append((x1, y1))
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Determine directions of steps
    xstep = 1 if dx >= 0 else -1
    ystep = 1 if dy >= 0 else -1
    
    dx = abs(dx)
    dy = abs(dy)
    
    # Double increments for dx, dy
    ddx = 2 * dx
    ddy = 2 * dy
    
    x, y = x1, y1  # start at first cell
    
    if dx >= dy:
        errorprev = error = dx  # start error in the middle
        for i in range(dx):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    points.append((x, y - ystep))
                elif error + errorprev > ddx:
                    points.append((x - xstep, y))
                else:
                    points.append((x, y - ystep))
                    points.append((x - xstep, y))
            points.append((x, y))
            errorprev = error
    else:
        errorprev = error = dy
        for i in range(dy):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    points.append((x - xstep, y))
                elif error + errorprev > ddy:
                    points.append((x, y - ystep))
                else:
                    points.append((x - xstep, y))
                    points.append((x, y - ystep))
            points.append((x, y))
            errorprev = error
            
    return points