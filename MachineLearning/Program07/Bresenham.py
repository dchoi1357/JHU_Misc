def pathWide(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = 2*dy - dx
    y = y0

    path = [None] * (dx+1) # pre-allocate list of paths
    for n,x in enumerate( range(x0,x1+1) ):
        path[n] = (x,y)
        if D > 0:
            y += yi
            D -= 2*dx
        D += 2*dy
    return path

def pathTall(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2*dx - dy
    x = x0
    
    path = [None] * (dy+1) # pre-allocate list of paths
    for n,y in enumerate( range(y0,y1+1) ):
        path[n] = (x,y)
        if D > 0:
            x += xi
            D -= 2*dy
        D += 2*dx
    return path

def getPath(x0, y0, dx, dy):
    x1,y1 = (x0+dx, y0+dy)
    
    if abs(dy) < abs(dx):
        if x0 > x1:
            tmp = pathWide(x1, y1, x0, y0)
            tmp.reverse()
            return tmp[1:]
        else:
            return pathWide(x0, y0, x1, y1)[1:]
    else:
        if y0 > y1:
            tmp = pathTall(x1, y1, x0, y0)
            tmp.reverse()
            return tmp[1:]
        else:
            return pathTall(x0, y0, x1, y1)[1:]