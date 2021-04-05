import numpy as np
from stl import mesh
from numba import jit, prange
import time



# Create the mesh
cube = mesh.Mesh(np.zeros(64000, dtype=mesh.Mesh.dtype))

@jit(nopython=True)
def f(x,y,z):
    r = 8
    return 1 if r**2 > ((2*x)**2 + y**2 + z**2) else 0


@jit(nopython=True,parallel=True)
def findSurfacePoint(fun, startPnt, direction, r, res):
    t = 0
    d = 1
    suc = 0
    normDir = direction/np.sum(direction**2)
    for i in range(res):
        step = r/(2.**i)
        p = startPnt + t*normDir
        s = fun(p[0],p[1],p[2])
        if i == 0:
            so = s
        if 0 != s - so:
            suc = 1
            d *= -1
        t += step*d
        so = s
    return p, suc



@jit(nopython=True,parallel=True)
def render(fun,v):
    pnts = np.zeros((100000,3))
    i = 0
    suc = 0
    for x in range(-10,10):
        for y in range(-10,10):
            for z in range(-10,10):
                for d in np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]]):
                    p, suc = findSurfacePoint(f,np.array([x,y,z]),d,2,20)
                    if 1 == suc:
                        pnts[i] = p
                        i += 1
    print(pnts)
    #return v


v = np.zeros((40000,3,3), dtype=np.dtype('f8'))
t0 = time.time()
render(f,v)
print(time.time() - t0)


