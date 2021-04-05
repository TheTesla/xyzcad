import numpy as np
from stl import mesh
from numba import jit, prange
import time



# Create the mesh
cube = mesh.Mesh(np.zeros(6400000, dtype=mesh.Mesh.dtype))

@jit(nopython=True)
def f(x,y,z):
    r = 180
    return 1 if r**2 > ((2*x)**2 + y**2 + z**2) else 0

@jit(nopython=True,parallel=False)
def render(fun,v,xmin,xmax):
    #v = np.zeros((10000,3,3))
    i = 0
    for x in range(xmin,xmax):
        for y in range(-1000,1000):
            for z in range(-1000,1000):
                ft = f(x,y,z)
                if 0 != ft - f(x+1,y,z):
                    v[i][0] = [x+0.5,y-0.5,z-0.5]
                    v[i][1] = [x+0.5,y+0.5,z-0.5]
                    v[i][2] = [x+0.5,y-0.5,z+0.5]
                    i += 1
                    v[i][0] = [x+0.5,y+0.5,z+0.5]
                    v[i][1] = [x+0.5,y-0.5,z+0.5]
                    v[i][2] = [x+0.5,y+0.5,z-0.5]
                    i += 1
                if 0 != ft - f(x,y+1,z):
                    v[i][0] = [x-0.5,y+0.5,z-0.5]
                    v[i][1] = [x+0.5,y+0.5,z-0.5]
                    v[i][2] = [x-0.5,y+0.5,z+0.5]
                    i += 1
                    v[i][0] = [x+0.5,y+0.5,z+0.5]
                    v[i][1] = [x-0.5,y+0.5,z+0.5]
                    v[i][2] = [x+0.5,y+0.5,z-0.5]
                    i += 1
                if 0 != ft - f(x,y,z+1):
                    v[i][0] = [x-0.5,y-0.5,z+0.5]
                    v[i][1] = [x+0.5,y-0.5,z+0.5]
                    v[i][2] = [x-0.5,y+0.5,z+0.5]
                    i += 1
                    v[i][0] = [x+0.5,y+0.5,z+0.5]
                    v[i][1] = [x-0.5,y+0.5,z+0.5]
                    v[i][2] = [x+0.5,y-0.5,z+0.5]
                    i += 1
    return v


@jit(nopython=True,parallel=True)
def prender():
    xrange = 1000
    threads = 8
    v = np.zeros((threads,4000000,3,3), dtype=np.dtype('f8'))
    for j in prange(threads):
        render(f,v[j], -xrange+xrange/threads*2*j, -xrange+xrange/threads*2*(j+1))
    return v

t0 = time.time()
v = prender()
print(time.time() - t0)

#prender.parallel_diagnostics(level=4)

l = 0
for k in range(8):
    for i in range(800000):
        if 0.1 < np.sum(v[k,i,0]**2+v[k,i,1]**2+v[k,i,2]**2):
            for j in range(3):
                cube.vectors[l][j] = v[k,i,j]
            l += 1
#
## Write the mesh to file "cube.stl"
cube.save('sphere.stl')

