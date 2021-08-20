import numpy as np
import open3d as o3d
from stl import mesh
from numba import njit, jit, prange
from numba.typed import Dict
from numba.core import types
import struct


import time

import matplotlib.pyplot as plt



@jit(nopython=True)
def f2c(x):
    s = ''
    x *= 10000
    x = int(x)
    for i in range(10):
        s = chr(int(48+x%10)) + s
        x /=10
    return s

@jit(nopython=True)
def c2f(s):
    x = 0
    for i in range(10):
        x += 10**i * (ord(s[9-i])-48)
    return x/10000


@jit(nopython=True)
def f(x,y,z):
    r = 3
    #return 1 if r**2 > ((1.+(x+3)/10)*(x+3)**2 + y**2 + z**2) else 0
    return 1 if r**2 > ((x-1)**2 + (y+13)**2 + z**2) else 0




@jit(nopython=True)
def getInitPnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
    s0 = func(0,0,0)
    xOld = 0
    yOld = 0
    zOld = 0
    for d in np.arange(resSteps):
        iterVals = np.arange(1/(2**(d+1)),1,1/(2**d))
        iterVals = iterVals*(maxVal-minVal)+minVal
        for x in iterVals:
            for y in iterVals:
                for z in iterVals:
                    s = func(x,y,z)
                    if s != s0:
                        return x,y,z,xOld,yOld,zOld
                    xOld = x
                    yOld = y
                    zOld = z
    return 0,0,0,0,0,0

@jit(nopython=True)
def getSurfacePnt(func, p0, p1, resSteps=24):
    s0 = func(p0[0],p0[1],p0[2])
    u = 0
    d = +1
    for i in range(resSteps):
        p = p0 * (1-u) + p1 * u
        s = func(p[0],p[1],p[2])
        if s != s0:
            s0 = s
            d = -d
        u += d*1/2**i
    return p




@jit(nopython=True)
def findSurfacePnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
    ps = np.array(getInitPnt(func, minVal, maxVal, resSteps))
    return getSurfacePnt(func, ps[:3], ps[3:], resSteps)


t0 = time.time()




#@njit
@jit(nopython=True)
def getSurface(func, startPnt, res=0.1, maxIter=1000000):
    ptsList = [startPnt]
    ptsResDict = dict()
    ptsResArray = np.zeros((20000,6))
    r = res
    f = lambda a: func(a[0],a[1],a[2])
    j = 0
    for i in range(maxIter):
        if len(ptsList) == 0:
            break
        p = ptsList.pop()
        p = np.floor(1000*p+0.5)/1000
        xu = 1.0 if f(p+np.array([-r,0,0])) else 0.0
        xo = 1.0 if f(p+np.array([+r,0,0])) else 0.0
        yu = 1.0 if f(p+np.array([0,-r,0])) else 0.0
        yo = 1.0 if f(p+np.array([0,+r,0])) else 0.0
        zu = 1.0 if f(p+np.array([0,0,-r])) else 0.0
        zo = 1.0 if f(p+np.array([0,0,+r])) else 0.0
        s = xu + xo + yu + yo + zu + zo
        if s == 6 or s == 0:
            continue
        k = f2c(p[0]) + f2c(p[1]) + f2c(p[2])
        if k not in ptsResDict:
            ptsResDict[k] = True #np.array([xu - xo, yu - yo, zu - zo])
            ptsResArray[j] = np.array([p[0],p[1],p[2],xu-xo,yu-yo,zu-zo])
            j += 1
        else:
            continue
        n = np.array([0,0,0])
        ptsList.append(p+np.array([-r,0,0]))
        ptsList.append(p+np.array([+r,0,0]))
        ptsList.append(p+np.array([0,-r,0]))
        ptsList.append(p+np.array([0,+r,0]))
        ptsList.append(p+np.array([0,0,-r]))
        ptsList.append(p+np.array([0,0,+r]))

    return ptsResArray, j

sp = findSurfacePnt(f)
print(sp)

ptsResArray, i = getSurface(f, sp)


print(time.time() - t0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ptsResArray[:i,:3])
n = len(ptsResArray)
print(n)
pcd.colors = o3d.utility.Vector3dVector(np.array(n*[np.array([0,0,1])]))
pcd.normals = o3d.utility.Vector3dVector(ptsResArray[:i,3:])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.2, linear_fit=False)[0]
mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
bbox = pcd.get_axis_aligned_bounding_box()
meshcrp = mesh.crop(bbox)

o3d.io.write_triangle_mesh("sphere3.stl", meshcrp)

#ax = plt.axes(projection='3d')
#ax.scatter(ptsResArray[:,0], ptsResArray[:,1], ptsResArray[:,2], s=0.1)
#plt.show()




