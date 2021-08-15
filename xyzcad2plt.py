import numpy as np
import open3d as o3d
from stl import mesh
from numba import njit, jit, prange
from numba.typed import Dict
from numba.core import types


import time

import matplotlib.pyplot as plt





@jit(nopython=True)
def f(x,y,z):
    r = 3
    #return 1 if r**2 > ((1.+(x+3)/10)*(x+3)**2 + y**2 + z**2) else 0
    return 1 if r**2 > ((x-1)**2 + (y+13)**2 + z**2) else 0


#@jit(nopython=True)
def findSurfacePoint(fun, startPnt, direction, r, res):
    t = 0
    d = 1
    suc = 0
    dr = 0
    normDir = direction/np.sum(direction**2)
    for i in range(res):
        step = r/(2.**i)
        p = startPnt + t*normDir
        s = fun(p[0],p[1],p[2])
        if i == 0:
            so = s
        if 0 != s - so:
            dr = d * np.sign(s - so)
            suc = 1
            d *= -1
        t += step*d
        so = s
    return p, suc, dr

@jit(nopython=True)
def findNorm(fun, startPnt, d, r, res):
    ps = np.zeros((10,3))
    cr = np.zeros((10,3))
    i = 0
    for delta in np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]):
        p, suc, dr = findSurfacePoint(fun,startPnt+300000*r*delta/(2**res),d,r,res)
        if 1 == suc:
            ps[i] = p
            i += 1
    k = 0
    if i > 3:
        for a in range(i):
            for b in range(a):
                for c in range(b):
                    cr[k] = np.cross(-ps[a]+ps[b],-ps[a]+ps[c])
                    k += 1

        return np.sum(cr,axis=0), 1
    return np.zeros(3), 0

@jit(nopython=True)
def getPoints(fun,pnts,nrms):
    i = 0
    suc = 0
    dr = 0
    for x in np.arange(-10,10,0.5):
        for y in np.arange(-10,10,0.5):
            for z in np.arange(-10,10,0.5):
                for d in np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]]):
                    p, suc, dr = findSurfacePoint(f,np.array([x,y,z]),0.5*d,2,20)
                    if 1 == suc:
                        nrm, t = findNorm(f,np.array([x,y,z]),d,2,20)
                        if 1 == t:
                            nrms[i] = dr*nrm
                            pnts[i] = p
                            i += 1
    return i




@jit(nopython=True)
def getInitPnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
    s0 = func(0,0,0)
    xOld = 0
    yOld = 0
    zOld = 0
    for d in np.arange(resSteps):
        print(d)
        #iterVals = np.arange((maxVal-minVal)/(2**(d+1))-minVal,maxVal,(maxVal-minVal)/(2**d))
        iterVals = np.arange(1/(2**(d+1)),1,1/(2**d))
        iterVals = iterVals*(maxVal-minVal)+minVal
        print(iterVals)
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
        print(i)
        print(d)
        p = p0 * (1-u) + p1 * u
        print(p)
        s = func(p[0],p[1],p[2])
        if s != s0:
            s0 = s
            d = -d
        u += d*1/2**i
    return p




def findSurfacePnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
    ps = np.array(getInitPnt(func, minVal, maxVal, resSteps))
    return getSurfacePnt(func, ps[:3], ps[3:], resSteps)


p = np.zeros((100000,3), dtype=np.dtype('f8'))
nrm = np.zeros((100000,3), dtype=np.dtype('f8'))
t0 = time.time()

#sp = findSurfacePoint(f, np.array([-3,0,0]), np.array([1,0,0]),3, 1000)


#@jit(nopython=True)
#@njit
def getSurface(func, startPnt, res=0.1, maxIter=1000000):
    ptsList = [startPnt]
    ptsResDict = dict()
    r = res
    f = lambda a: func(a[0],a[1],a[2])
    for i in range(maxIter):
        if len(ptsList) == 0:
            break
        p = ptsList.pop()
        p = np.floor(1000*p+0.5)/1000
        xu = 1 if f(p+np.array([-r,0,0])) else 0
        xo = 1 if f(p+np.array([+r,0,0])) else 0
        yu = 1 if f(p+np.array([0,-r,0])) else 0
        yo = 1 if f(p+np.array([0,+r,0])) else 0
        zu = 1 if f(p+np.array([0,0,-r])) else 0
        zo = 1 if f(p+np.array([0,0,+r])) else 0
        s = xu + xo + yu + yo + zu + zo
        #s = int(sum([xu, xo, yu, yo, zu, zo]))
        if s == 6 or s == 0:
            continue
        if p[0] not in ptsResDict:
            ptsResDict[p[0]] = dict()
        if p[1] not in ptsResDict[p[0]]:
            ptsResDict[p[0]][p[1]] = dict()
        if p[2] not in ptsResDict[p[0]][p[1]]:
            ptsResDict[p[0]][p[1]][p[2]] = np.array([xu - xo, yu - yo, zu - zo])
        else:
            continue
        n = np.array([0,0,0])
        ptsList.append(p+np.array([-r,0,0]))
        ptsList.append(p+np.array([+r,0,0]))
        ptsList.append(p+np.array([0,-r,0]))
        ptsList.append(p+np.array([0,+r,0]))
        ptsList.append(p+np.array([0,0,-r]))
        ptsList.append(p+np.array([0,0,+r]))

    ptsResArray = np.array([np.array([x,y,z,n[0],n[1],n[2]]) for x in ptsResDict.keys() for y in
        ptsResDict[x].keys() for z, n in
            ptsResDict[x][y].items()])
    return ptsResArray

sp = findSurfacePnt(f)
print(sp)

ptsResArray = getSurface(f, sp)


print(time.time() - t0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ptsResArray[:,:3])
n = len(ptsResArray)
pcd.colors = o3d.utility.Vector3dVector(np.array(n*[np.array([1,0,0])]))
pcd.normals = o3d.utility.Vector3dVector(ptsResArray[:,3:])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.2, linear_fit=False)[0]
mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
bbox = pcd.get_axis_aligned_bounding_box()
meshcrp = mesh.crop(bbox)

o3d.io.write_triangle_mesh("sphere3.stl", meshcrp)

#ax = plt.axes(projection='3d')
#ax.scatter(ptsResArray[:,0], ptsResArray[:,1], ptsResArray[:,2], s=0.1)
#plt.show()




