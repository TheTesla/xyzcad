#/usr/bin/env python3

import numpy as np
import open3d as o3d
from numba import jit


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


def getSurface(func, startPnt=None, res=1.3, maxIter=10000000):
    if startPnt is None:
        startPnt = findSurfacePnt(func)
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

    ptsResArray = np.array([np.array([x,y,z,n[0],n[1],n[2]])
                            for x in ptsResDict.keys()
                            for y in ptsResDict[x].keys()
                            for z, n in ptsResDict[x][y].items()])
    return ptsResArray


def renderAndSave(func, filename, res=1):
    ptsResArray = getSurface(func, None, res)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptsResArray[:,:3])
    n = len(ptsResArray)
    pcd.colors = o3d.utility.Vector3dVector(np.array(n*[np.array([0,0,1])]))
    pcd.normals = o3d.utility.Vector3dVector(ptsResArray[:,3:])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.2, linear_fit=False)[0]
    mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
    bbox = pcd.get_axis_aligned_bounding_box()
    meshcrp = mesh.crop(bbox)
    o3d.io.write_triangle_mesh(filename, meshcrp)




