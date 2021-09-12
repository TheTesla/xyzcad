#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


import numpy as np
import open3d as o3d
import time
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
    for i in np.arange(resSteps):
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





@jit(nopython=True)
def getSurface(func, startPnt=None, res=1.3, maxIter=100000000):
    if startPnt is None:
        startPnt = findSurfacePnt(func)
    ptsList = [(startPnt[0],startPnt[1],startPnt[2])]
    ptsSet = set()
    ptsResList = []
    r = res

    for i in np.arange(maxIter):
        if len(ptsList) == 0:
            break
        p = ptsList.pop()
        pl = [np.floor(1000*e+0.5)/1000 for e in p]
        x,y,z = pl[0],pl[1],pl[2]
        if (x,y,z) in ptsSet:
            continue
        ptsSet.add((x,y,z))

        xu = func(x-r,y,z)
        xo = func(x+r,y,z)
        yu = func(x,y-r,z)
        yo = func(x,y+r,z)
        zu = func(x,y,z-r)
        zo = func(x,y,z+r)

        s = xu + xo + yu + yo + zu + zo
        if s == 6 or s == 0:
            continue
        ptsResList.append((x,y,z,xu-xo,yu-yo,zu-zo))

        ptsList.append((x-r,y,z))
        ptsList.append((x+r,y,z))
        ptsList.append((x,y-r,z))
        ptsList.append((x,y+r,z))
        ptsList.append((x,y,z-r))
        ptsList.append((x,y,z+r))

    ptsResArray = np.array(ptsResList)
    return ptsResArray


def renderAndSave(func, filename, res=1):
    t0 = time.time()
    ptsResArray = getSurface(func, None, res)
    print('getSurface time: {}'.format(time.time()-t0))
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




