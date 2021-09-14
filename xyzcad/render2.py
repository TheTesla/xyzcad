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
def getSurface(func, startPnt=None, res=1.3):
    if startPnt is None:
        startPnt = findSurfacePnt(func)
    ptsList = [(np.floor(1000*startPnt[0]+0.5)/1000,
                np.floor(1000*startPnt[1]+0.5)/1000,
                np.floor(1000*startPnt[2]+0.5)/1000) ]
    x,y,z = ptsList[0]
    ptsSet = set()
    ptsSet.add((x,y,z))
    ptsResList = []
    r = res

    while ptsList:
        x,y,z = ptsList.pop()

        xl = np.floor(1000*(x-r)+0.5)/1000
        xh = np.floor(1000*(x+r)+0.5)/1000
        yl = np.floor(1000*(y-r)+0.5)/1000
        yh = np.floor(1000*(y+r)+0.5)/1000
        zl = np.floor(1000*(z-r)+0.5)/1000
        zh = np.floor(1000*(z+r)+0.5)/1000

        xu = func(xl,y,z)
        xo = func(xh,y,z)
        yu = func(x,yl,z)
        yo = func(x,yh,z)
        zu = func(x,y,zl)
        zo = func(x,y,zh)

        s = xu + xo + yu + yo + zu + zo
        if s == 6 or s == 0:
            continue

        ptsResList.append((x,y,z,xu-xo,yu-yo,zu-zo))

        if s == 5:
            if (xl,y,z) not in ptsSet:
                if xo:
                    ptsSet.add((xl,y,z))
                    ptsList.append((xl,y,z))
            if (xh,y,z) not in ptsSet:
                if xu:
                    ptsSet.add((xh,y,z))
                    ptsList.append((xh,y,z))
            if (x,yl,z) not in ptsSet:
                if yo:
                    ptsSet.add((x,yl,z))
                    ptsList.append((x,yl,z))
            if (x,yh,z) not in ptsSet:
                if yu:
                    ptsSet.add((x,yh,z))
                    ptsList.append((x,yh,z))
            if (x,y,zl) not in ptsSet:
                if zo:
                    ptsSet.add((x,y,zl))
                    ptsList.append((x,y,zl))
            if (x,y,zh) not in ptsSet:
                if zu:
                    ptsSet.add((x,y,zh))
                    ptsList.append((x,y,zh))
        elif s == 2:
            if (xl,y,z) not in ptsSet:
                if not xo:
                    ptsSet.add((xl,y,z))
                    ptsList.append((xl,y,z))
            if (xh,y,z) not in ptsSet:
                if not xu:
                    ptsSet.add((xh,y,z))
                    ptsList.append((xh,y,z))
            if (x,yl,z) not in ptsSet:
                if not yo:
                    ptsSet.add((x,yl,z))
                    ptsList.append((x,yl,z))
            if (x,yh,z) not in ptsSet:
                if not yu:
                    ptsSet.add((x,yh,z))
                    ptsList.append((x,yh,z))
            if (x,y,zl) not in ptsSet:
                if not zo:
                    ptsSet.add((x,y,zl))
                    ptsList.append((x,y,zl))
            if (x,y,zh) not in ptsSet:
                if not zu:
                    ptsSet.add((x,y,zh))
                    ptsList.append((x,y,zh))
        else:
            if (xl,y,z) not in ptsSet:
                ptsSet.add((xl,y,z))
                ptsList.append((xl,y,z))
            if (xh,y,z) not in ptsSet:
                ptsSet.add((xh,y,z))
                ptsList.append((xh,y,z))
            if (x,yl,z) not in ptsSet:
                ptsSet.add((x,yl,z))
                ptsList.append((x,yl,z))
            if (x,yh,z) not in ptsSet:
                ptsSet.add((x,yh,z))
                ptsList.append((x,yh,z))
            if (x,y,zl) not in ptsSet:
                ptsSet.add((x,y,zl))
                ptsList.append((x,y,zl))
            if (x,y,zh) not in ptsSet:
                ptsSet.add((x,y,zh))
                ptsList.append((x,y,zh))

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




