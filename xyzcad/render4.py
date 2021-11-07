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
from numba.typed import List

from stl import mesh


edgeRelCoordMapConst = [(0,-1,-1), (-1,0,-1), (-1,-1,0), (0,+1,+1), (+1,0,+1),
            (+1,+1,0), (+1,-1,0), (+1,0,-1), (0,+1,-1), (-1,+1,0), (-1,0,+1),
            (0,-1,+1)]

@jit(nopython=True,cache=True)
def round(x):
    return np.floor(10000*x+0.5)/10000

@jit(nopython=True,cache=True)
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

@jit(nopython=True,cache=True)
def getSurfacePnt(func, p0, p1, resSteps=24):
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    x0 = p0[0]
    y0 = p0[1]
    z0 = p0[2]

    s0 = func(x0,y0,z0)
    u = 0
    d = +1
    for i in range(resSteps):
        x = x0 * (1-u) + x1 * u
        y = y0 * (1-u) + y1 * u
        z = z0 * (1-u) + z1 * u
        s = func(x,y,z)
        if s != s0:
            s0 = s
            d = -d
        u += d*1/2**i
    return (x,y,z)




@jit(nopython=True,cache=True)
def findSurfacePnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
    ps = np.array(getInitPnt(func, minVal, maxVal, resSteps))
    return getSurfacePnt(func, (ps[0],ps[1],ps[2]), (ps[3],ps[4],ps[5]), resSteps)





@jit(nopython=True,cache=True)
def getSurface(func, startPnt=None, res=1.3):
    if startPnt is None:
        x,y,z = findSurfacePnt(func)
    else:
        x,y,z = startPnt
    ptsList = [(round(x), round(y), round(z))]
    cubeExistsSet = set()
    ptsResDict = dict()
    r = res
    while ptsList:
        x,y,z = ptsList.pop()
        xh = round(x+r)
        yh = round(y+r)
        zh = round(z+r)
        xl = round(x-r)
        yl = round(y-r)
        zl = round(z-r)
        v000 = func(x  , y  , z )
        v100 = func(xh , y  , z )
        v010 = func(x  , yh , z )
        v110 = func(xh , yh , z )
        v001 = func(x  , y  , zh)
        v101 = func(xh , y  , zh)
        v011 = func(x  , yh , zh)
        v111 = func(xh , yh , zh)
        s = v000 + v001 + v010 + v011 + v100 + v101 + v110 + v111
        if s == 8 or s == 0:
            continue
        if (xh,y,z) not in cubeExistsSet:
            ptsList.append((xh,y,z))
        if (x,yh,z) not in cubeExistsSet:
            ptsList.append((x,yh,z))
        if (x,y,zh) not in cubeExistsSet:
            ptsList.append((x,y,zh))
        if (xl,y,z) not in cubeExistsSet:
            ptsList.append((xl,y,z))
        if (x,yl,z) not in cubeExistsSet:
            ptsList.append((x,yl,z))
        if (x,y,zl) not in cubeExistsSet:
            ptsList.append((x,y,zl))
        cubeExistsSet.add((x,y,z))
        ptsResDict[(x,y,z)] = v000
        ptsResDict[(xh,y,z)] = v100
        ptsResDict[(x,yh,z)] = v010
        ptsResDict[(xh,yh,z)] = v110
        ptsResDict[(x,y,zh)] = v001
        ptsResDict[(xh,y,zh)] = v101
        ptsResDict[(x,yh,zh)] = v011
        ptsResDict[(xh,yh,zh)] = v111
    return cubeExistsSet, ptsResDict


@jit(nopython=True,cache=True)
def coords2relations(cubeCoordSet, ptCoordDict, res):
    r = res
    cube2ptIdxList = []
    ptCoordList = list(ptCoordDict.keys())
    ptValueList = list(ptCoordDict.values())
    for p in cubeCoordSet:
        x, y, z = p
        xh = round(x+r)
        yh = round(y+r)
        zh = round(z+r)
        cube2ptIdxList.append( (  ptCoordList.index((x,y,z)),
                                  ptCoordList.index((xh,y,z)),
                                  ptCoordList.index((x,yh,z)),
                                  ptCoordList.index((xh,yh,z)),
                                  ptCoordList.index((x,y,zh)),
                                  ptCoordList.index((xh,y,zh)),
                                  ptCoordList.index((x,yh,zh)),
                                  ptCoordList.index((xh,yh,zh))   ) )

    cEdgesSet = set()
    for cube in cube2ptIdxList:
        cEdgesSet.add((cube[0], cube[1]))
        cEdgesSet.add((cube[0], cube[2]))
        cEdgesSet.add((cube[0], cube[4]))
        cEdgesSet.add((cube[6], cube[7]))
        cEdgesSet.add((cube[5], cube[7]))
        cEdgesSet.add((cube[3], cube[7]))
        cEdgesSet.add((cube[1], cube[5]))
        cEdgesSet.add((cube[1], cube[3]))
        cEdgesSet.add((cube[2], cube[3]))
        cEdgesSet.add((cube[2], cube[6]))
        cEdgesSet.add((cube[4], cube[6]))
        cEdgesSet.add((cube[4], cube[5]))
    edge2ptIdxList = list(cEdgesSet)

    cube2edgeIdxList = []
    for cube in cube2ptIdxList:
        cube2edgeIdxList.append((   edge2ptIdxList.index((cube[0], cube[1])),
                                    edge2ptIdxList.index((cube[0], cube[2])),
                                    edge2ptIdxList.index((cube[0], cube[4])),
                                    edge2ptIdxList.index((cube[6], cube[7])),
                                    edge2ptIdxList.index((cube[5], cube[7])),
                                    edge2ptIdxList.index((cube[3], cube[7])),
                                    edge2ptIdxList.index((cube[1], cube[5])),
                                    edge2ptIdxList.index((cube[1], cube[3])),
                                    edge2ptIdxList.index((cube[2], cube[3])),
                                    edge2ptIdxList.index((cube[2], cube[6])),
                                    edge2ptIdxList.index((cube[4], cube[6])),
                                    edge2ptIdxList.index((cube[4], cube[5])) ))

    return (cube2ptIdxList, cube2edgeIdxList, edge2ptIdxList, ptCoordList,
            ptValueList)

def cutCedgeIdx(edge2ptIdxList, ptValueList):
    return [i for i, e in enumerate(edge2ptIdxList) if ptValueList[e[0]]
            != ptValueList[e[1]]]

#@jit(nopython=True,cache=True,parallel=True)
def precTrPnts(func, cutCedgeIdxList, edge2ptIdxList, ptCoordList):
    return [getSurfacePnt(func, ptCoordList[edge2ptIdxList[e][0]],
        ptCoordList[edge2ptIdxList[e][1]]) for e in cutCedgeIdxList]

def isInnerEdge2(pt1, pt2):
    d = np.array(pt2) - np.array(pt1)
    return np.sum(np.abs(d)) != 2

def findInnerEdges2(pts):
    return [(i,k+i+1) for i, p1 in enumerate(pts) for k, p2 in
            enumerate(pts[(i+1):]) if isInnerEdge(p1, p2)]

def findOuterEdges2(pts):
    return [(i,k+i+1) for i, p1 in enumerate(pts) for k, p2 in
            enumerate(pts[(i+1):]) if not isInnerEdge2(p1, p2)]

def cube2outerTrEdgeList(cube, cutCedgeIdxList):
    cutEdges = [(i, e) for i, e in enumerate(cube) if e in cutCedgeIdxList]
    cutEdgeCoordList = [edgeRelCoordMapConst[i[0]] for i in cutEdges]
    outerEdgeIdxList = findOuterEdges2(cutEdgeCoordList)
    return [(cutEdges[i[0]], cutEdges[i[1]]) for i in outerEdgeIdxList]

def findOuterTrEdges(cube2edgeIdxList, cutCedgeIdxList):
    r=[cube2outerTrEdgeList(cube, cutCedgeIdxList) for cube in cube2edgeIdxList]
    return r

def findOuterCirc2(outerTrEdgesList):
    oe = outerTrEdgesList
    oe = repairOuterCirc(oe)
    #print(oe)
    x = oe
    #print(x)
    c = []
    a,e = x.pop()
    c.append(e)
    while e!=a:
        y = [f for k in x for f in k]
        p = y.index(e)
        e = x.pop(int(p/2))[int((p+1)%2)]
        c.append(e)
    return c

def calcTriangles(cube2outerTrEdgesList):
    trList = []
    for cube in cube2outerTrEdgesList:
        oe = [(e[0][1], e[1][1]) for e in cube]
        circ = findOuterCirc2(oe)
        n = len(circ)
        trInCubeList = [(circ[0], circ[i+1], circ[i+2]) for i in range(n-2)]
        trList.extend(trInCubeList)
        #print(cube)
        #print(circ)
    return trList

def TrIdx2TrCoord(trList, cutCedgeIdxList, precTrPnts):
    #print(trList)
    cutCedgeIdxRevDict = {e: i for i, e in enumerate(cutCedgeIdxList)}
    #print(cutCedgeIdxRevDict)
    return [[precTrPnts[cutCedgeIdxRevDict[f]] for f in e] for e in trList]



def repairOuterCirc(oe):
    x = oe
    y = [f for k in x for f in k]
    bc = np.bincount(y)
    idx = np.where(bc==1)[0]
    if len(idx) == 2:
        print('repair!')
        print(oe)
        oe.append((idx[0], idx[1]))
    return oe






def renderAndSave(func, filename, res=1):
    t0 = time.time()
    cubesSet, ptsDict = getSurface(func, None, res)
    print('getSurface time: {}'.format(time.time()-t0))
    print(len(ptsDict))
    t0 = time.time()
    c2p, c2e, e2p, pc, pv = coords2relations(List(cubesSet), ptsDict, res)
    print('coords2relations time: {}'.format(time.time()-t0))
    print('{} - {} - {} - {} - {}'.format(len(c2p), len(c2e), len(e2p),
        len(pc), len(pv)))
    t0 = time.time()
    cCeI = cutCedgeIdx(e2p, pv)
    print('cutCedgeIdx time: {}'.format(time.time()-t0))
    print(len(cCeI))
    t0 = time.time()
    precTrPtsList = precTrPnts(func, cCeI, e2p, pc)
    print('precTrPnts time: {}'.format(time.time()-t0))
    print(len(precTrPtsList))

    t0 = time.time()
    cube2outerTrEdgesList = findOuterTrEdges(c2e, cCeI)
    print('findOuterTrEdges time: {}'.format(time.time()-t0))
    print(len(cube2outerTrEdgesList))
    t0 = time.time()
    triangleList = calcTriangles(cube2outerTrEdgesList)
    print('calcTriangles time: {}'.format(time.time()-t0))
    print(len(triangleList))
    t0 = time.time()
    trPtsCoordList = TrIdx2TrCoord(triangleList, cCeI, precTrPtsList)
    print('TrIdx2TrCoord time: {}'.format(time.time()-t0))
    print(len(trPtsCoordList))



    vertices = trPtsCoordList
    solid = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
    #solid = mesh.Mesh(np.zeros(10, dtype=mesh.Mesh.dtype))
    for i, v in enumerate(vertices):
        #print('{} {} {}'.format(vertices[i][0], vertices[i][1], vertices[i][2]))
        #if i == 10:
        #    break
        for j in range(3):
            solid.vectors[i][j] = vertices[i][j]
    solid.save(filename)

#    pcd = o3d.geometry.PointCloud()
#    pcd.points = o3d.utility.Vector3dVector(ptsResArray[:,:3])
#    n = len(ptsResArray)
#    pcd.colors = o3d.utility.Vector3dVector(np.array(n*[np.array([0,0,1])]))
#    pcd.normals = o3d.utility.Vector3dVector(ptsResArray[:,3:])
#    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#            pcd, depth=8, width=0, scale=1.2, linear_fit=False)[0]
#    mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
#    bbox = pcd.get_axis_aligned_bounding_box()
#    meshcrp = mesh.crop(bbox)
#    o3d.io.write_triangle_mesh(filename, meshcrp)



