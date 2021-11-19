#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


import numpy as np
#import open3d as o3d
import time
from numba import jit, prange
from numba.typed import List

from stl import mesh


edgeRelCoordMapConst = ((0,-1,-1), (-1,0,-1), (-1,-1,0), (0,+1,+1), (+1,0,+1),
            (+1,+1,0), (+1,-1,0), (+1,0,-1), (0,+1,-1), (-1,+1,0), (-1,0,+1),
            (0,-1,+1))

@jit(nopython=True,cache=True)
def round(x):
    return np.floor(10000*x+0.5)/10000

# don't use np.arange, it is very slow (1 s startup time)
#@jit(nopython=True,cache=True)
def getInitPnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
    s0 = func(0,0,0)
    for d in range(resSteps):
        for xi in range(2**d):
            x = (xi+0.5)/(2**d)*(maxVal-minVal)+minVal
            for yi in range(2**d):
                y = (yi+0.5)/(2**d)*(maxVal-minVal)+minVal
                for zi in range(2**d):
                    z = (zi+0.5)/(2**d)*(maxVal-minVal)+minVal
                    s = func(x,y,z)
                    if s != s0:
                        return x,y,z,0,0,0
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




#@jit(nopython=True,cache=True)
def findSurfacePnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
    t0 = time.time()
    ps = getInitPnt(func, minVal, maxVal, resSteps)
    print('  getInitPnt time: {}'.format(time.time()-t0))
    print('  {}'.format(ps))
    t0 = time.time()
    p =  getSurfacePnt(func, (ps[0],ps[1],ps[2]), (ps[3],ps[4],ps[5]), resSteps)
    print('  getSurfacePnt time: {}'.format(time.time()-t0))
    print('  {}'.format(p))
    return p




@jit(nopython=True,cache=True)
def getSurface(func, startPnt, res=1.3):
    x,y,z = startPnt
    ptsList = List([(round(x-res/2), round(y-res/2), round(z-res/2))])
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
    cube2ptIdxList = List()
    ptCoordList = List(list(ptCoordDict.keys()))
    ptCoordDictRev = {e: i for i, e in enumerate(ptCoordList)}
    ptValueList = List(list(ptCoordDict.values()))
    for p in cubeCoordSet:
        x, y, z = p
        xh = round(x+r)
        yh = round(y+r)
        zh = round(z+r)
        cube2ptIdxList.append( ( ptCoordDictRev[(x,y,z)],
                                 ptCoordDictRev[(xh,y,z)],
                                 ptCoordDictRev[(x,yh,z)],
                                 ptCoordDictRev[(xh,yh,z)],
                                 ptCoordDictRev[(x,y,zh)],
                                 ptCoordDictRev[(xh,y,zh)],
                                 ptCoordDictRev[(x,yh,zh)],
                                 ptCoordDictRev[(xh,yh,zh)]   ) )

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
    edge2ptIdxList = List(list(cEdgesSet))
    edge2ptIdxDict = {e: i for i, e in enumerate(edge2ptIdxList)}

    cube2edgeIdxList = List()
    for cube in cube2ptIdxList:
        cube2edgeIdxList.append((   edge2ptIdxDict[(cube[0], cube[1])],
                                    edge2ptIdxDict[(cube[0], cube[2])],
                                    edge2ptIdxDict[(cube[0], cube[4])],
                                    edge2ptIdxDict[(cube[6], cube[7])],
                                    edge2ptIdxDict[(cube[5], cube[7])],
                                    edge2ptIdxDict[(cube[3], cube[7])],
                                    edge2ptIdxDict[(cube[1], cube[5])],
                                    edge2ptIdxDict[(cube[1], cube[3])],
                                    edge2ptIdxDict[(cube[2], cube[3])],
                                    edge2ptIdxDict[(cube[2], cube[6])],
                                    edge2ptIdxDict[(cube[4], cube[6])],
                                    edge2ptIdxDict[(cube[4], cube[5])] ))

    return (cube2ptIdxList, cube2edgeIdxList, edge2ptIdxList, ptCoordList,
            ptValueList)


@jit(nopython=True,cache=True)
def cutCedgeIdx(edge2ptIdxList, ptValueList):
    return [i for i, e in enumerate(edge2ptIdxList) if ptValueList[e[0]]
            != ptValueList[e[1]]]

@jit(nopython=True,cache=True)
def precTrPnts(func, cutCedgeIdxList, edge2ptIdxList, ptCoordList):
    pL = [edge2ptIdxList[e] for e in cutCedgeIdxList]
    pcL = [(ptCoordList[p[0]], ptCoordList[p[1]]) for p in pL]
    r = [getSurfacePnt(func, p[0], p[1]) for p in pcL]
    return r

@jit(nopython=True,cache=True)
def isOuterEdge(pt1, pt2):
    d = np.array(pt2) - np.array(pt1)
    return np.sum(np.abs(d)) == 2

@jit(nopython=True,cache=True)
def findOuterEdges(pts):
    return [(i,k+i+1) for i, p1 in enumerate(pts) for k, p2 in
            enumerate(pts[(i+1):]) if isOuterEdge(p1, p2)]

@jit(nopython=True,cache=True)
def cube2outerTrEdgeList(cube, cutCedgeIdxSet):
    cutEdges = [(i, e) for i, e in enumerate(cube) if e in cutCedgeIdxSet]
    cutEdgeCoordList = [edgeRelCoordMapConst[i[0]] for i in cutEdges]
    outerEdgeIdxList = findOuterEdges(List(cutEdgeCoordList))
    return List([(cutEdges[i[0]], cutEdges[i[1]]) for i in outerEdgeIdxList])


@jit(nopython=True,cache=True,parallel=True)
def findOuterTrEdges(cube2edgeIdxList, cutCedgeIdxList):
    r = List([cube2outerTrEdgeList(cube2edgeIdxList[0],
        cutCedgeIdxList)]*len(cube2edgeIdxList))
    cutCedgeIdxSet = set(cutCedgeIdxList)
    for i in prange(len(cube2edgeIdxList)):
        r[i] = cube2outerTrEdgeList(cube2edgeIdxList[i], cutCedgeIdxSet)
    return r


@jit(nopython=True,cache=True)
def isComplexCube(outerTrEdgesList):
    h = {}
    for f in outerTrEdgesList:
        for e in f:
            if e not in h:
                h[e] = 0
            h[e] += 1
            if h[e] > 2:
                return True
    return False


@jit(nopython=True,cache=True)
def findOuterCirc(outerTrEdgesList):
    oe = outerTrEdgesList
    x = oe
    c = []
    a,e = x.pop()
    c.append(e)
    while e!=a:
        y = [f for k in x for f in k]
        p = y.index(e)
        e = x.pop(int(p/2))[int((p+1)%2)]
        c.append(e)
    return c




@jit(nopython=True,cache=True)
def circIdx2trEdge(cube2outerTrEdgesList):
    circList = []
    for cube in cube2outerTrEdgesList:
        oe = [(e[0][1], e[1][1]) for e in cube]
        if isComplexCube(oe):
            continue
        circ = findOuterCirc(oe)
        circList.append(circ)
    return circList

#@jit(nopython=True,cache=True,parallel=True)
def trEdge2circ(circList, offset=0):
    r = {}
    for i, c in enumerate(circList):
        i += offset
        for k in range(len(c)):
            if (c[k], c[(k+1)%len(c)]) not in r:
                r[(c[k], c[(k+1)%len(c)])] = [(i, c)]
            else:
                r[(c[k], c[(k+1)%len(c)])].append((i, c))
            if (c[(k+1)%len(c)], c[k]) not in r:
                r[(c[(k+1)%len(c)], c[k])] = [(i, c[::-1])]
            else:
                r[(c[(k+1)%len(c)], c[k])].append((i, c[::-1]))
    return r

def repairComplexCircs(trEdge2circDict):
    singleEdgeSet = set()
    for k, v in trEdge2circDict.items():
        if len(v) != 2:
            singleEdgeSet.add(k)
    singleEdgeSet = {e if e[1] > e[0] else e[::-1] for e in singleEdgeSet}
    repairCircs = []
    singleEdgeList = List(singleEdgeSet)
    while len(singleEdgeList) > 0:
        repairCircs.append(findOuterCirc(singleEdgeList))
    return repairCircs


#@jit(nopython=True,cache=True,parallel=True)
def correctCircs(trEdge2circDict):
    x = trEdge2circDict
    circUsedSet = set()
    edgeOpList = [list(x.values())[0][0]]
    edgeResList = []
    while 0 < len(edgeOpList):
        ci, c = edgeOpList.pop()
        if ci in circUsedSet:
            continue
        edgeResList.append(c[::-1])
        circUsedSet.add(ci)
        for i in range(len(c)):
            e = (c[i], c[(i+1)%len(c)])
            edgeOpList.extend([s for s in x[e[::-1]] if s[0] != ci])
    return edgeResList


def extendTrEdge2circDict(x, y):
    for k, v in y.items():
        x[k].extend(v)
    return x

def calcCorCircList(cube2outerTrEdgesList):
    circList = circIdx2trEdge(cube2outerTrEdgesList)
    trEdge2circDict = trEdge2circ(circList)
    repairCircs = repairComplexCircs(trEdge2circDict)
    repairTrEdge2circDict = trEdge2circ(repairCircs, len(cube2outerTrEdgesList))
    extendTrEdge2circDict(trEdge2circDict, repairTrEdge2circDict)
    corCircList = correctCircs(trEdge2circDict)
    return corCircList


def findConvexness(func, corCircList):
    s = 0
    for i, circ in enumerate(corCircList):
        circ = np.array(circ)
        a = circ[1] - circ[0]
        b = circ[2] - circ[0]
        cr = np.cross(a, b)
        crn = cr / np.linalg.norm(cr)**0.5
        p = crn + circ[0]
        v = func(p[0], p[1], p[2])
        s += v
        if i > 20:
            break
    print(s)
    print(i+1)
    return s/(i+1) < 0.5



def calcTrianglesCor(corCircList, invertConvexness=False):
    circ = corCircList[0]
    trList = [(circ[0], circ[1], circ[2])]
    trList.pop()
    for circ in corCircList:
        n = len(circ)
        if invertConvexness:
            trInCubeList = [(circ[0], circ[i+1], circ[i+2]) for i in range(n-2)]
        else:
            trInCubeList = [(circ[0], circ[i+2], circ[i+1]) for i in range(n-2)]
        trList.extend(trInCubeList)
    return trList



def TrIdx2TrCoord(trList, cutCedgeIdxList, precTrPnts):
    cutCedgeIdxRevDict = {e: i for i, e in enumerate(cutCedgeIdxList)}
    return [[precTrPnts[cutCedgeIdxRevDict[f]] for f in e] for e in trList]


def renderAndSave(func, filename, res=1):
    t0 = time.time()
    p = findSurfacePnt(func)
    print('findSurfacePnt time: {}'.format(time.time()-t0))
    #print(p)
    t0 = time.time()
    cubesSet, ptsDict = getSurface(func, p, res)
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
    precTrPtsList = precTrPnts(func, List(cCeI), List(e2p), List(pc))
    print('precTrPnts time: {}'.format(time.time()-t0))
    print(len(precTrPtsList))

    t0 = time.time()
    cube2outerTrEdgesList = findOuterTrEdges(List(c2e), List(cCeI))
    print('findOuterTrEdges time: {}'.format(time.time()-t0))
    print(len(cube2outerTrEdgesList))



    t0 = time.time()
    corCircList = calcCorCircList(List(cube2outerTrEdgesList))
    print('calcCorCirc time: {}'.format(time.time()-t0))
    print(len(corCircList))

    t0 = time.time()
    circPtsCoordList = TrIdx2TrCoord(corCircList, cCeI, precTrPtsList)
    print('TrIdx2TrCoord time: {}'.format(time.time()-t0))
    print(len(circPtsCoordList))

    t0 = time.time()
    conv = findConvexness(func, circPtsCoordList)
    print('findConvexness time: {}'.format(time.time()-t0))
    print(conv)

    t0 = time.time()
    trPtsCoordList = calcTrianglesCor(circPtsCoordList, conv)
    print('calcTriangles time: {}'.format(time.time()-t0))
    print(len(trPtsCoordList))

    vertices = trPtsCoordList
    solid = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
    for i, v in enumerate(vertices):
        for j in range(3):
            solid.vectors[i][j] = vertices[i][j]
    solid.save(filename)




