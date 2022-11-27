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
from numba import jit, prange, types
from numba.typed import List, Dict

from stl import mesh


edgeRelCoordMapConst = ((0,-1,-1), (-1,0,-1), (-1,-1,0), (0,+1,+1), (+1,0,+1),
            (+1,+1,0), (+1,-1,0), (+1,0,-1), (0,+1,-1), (-1,+1,0), (-1,0,+1),
            (0,-1,+1))

@jit(nopython=True,cache=True)
def round(x):
    return np.floor(10000.*x+0.5)/10000.


@jit(nopython=True,cache=True)
def getInitPnt(func, minVal=-1000., maxVal=+1000., resSteps=24):
    s0 = func(0.,0.,0.)
    for d in range(resSteps):
        for xi in range(2**d):
            x = (xi+0.5)/(2**d)*(maxVal-minVal)+minVal
            for yi in range(2**d):
                y = (yi+0.5)/(2**d)*(maxVal-minVal)+minVal
                for zi in range(2**d):
                    z = (zi+0.5)/(2**d)*(maxVal-minVal)+minVal
                    s = func(x,y,z)
                    if s != s0:
                        return x,y,z,0.,0.,0.
    return 0.,0.,0.,0.,0.,0.



#@jit(nopython=True,cache=True,parallel=True)
#def initPntZ(func, s0, x, y, d, minVal, maxVal):
#    si = np.zeros((2**d),dtype=np.dtype('bool'))
#    for zi in prange(2**d):
#        z = (zi+0.5)/(2**d)*(maxVal-minVal)+minVal
#        s = func(x,y,z)
#        si[zi] = s != s0
#    return si
#
#
## don't use np.arange, it is very slow (1 s startup time)
#@jit(nopython=True,cache=True)
##@jit(nopython=True,cache=True)
#def getInitPnt(func, minVal=-1000, maxVal=+1000, resSteps=24):
#    s0 = func(0,0,0)
#    for d in range(resSteps):
#        for xi in range(2**d):
#            x = (xi+0.5)/(2**d)*(maxVal-minVal)+minVal
#            for yi in range(2**d):
#                y = (yi+0.5)/(2**d)*(maxVal-minVal)+minVal
#                si = initPntZ(func, s0, x, y, d, minVal, maxVal)
##                si = np.zeros((2**d),dtype=np.dtype('bool'))
##                for zi in prange(2**d):
##                    z = (zi+0.5)/(2**d)*(maxVal-minVal)+minVal
##                    s = func(x,y,z)
##                    si[zi] = s != s0
#                zia = np.where(si)[0]
#                if len(zia) > 0:
#                    zi = zia[0]
#                    z = (zi+0.5)/(2**d)*(maxVal-minVal)+minVal
#                    return x,y,z,0.,0.,0.
#
#
#                    #if s != s0:
#                    #    return x,y,z,0,0,0
#    return 0.,0.,0.,0.,0.,0.


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
def findSurfacePnt(func, minVal=-1000., maxVal=+1000., resSteps=24):
    #t0 = time.time()
    ps = getInitPnt(func, minVal, maxVal, resSteps)
    #print('  getInitPnt time: {}'.format(time.time()-t0))
    #print('  {}'.format(ps))
    #t0 = time.time()
    p =  getSurfacePnt(func, (ps[0],ps[1],ps[2]), (ps[3],ps[4],ps[5]), resSteps)
    #print('  getSurfacePnt time: {}'.format(time.time()-t0))
    #print('  {}'.format(p))
    return p




@jit(nopython=True,cache=True)
def getSurface(func, startPnt, res=1.3):
    x,y,z = startPnt
    ptsList = List([(round(x-res/2), round(y-res/2), round(z-res/2))])
    cubeExistsSet = set()
    ptsResDict = Dict()
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
    cubesArray = np.asarray(list(cubeExistsSet))
    ptCoordDictKeys = np.asarray(list(ptsResDict.keys()))
    ptCoordDictVals = np.asarray(list(ptsResDict.values()))
    return cubesArray, ptCoordDictKeys, ptCoordDictVals #ptsResDict


@jit(nopython=True,cache=True)
def tuple2int64(x0, x1):
    return x1*2**32 + x0

@jit(nopython=True,cache=True,parallel=True)
def coords2relations(cubeCoordArray, ptCoordArray, ptValueArray, res):
    r = res

    ptCoordDictRev = {(e[0], e[1], e[2]): i for i, e in enumerate(ptCoordArray)}

    cube2ptIdxArray = np.zeros((cubeCoordArray.shape[0],8),dtype='int')
    for i in prange(cubeCoordArray.shape[0]):
        p = cubeCoordArray[i]
        x, y, z = p
        xh = round(x+r)
        yh = round(y+r)
        zh = round(z+r)
        cube2ptIdxArray[i] = [  ptCoordDictRev[(x, y, z )],
                                ptCoordDictRev[(xh,y, z )],
                                ptCoordDictRev[(x, yh,z )],
                                ptCoordDictRev[(xh,yh,z )],
                                ptCoordDictRev[(x, y, zh)],
                                ptCoordDictRev[(xh,y, zh)],
                                ptCoordDictRev[(x, yh,zh)],
                                ptCoordDictRev[(xh,yh,zh)]]

    cEdgeArray = np.zeros((cube2ptIdxArray.shape[0]*12,2),dtype='int')
    for i in prange(cube2ptIdxArray.shape[0]):
        cube = cube2ptIdxArray[i]
        cEdgeArray[12*i+ 0] = (cube[0], cube[1])
        cEdgeArray[12*i+ 1] = (cube[0], cube[2])
        cEdgeArray[12*i+ 2] = (cube[0], cube[4])
        cEdgeArray[12*i+ 3] = (cube[6], cube[7])
        cEdgeArray[12*i+ 4] = (cube[5], cube[7])
        cEdgeArray[12*i+ 5] = (cube[3], cube[7])
        cEdgeArray[12*i+ 6] = (cube[1], cube[5])
        cEdgeArray[12*i+ 7] = (cube[1], cube[3])
        cEdgeArray[12*i+ 8] = (cube[2], cube[3])
        cEdgeArray[12*i+ 9] = (cube[2], cube[6])
        cEdgeArray[12*i+10] = (cube[4], cube[6])
        cEdgeArray[12*i+11] = (cube[4], cube[5])
    print(cEdgeArray.shape)
    cEdgesSet = set([(e[0], e[1]) for e in cEdgeArray])

    print(len(cEdgesSet))
    edge2ptIdxArray = np.asarray(list(cEdgesSet))

    edge2ptIdxDict = {(e[0], e[1]): i for i, e in enumerate(edge2ptIdxArray)}

    cube2edgeIdxArray = np.zeros((cube2ptIdxArray.shape[0],12),dtype='int')

    for i in prange(len(cube2ptIdxArray)):
        cube = cube2ptIdxArray[i]
        cube2edgeIdxArray[i] = [edge2ptIdxDict[(cube[0], cube[1])],
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
                                edge2ptIdxDict[(cube[4], cube[5])]]

    return (cube2ptIdxArray, cube2edgeIdxArray, edge2ptIdxArray, ptCoordArray,
            ptValueArray)


@jit(nopython=True,cache=True)
def cutCedgeIdx(edge2ptIdxList, ptValueList):
    return np.asarray([i for i, e in enumerate(edge2ptIdxList) if ptValueList[e[0]]
            != ptValueList[e[1]]])

@jit(nopython=True,cache=True,parallel=True)
def precTrPnts(func, cutCedgeIdxArray, edge2ptIdxArray, ptCoordArray):
    lcceil = len(cutCedgeIdxArray)
    r = np.zeros((lcceil,3))
    for i in prange(lcceil):
        p0, p1 = edge2ptIdxArray[cutCedgeIdxArray[i]]
        r[i] = getSurfacePnt(func, ptCoordArray[p0], ptCoordArray[p1])
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
    c = List()
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
    circList = List()
    for cube in cube2outerTrEdgesList:
        oe = [(e[0][1], e[1][1]) for e in cube]
        if isComplexCube(oe):
            continue
        circ = findOuterCirc(oe)
        circList.append(circ)
    return circList

@jit(nopython=True,cache=True)
def trEdge2circ(circList, offset=0):
    r = Dict()
    for i, c in enumerate(circList):
        i += offset
        for k in range(len(c)):
            if (c[k], c[(k+1)%len(c)]) not in r:
                r[(c[k], c[(k+1)%len(c)])] = List([(i, c)])
            else:
                r[(c[k], c[(k+1)%len(c)])].append((i, c))
            if (c[(k+1)%len(c)], c[k]) not in r:
                r[(c[(k+1)%len(c)], c[k])] = List([(i, c[::-1])])
            else:
                r[(c[(k+1)%len(c)], c[k])].append((i, c[::-1]))
    return r

@jit(nopython=True,cache=True)
def repairComplexCircs(trEdge2circDict):
    singleEdgeSet = set()
    for k, v in trEdge2circDict.items():
        if len(v) != 2:
            singleEdgeSet.add(k)
    singleEdgeSet = set([e if e[1] > e[0] else e[::-1] for e in singleEdgeSet])
    repairCircs = []
    singleEdgeList = List(singleEdgeSet)
    while len(singleEdgeList) > 0:
        repairCircs.append(findOuterCirc(singleEdgeList))
    return List(repairCircs)


@jit(nopython=True,cache=True)
def correctCircs(trEdge2circDict):
    x = trEdge2circDict
    circUsedSet = set()
    edgeOpList = List([list(x.values())[0][0]])
    edgeResList = List()
    while 0 < len(edgeOpList):
        ci, c = edgeOpList.pop()
        if ci in circUsedSet:
            continue
        edgeResList.append(c[::-1])
        circUsedSet.add(ci)
        for i in range(len(c)):
            e = (c[i], c[(i+1)%len(c)])
            edgeOpList.extend([s for s in x[e[::-1]] if s[0] != ci])
    return List(edgeResList)


@jit(nopython=True,cache=True)
def extendTrEdge2circDict(x, y):
    for k, v in y.items():
        x[k].extend(v)
    return x

#@jit(nopython=True,cache=True)
def calcCorCircList(cube2outerTrEdgesList):
    t0 = time.time()
    circList = circIdx2trEdge(cube2outerTrEdgesList)
    print('  circIdx2trEdge time: {}'.format(time.time()-t0))
    t0 = time.time()
    trEdge2circDict = trEdge2circ(circList)
    print('  trEdge2circ time: {}'.format(time.time()-t0))
    t0 = time.time()
    repairCircs = repairComplexCircs(trEdge2circDict)
    print('  repairComplexCircs time: {}'.format(time.time()-t0))
    print(len(repairCircs))
    t0 = time.time()
    repairTrEdge2circDict = trEdge2circ(repairCircs, len(cube2outerTrEdgesList))
    print('  trEdge2circ time: {}'.format(time.time()-t0))
    t0 = time.time()
    extendTrEdge2circDict(trEdge2circDict, repairTrEdge2circDict)
    print('  extendTrEdge2circDict time: {}'.format(time.time()-t0))
    t0 = time.time()
    corCircList = correctCircs(trEdge2circDict)
    print('  correctCircs time: {}'.format(time.time()-t0))
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



@jit(nopython=True,cache=True)
def calcTrianglesCor(corCircList, invertConvexness=False):
    trList = List()
    if invertConvexness:
        for circ in corCircList:
            n = len(circ)
            trInCubeList = [(circ[0], circ[i+1], circ[i+2]) for i in range(n-2)]
            trList.extend(trInCubeList)
    else:
        for circ in corCircList:
            n = len(circ)
            trInCubeList = [(circ[0], circ[i+2], circ[i+1]) for i in range(n-2)]
            trList.extend(trInCubeList)
    return np.asarray([[[p[0],p[1],p[2]] for p in c] for c in trList])



@jit(nopython=True,cache=True)
def TrIdx2TrCoord(trList, cutCedgeIdxList, precTrPnts):
    cutCedgeIdxRevDict = {e: i for i, e in enumerate(cutCedgeIdxList)}
    return List([[precTrPnts[cutCedgeIdxRevDict[f]] for f in e] for e in
        trList])



def renderAndSave(func, filename, res=1):
    t0 = time.time()
    p = findSurfacePnt(func)
    print('findSurfacePnt time: {}'.format(time.time()-t0))
    #print(p)
    t0 = time.time()
    cubesArray, ptsKeys, ptsVals = getSurface(func, p, res)
    print('getSurface time: {}'.format(time.time()-t0))
    print(len(ptsKeys))
    t0 = time.time()
    c2p, c2e, e2p, pc, pv = coords2relations(cubesArray, ptsKeys, ptsVals, res)
    print('coords2relations time: {}'.format(time.time()-t0))
    print('{} - {} - {} - {} - {}'.format(len(c2p), len(c2e), len(e2p),
        len(pc), len(pv)))
    t0 = time.time()
    cCeI = cutCedgeIdx(e2p, pv)
    print('cutCedgeIdx time: {}'.format(time.time()-t0))
    print(len(cCeI))
    t1 = time.time()

    t0 = time.time()
    lcceil = len(cCeI)



    precTrPtsList = precTrPnts(func, cCeI, e2p, pc)
    print('precTrPnts time: {}'.format(time.time()-t0))
    print(len(precTrPtsList))

    t0 = time.time()
    cube2outerTrEdgesList = findOuterTrEdges(c2e, cCeI)
    print('findOuterTrEdges time: {}'.format(time.time()-t0))
    print(len(cube2outerTrEdgesList))



    t0 = time.time()
    corCircList = calcCorCircList(cube2outerTrEdgesList)
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
    verticesArray = calcTrianglesCor(circPtsCoordList, conv)
    print('calcTriangles time: {}'.format(time.time()-t0))
    print(verticesArray.shape[0])

    t0 = time.time()
    solid = mesh.Mesh(np.zeros(verticesArray.shape[0], dtype=mesh.Mesh.dtype))
    solid.vectors[:] = verticesArray
    print('to mesh time: {}'.format(time.time()-t0))
    t0 = time.time()
    solid.save(filename)
    print('save time: {}'.format(time.time()-t0))





