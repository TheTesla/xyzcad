#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################

import pickle


import numpy as np
#import open3d as o3d
import time
from numba import njit, jit, prange, types
from numba.typed import List, Dict

from stl import mesh

tlt = [[[0]]] * 256
#tlt[0] = [[]]
tlt[1] = [[0, 1, 2]]
tlt[254] = [[0, 2, 1]]
tlt[2] = [[2, 10, 11]]
tlt[253] = [[2, 11, 10]]
tlt[3] = [[0, 1, 10, 11]]
tlt[252] = [[0, 11, 10, 1]]
tlt[4] = [[1, 8, 9]]
tlt[251] = [[1, 9, 8]]
tlt[5] = [[0, 8, 9, 2]]
tlt[250] = [[0, 2, 9, 8]]
tlt[6] = [[1, 8, 9], [2, 10, 11]]
tlt[249] = [[9, 8, 1], [11, 10, 2]]
tlt[7] = [[0, 8, 9, 10, 11]]
tlt[248] = [[0, 11, 10, 9, 8]]
tlt[8] = [[3, 10, 9]]
tlt[247] = [[3, 9, 10]]
tlt[9] = [[0, 1, 2], [3, 10, 9]]
tlt[246] = [[2, 1, 0], [9, 10, 3]]
tlt[10] = [[2, 9, 3, 11]]
tlt[245] = [[2, 11, 3, 9]]
tlt[11] = [[0, 1, 9, 3, 11]]
tlt[244] = [[0, 11, 3, 9, 1]]
tlt[12] = [[1, 8, 3, 10]]
tlt[243] = [[1, 10, 3, 8]]
tlt[13] = [[0, 8, 3, 10, 2]]
tlt[242] = [[0, 2, 10, 3, 8]]
tlt[14] = [[1, 8, 3, 11, 2]]
tlt[241] = [[1, 2, 11, 3, 8]]
tlt[15] = [[0, 8, 3, 11]]
tlt[240] = [[0, 11, 3, 8]]
tlt[16] = [[0, 6, 7]]
tlt[239] = [[0, 7, 6]]
tlt[17] = [[1, 2, 6, 7]]
tlt[238] = [[1, 7, 6, 2]]
tlt[18] = [[0, 6, 7], [2, 10, 11]]
tlt[237] = [[7, 6, 0], [11, 10, 2]]
tlt[19] = [[1, 10, 11, 6, 7]]
tlt[236] = [[1, 7, 6, 11, 10]]
tlt[20] = [[0, 6, 7], [1, 8, 9]]
tlt[235] = [[7, 6, 0], [9, 8, 1]]
tlt[21] = [[2, 6, 7, 8, 9]]
tlt[234] = [[2, 9, 8, 7, 6]]
tlt[22] = [[2,10,11],[0,6,7],[1,8,9]]
tlt[233] = [[11,10,2],[0,7,6],[1,9,8]]
tlt[23] = [[6, 7, 8, 9, 10, 11]]
tlt[232] = [[11, 10, 9, 8, 7, 6]]
tlt[24] = [[3,10,9],[0,6,7]]
tlt[231] = [[3,9,10],[0,7,6]]
tlt[25] = [[1,2,6,7],[3,10,9]]
tlt[230] = [[7,6,2,1],[3,9,10]]
tlt[26] = [[2, 9, 3, 11],[0,6,7]]
tlt[229] = [[11, 3, 9, 2],[0,7,6]]
tlt[27] = [[1, 9, 3, 11, 6, 7]]
tlt[228] = [[7, 6, 11, 3, 9, 1]]
tlt[28] = [[1, 8, 3, 10],[0, 6, 7]]
tlt[227] = [[10, 3, 8, 1],[0, 7, 6]]
#tlt[29] = [[]]
tlt[30] = [[0,6,7],[1, 8, 3, 11, 2]]
tlt[225] = [[0,7,6],[2, 11, 3, 8, 1]]
tlt[31] = [[3, 11, 6, 7, 8]]
tlt[224] = [[3, 8, 7, 6, 11]]
tlt[32] = [[6, 11, 4]]
tlt[223] = [[6, 4, 11]]
tlt[33] = [[0, 1, 2], [4, 6, 11]]
tlt[222] = [[2, 1, 0], [11, 6, 4]]
tlt[34] = [[2, 10, 4, 6]]
tlt[221] = [[2, 6, 4, 10]]
tlt[35] = [[0, 1, 10, 4, 6]]
tlt[220] = [[0, 6, 4, 10, 1]]
tlt[36] = [[4,6,11],[1,8,9]]
tlt[219] = [[4,11,6],[1,9,8]]
#tlt[37] = [[]]
#tlt[38] = [[]]

tlt[39] = [[0, 8, 9, 10, 4, 6]]
tlt[216] = [[6, 4, 10, 9, 8, 0]]

tlt[40] = [[4,6,11],[3,10,9]]
tlt[215] = [[4,11,6],[3,9,10]]

tlt[41] = [[0,1,2],[4,6,11],[3,10,9]]
tlt[214] = [[0,2,1],[4,11,6],[3,9,10]]

tlt[42] = [[2, 9, 3, 4, 6]]
tlt[213] = [[2, 6, 4, 3, 9]]

tlt[43] = [[0, 1, 9, 3, 4, 6]]
tlt[212] = [[6, 4, 3, 9, 1, 0]]

tlt[46] = [[1, 8, 3, 4, 6, 2]]
tlt[209] = [[2, 6, 4, 3, 8, 1]]

tlt[47] = [[0, 8, 3, 4, 6]]
tlt[208] = [[0, 6, 4, 3, 8]]

tlt[48] = [[0, 11, 4, 7]]
tlt[207] = [[0, 7, 4, 11]]

tlt[49] = [[1, 2, 11, 4, 7]]
tlt[206] = [[1, 7, 4, 11, 2]]

tlt[50] = [[0, 2, 10, 4, 7]]
tlt[205] = [[0, 7, 4, 10, 2]]

tlt[51] = [[1, 10, 4, 7]]
tlt[204] = [[1, 7, 4, 10]]

tlt[53] = [[4, 7, 8, 9, 2, 11]]
tlt[202] = [[11, 2, 9, 8, 7, 4]]

tlt[54] = [[0, 2, 10, 4, 7], [1,8,9]]
tlt[201] = [[0, 7, 4, 10, 2], [1,9,8]]

tlt[55] = [[4, 7, 8, 9, 10]]
tlt[200] = [[4, 10, 9, 8, 7]]

tlt[59] = [[1, 9, 3, 4, 7]]
tlt[196] = [[1, 7, 4, 3, 9]]

tlt[63] = [[3, 4, 7, 8]]
tlt[192] = [[3, 8, 7, 4]]

tlt[64] = [[5, 8, 7]]
tlt[191] = [[5, 7, 8]]

tlt[65] = [[0,1,2],[5, 8, 7]]
tlt[190] = [[0,2,1],[5, 7, 8]]

tlt[66] = [[2,10,11],[5, 8, 7]]
tlt[189] = [[2,11,10],[5, 7, 8]]

tlt[68] = [[1, 7, 5, 9]]
tlt[187] = [[1, 9, 5, 7]]

tlt[69] = [[0, 7, 5, 9, 2]]
tlt[186] = [[2, 9, 5, 7, 0]]

tlt[71] = [[7, 5, 9, 10, 11, 0]]
tlt[184] = [[0, 11, 10, 9, 5, 7]]

tlt[72] = [[5, 8, 7], [3,10,9]]
tlt[183] = [[5, 7, 8], [3,9,10]]

tlt[76] = [[1, 7, 5, 3, 10]]
tlt[179] = [[1, 10, 3, 5, 7]]

tlt[77] = [[7, 5, 3, 10, 2, 0]]
tlt[178] = [[0, 2, 10, 3, 5, 7]]

tlt[78] = [[1, 7, 5, 3, 11, 2]]
tlt[177] = [[2, 11, 3, 5, 7, 1]]

tlt[79] = [[0, 7, 5, 3, 11]]
tlt[176] = [[0, 11, 3, 5, 7]]

tlt[80] = [[0, 6, 5, 8]]
tlt[175] = [[0, 8, 5, 6]]

tlt[81] = [[1, 2, 6, 5, 8]]
tlt[174] = [[1, 8, 5, 6, 2]]

tlt[83] = [[5, 8, 1, 10, 11, 6]]
tlt[172] = [[6, 11, 10, 1, 8, 5]]

tlt[84] = [[0, 6, 5, 9, 1]]
tlt[171] = [[0, 1, 9, 5, 6]]

tlt[85] = [[2, 6, 5, 9]]
tlt[170] = [[2, 9, 5, 6]]

tlt[87] = [[6, 5, 9, 10, 11]]
tlt[168] = [[6, 11, 10, 9, 5]]

tlt[90] = [[0, 6, 5, 8],[2, 9, 3, 11]]
tlt[165] = [[8, 5, 6, 0],[11, 3, 9, 2]]


tlt[92] = [[6, 5, 3, 10, 1, 0]]
tlt[163] = [[0, 1, 10, 3, 5, 6]]

tlt[93] = [[2, 6, 5, 3, 10]]
tlt[162] = [[2, 10, 3, 5, 6]]

tlt[95] = [[3, 11, 6, 5]]
tlt[160] = [[3, 5, 6, 11]]

tlt[96] = [[4,6,11], [5,8,7]]
tlt[159] = [[4,11,6], [5,7,8]]

tlt[111] = [[3,4,5],[0,7,6]]
tlt[144] = [[3,5,4],[0,6,7]]

tlt[112] = [[0, 11, 4, 5, 8]]
tlt[143] = [[0, 8, 5, 4, 11]]

tlt[113] = [[1, 2, 11, 4, 5, 8]]
tlt[142] = [[8, 5, 4, 11, 2, 1]]

tlt[114] = [[2, 10, 4, 5, 8, 0]]
tlt[141] = [[0, 8, 5, 4, 10, 2]]

tlt[115] = [[1, 10, 4, 5, 8]]
tlt[140] = [[1, 8, 5, 4, 10]]

tlt[116] = [[11, 4, 5, 9, 1, 0]]
tlt[139] = [[0, 1, 9, 5, 4, 11]]

tlt[117] = [[2, 11, 4, 5, 9]]
tlt[138] = [[2, 9, 5, 4, 11]]

tlt[118] = [[0,2,1],[4, 5, 9, 10]]
tlt[137] = [[0,1,2],[4, 10, 9, 5]]

tlt[119] = [[4, 5, 9, 10]]
tlt[136] = [[4, 10, 9, 5]]

tlt[123] = [[3,4,5],[1,9,8]]
tlt[132] = [[3,5,4],[1,8,9]]

tlt[125] = [[3, 4, 5],[2,11,10]]
tlt[130] = [[3, 5, 4],[2,10,11]]

tlt[126] = [[0,2,1],[3, 4, 5]]
tlt[129] = [[0,1,2],[3, 5, 4]]

tlt[127] = [[3, 4, 5]]
tlt[128] = [[3, 5, 4]]





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
    cubeCornerValsDict = Dict()
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
        cVal = 128*v111+64*v110+32*v101+16*v100+8*v011+4*v010+2*v001+1*v000
        cubeCornerValsDict[(x,y,z)] = np.uint8(cVal)
        ptsResDict[(x,y,z)] = v000
        ptsResDict[(xh,y,z)] = v100
        ptsResDict[(x,yh,z)] = v010
        ptsResDict[(xh,yh,z)] = v110
        ptsResDict[(x,y,zh)] = v001
        ptsResDict[(xh,y,zh)] = v101
        ptsResDict[(x,yh,zh)] = v011
        ptsResDict[(xh,yh,zh)] = v111
    cubesList = list(cubeExistsSet)
    cubesArray = np.asarray(cubesList)
    ptCoordDictKeys = np.asarray(list(ptsResDict.keys()))
    ptCoordDictVals = np.asarray(list(ptsResDict.values()))
    cubeCornerValsDictKeys = np.asarray(list(cubeCornerValsDict.keys()))
    cubeCornerValsDictVals = np.asarray(list(cubeCornerValsDict.values()))
    cvList = [cubeCornerValsDict[c] for c in cubesList]
    return cubesArray, ptCoordDictKeys, ptCoordDictVals, cvList #cubeCornerValsDictKeys, cubeCornerValsDictVals


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
    #print(cEdgeArray.shape)
    cEdgesSet = set([(e[0], e[1]) for e in cEdgeArray])

    #print(len(cEdgesSet))
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
    #return List([[precTrPnts[cutCedgeIdxRevDict[f]] for f in e if f in cutCedgeIdxRevDict] for e in
    return List([[precTrPnts[cutCedgeIdxRevDict[f]] for f in e] for e in trList])

@njit
def filter_single_edge(poly_edge_list):
    single_edge_set = set()
    for e in poly_edge_list:
        if e not in single_edge_set:
            if (e[1], e[0]) not in single_edge_set:
                single_edge_set.add(e)
            else:
                single_edge_set.remove((e[1], e[0]))
    return single_edge_set

@njit
def build_repair_polygons(single_edge_dict):
    ac = List()
    while len(single_edge_dict) > 0:
        f = list()
        e = List(single_edge_dict.keys())[0]
        while e in single_edge_dict:
            en = single_edge_dict[e]
            f.append(e)
            del single_edge_dict[e]
            e = en
        ac.append(f)
    return ac


#@njit
def repair_surface(poly_list):
    poly_edge_list = [(e[(i+1)%len(e)], e[i]) for e in poly_list for i, f in enumerate(e)]
    singleEdgeSet = filter_single_edge(poly_edge_list)
    singleEdgeDict = {k: v for k, v in singleEdgeSet}
    ac = build_repair_polygons(singleEdgeDict)
    return ac

def calc_polygons(c2e, cvList, tlt):
    return [[c2e[i][k] for k in t] for i, c in enumerate(cvList) for t in tlt[c]]

def renderAndSave(func, filename, res=1):
    t0 = time.time()
    p = findSurfacePnt(func)
    print('findSurfacePnt time: {}'.format(time.time()-t0))
    #print(p)
    t0 = time.time()
    cubesArray, ptsKeys, ptsVals, cvList = getSurface(func, p, res)
    print('getSurface time: {}'.format(time.time()-t0))
    print('{} - {} - {} - {}'.format(len(cubesArray), len(ptsKeys),
                                          len(ptsVals), len(cvList)))
    #print(len(ptsKeys))
    #print(cvList)
    t0 = time.time()
    c2p, c2e, e2p, pc, pv = coords2relations(cubesArray, ptsKeys, ptsVals, res)
    print('coords2relations time: {}'.format(time.time()-t0))
    print('{} - {} - {} - {} - {}'.format(len(c2p), len(c2e), len(e2p),
        len(pc), len(pv)))
    #print("c2e=")
    #print(c2e)
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
    circList = calc_polygons(c2e, cvList, tlt)
    #circList = [[c2e[i][k] for k in t] for i, c in enumerate(cvList) for t in tlt[c]]
    print('circList time: {}'.format(time.time()-t0))


    with open('circList.pkl', 'wb') as outp:
        pickle.dump(circList, outp, pickle.HIGHEST_PROTOCOL)

    t0 = time.time()
    circList = List(circList)
    print('List(circList) time: {}'.format(time.time()-t0))

    corCircList = circList
    t0 = time.time()
    rep = repair_surface(circList)
    print('repair_surface time: {}'.format(time.time()-t0))





