#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - implicit surface function cad software for 3d printing
#    Copyright (c) 2021 - 2024 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


import importlib.metadata
import time

import numpy as np
from numba import njit, objmode, prange, types
from numba.typed import Dict, List
from stl import mesh

from xyzcad import __version__
from xyzcad.tlt import TLT


@njit(cache=True)
def round(x):
    return np.floor(10000.0 * x + 0.5) / 10000.0


@njit(cache=True)
def getInitPnt(func, minVal=-1000.0, maxVal=+1000.0, resSteps=24):
    s0 = func(0.0, 0.0, 0.0)
    for d in range(resSteps):
        for xi in range(2**d):
            x = (xi + 0.5) / (2**d) * (maxVal - minVal) + minVal
            for yi in range(2**d):
                y = (yi + 0.5) / (2**d) * (maxVal - minVal) + minVal
                for zi in range(2**d):
                    z = (zi + 0.5) / (2**d) * (maxVal - minVal) + minVal
                    s = func(x, y, z)
                    if s != s0:
                        return x, y, z, 0.0, 0.0, 0.0
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


@njit(cache=True)
def getSurfacePnt(func, p0, p1, resSteps=24):
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    x0 = p0[0]
    y0 = p0[1]
    z0 = p0[2]

    s0 = func(x0, y0, z0)
    u = 0
    d = +1
    for i in range(resSteps):
        x = x0 * (1 - u) + x1 * u
        y = y0 * (1 - u) + y1 * u
        z = z0 * (1 - u) + z1 * u
        s = func(x, y, z)
        if s != s0:
            s0 = s
            d = -d
        u += d * 1 / 2**i
    return (x, y, z)


@njit(cache=True)
def findSurfacePnt(func, minVal=-1000.0, maxVal=+1000.0, resSteps=24):
    ps = getInitPnt(func, minVal, maxVal, resSteps)
    p = getSurfacePnt(func, (ps[0], ps[1], ps[2]), (ps[3], ps[4], ps[5]), resSteps)
    return p


@njit(cache=True)
def getSurface(func, startPnt, res=1.3):
    x, y, z = startPnt
    ptsList = List([(round(x - res / 2), round(y - res / 2), round(z - res / 2), 0, 0)])
    cubeCornerValsDict = Dict()
    r = res
    while ptsList:
        x, y, z, d, c_val_old = ptsList.pop()
        xh = round(x + r)
        yh = round(y + r)
        zh = round(z + r)
        xl = round(x - r)
        yl = round(y - r)
        zl = round(z - r)

        if d == 1:
            v000 = 0 < (c_val_old & 16)  # v100 old
            v100 = func(xh, y, z)
            v010 = 0 < (c_val_old & 64)  # v110 old
            v110 = func(xh, yh, z)
            v001 = 0 < (c_val_old & 32)  # v101 old
            v101 = func(xh, y, zh)
            v011 = 0 < (c_val_old & 128)  # v111 old
            v111 = func(xh, yh, zh)
        elif d == -1:
            v000 = func(x, y, z)
            v100 = 0 < (c_val_old & 1)  # v000 old
            v010 = func(x, yh, z)
            v110 = 0 < (c_val_old & 4)  # v010 old
            v001 = func(x, y, zh)
            v101 = 0 < (c_val_old & 2)  # v001 old
            v011 = func(x, yh, zh)
            v111 = 0 < (c_val_old & 8)  # v011 old
        elif d == 2:
            v000 = 0 < (c_val_old & 4)  # v010 old
            v100 = 0 < (c_val_old & 64)  # v110 old
            v010 = func(x, yh, z)
            v110 = func(xh, yh, z)
            v001 = 0 < (c_val_old & 8)  # v011 old
            v101 = 0 < (c_val_old & 128)  # v111 old
            v011 = func(x, yh, zh)
            v111 = func(xh, yh, zh)
        elif d == -2:
            v000 = func(x, y, z)
            v100 = func(xh, y, z)
            v010 = 0 < (c_val_old & 1)  # v000 old
            v110 = 0 < (c_val_old & 16)  # v100 old
            v001 = func(x, y, zh)
            v101 = func(xh, y, zh)
            v011 = 0 < (c_val_old & 2)  # v001 old
            v111 = 0 < (c_val_old & 32)  # v101 old
        elif d == 4:
            v000 = 0 < (c_val_old & 2)  # v001 old
            v100 = 0 < (c_val_old & 32)  # v101 old
            v010 = 0 < (c_val_old & 8)  # v011 old
            v110 = 0 < (c_val_old & 128)  # v111 old
            v001 = func(x, y, zh)
            v101 = func(xh, y, zh)
            v011 = func(x, yh, zh)
            v111 = func(xh, yh, zh)
        elif d == -4:
            v000 = func(x, y, z)
            v100 = func(xh, y, z)
            v010 = func(x, yh, z)
            v110 = func(xh, yh, z)
            v001 = 0 < (c_val_old & 1)  # v000 old
            v101 = 0 < (c_val_old & 16)  # v100 old
            v011 = 0 < (c_val_old & 4)  # v010 old
            v111 = 0 < (c_val_old & 64)  # v110 old
        else:
            v000 = func(x, y, z)
            v100 = func(xh, y, z)
            v010 = func(x, yh, z)
            v110 = func(xh, yh, z)
            v001 = func(x, y, zh)
            v101 = func(xh, y, zh)
            v011 = func(x, yh, zh)
            v111 = func(xh, yh, zh)
        cVal = (
            128 * v111
            + 64 * v110
            + 32 * v101
            + 16 * v100
            + 8 * v011
            + 4 * v010
            + 2 * v001
            + 1 * v000
        )
        if cVal == 255 or cVal == 0:
            continue
        if (not (v100 and v110 and v101 and v111)) and (v100 or v110 or v101 or v111):
            if not d == -1:
                if (xh, y, z) not in cubeCornerValsDict:
                    ptsList.append((xh, y, z, +1, cVal))
        if (not (v010 and v110 and v011 and v111)) and (v010 or v110 or v011 or v111):
            if not d == -2:
                if (x, yh, z) not in cubeCornerValsDict:
                    ptsList.append((x, yh, z, +2, cVal))
        if (not (v001 and v101 and v011 and v111)) and (v001 or v101 or v011 or v111):
            if not d == -4:
                if (x, y, zh) not in cubeCornerValsDict:
                    ptsList.append((x, y, zh, +4, cVal))
        if (not (v000 and v010 and v001 and v011)) and (v000 or v010 or v001 or v011):
            if not d == 1:
                if (xl, y, z) not in cubeCornerValsDict:
                    ptsList.append((xl, y, z, -1, cVal))
        if (not (v000 and v100 and v001 and v101)) and (v000 or v100 or v001 or v101):
            if not d == 2:
                if (x, yl, z) not in cubeCornerValsDict:
                    ptsList.append((x, yl, z, -2, cVal))
        if (not (v000 and v100 and v010 and v110)) and (v000 or v100 or v010 or v110):
            if not d == 4:
                if (x, y, zl) not in cubeCornerValsDict:
                    ptsList.append((x, y, zl, -4, cVal))
        cubeCornerValsDict[(x, y, z)] = np.uint8(cVal)

    return cubeCornerValsDict


# removed convert to set() - may trigger numba bug
@njit(cache=True)
def convert_corners2cubes(cubes_coord2cornersval_dict):
    return np.asarray(list(cubes_coord2cornersval_dict.keys())), np.asarray(
        list(cubes_coord2cornersval_dict.values())
    )


@njit(cache=True)
def convert_corners2pts(cubeCornerValsDict, r):
    pts_res_dict = {}
    for k, v in cubeCornerValsDict.items():
        x, y, z = k
        xh = round(x + r)
        yh = round(y + r)
        zh = round(z + r)
        pts_res_dict[(x, y, z)] = int(0 < (v & 1))
        pts_res_dict[(xh, y, z)] = int(0 < (v & 16))
        pts_res_dict[(x, yh, z)] = int(0 < (v & 4))
        pts_res_dict[(x, y, zh)] = int(0 < (v & 2))
        pts_res_dict[(xh, y, zh)] = int(0 < (v & 32))
        pts_res_dict[(x, yh, zh)] = int(0 < (v & 8))
        pts_res_dict[(xh, yh, z)] = int(0 < (v & 64))
        pts_res_dict[(xh, yh, zh)] = int(0 < (v & 128))
    ptCoordDictKeys = np.asarray(list(pts_res_dict.keys()))
    ptCoordDictVals = np.asarray(list(pts_res_dict.values()))
    return ptCoordDictKeys, ptCoordDictVals


@njit(cache=True, parallel=True)
def coords2relations(cubeCoordArray, ptCoordArray, res):
    r = res

    arr_split = 16
    spl_dict = [{(0.0, 0.0, 0.0): 0} for i0 in range(arr_split)]
    for k in range(arr_split):
        spl_dict[k].clear()
    for k in prange(arr_split):
        len_arr = int(ptCoordArray.shape[0] / arr_split + 1.0)
        splitted_arr = ptCoordArray[
            (len_arr * k) : min(len_arr * (k + 1), ptCoordArray.shape[0])
        ]
        spl_dict[k] = {
            (p[0], p[1], p[2]): i1 + len_arr * k for i1, p in enumerate(splitted_arr)
        }

    ptCoordDictRev = spl_dict[0]
    for k in range(1, arr_split):
        ptCoordDictRev.update(spl_dict[k])

    cube2ptIdxArray = np.zeros((cubeCoordArray.shape[0], 8), dtype="int")
    for i2 in prange(cubeCoordArray.shape[0]):
        p = cubeCoordArray[i2]
        x, y, z = p
        xh = round(x + r)
        yh = round(y + r)
        zh = round(z + r)
        cube2ptIdxArray[i2] = [
            ptCoordDictRev[(x, y, z)],
            ptCoordDictRev[(xh, y, z)],
            ptCoordDictRev[(x, yh, z)],
            ptCoordDictRev[(xh, yh, z)],
            ptCoordDictRev[(x, y, zh)],
            ptCoordDictRev[(xh, y, zh)],
            ptCoordDictRev[(x, yh, zh)],
            ptCoordDictRev[(xh, yh, zh)],
        ]

    cEdgeArray = np.zeros((cube2ptIdxArray.shape[0] * 12, 2), dtype="int")
    for i3 in prange(cube2ptIdxArray.shape[0]):
        cube = cube2ptIdxArray[i3]
        cEdgeArray[12 * i3 + 0] = (cube[0], cube[1])
        cEdgeArray[12 * i3 + 1] = (cube[0], cube[2])
        cEdgeArray[12 * i3 + 2] = (cube[0], cube[4])
        cEdgeArray[12 * i3 + 3] = (cube[6], cube[7])
        cEdgeArray[12 * i3 + 4] = (cube[5], cube[7])
        cEdgeArray[12 * i3 + 5] = (cube[3], cube[7])
        cEdgeArray[12 * i3 + 6] = (cube[1], cube[5])
        cEdgeArray[12 * i3 + 7] = (cube[1], cube[3])
        cEdgeArray[12 * i3 + 8] = (cube[2], cube[3])
        cEdgeArray[12 * i3 + 9] = (cube[2], cube[6])
        cEdgeArray[12 * i3 + 10] = (cube[4], cube[6])
        cEdgeArray[12 * i3 + 11] = (cube[4], cube[5])
    cEdgesSet = set([(e[0], e[1]) for e in cEdgeArray])

    edge2ptIdxArray = np.asarray(list(cEdgesSet))

    edge2ptIdxDict = {(e[0], e[1]): i4 for i4, e in enumerate(edge2ptIdxArray)}

    cube2edgeIdxArray = np.zeros((cube2ptIdxArray.shape[0], 12), dtype="int")

    for i5 in prange(len(cube2ptIdxArray)):
        cube = cube2ptIdxArray[i5]
        cube2edgeIdxArray[i5] = [
            edge2ptIdxDict[(cube[0], cube[1])],
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
            edge2ptIdxDict[(cube[4], cube[5])],
        ]

    return (
        cube2ptIdxArray,
        cube2edgeIdxArray,
        edge2ptIdxArray,
    )


@njit(cache=True)
def cutCedgeIdx(edge2ptIdxList, ptValueList):
    return np.asarray(
        [
            i
            for i, e in enumerate(edge2ptIdxList)
            if ptValueList[e[0]] != ptValueList[e[1]]
        ]
    )


@njit(cache=True, parallel=True)
def precTrPnts(func, cutCedgeIdxArray, edge2ptIdxArray, ptCoordArray):
    lcceil = len(cutCedgeIdxArray)
    r = np.zeros((lcceil, 3))
    for i in prange(lcceil):
        p0, p1 = edge2ptIdxArray[cutCedgeIdxArray[i]]
        r[i] = getSurfacePnt(func, ptCoordArray[p0], ptCoordArray[p1])
    return r


@njit(cache=True)
def tridx2triangle(tr_lst, cutCedgeIdxList, precTrPnts):
    cutCedgeIdxRevDict = {e: i for i, e in enumerate(cutCedgeIdxList)}
    tr_arr = np.zeros((len(tr_lst) * 8, 3, 3))
    c = 0
    poly = np.zeros((6, 3))
    for k in range(len(tr_lst)):
        tr = tr_lst[k]
        n = 0
        for m in range(len(tr)):
            f = tr[m]
            if f in cutCedgeIdxRevDict:
                poly[n] = precTrPnts[cutCedgeIdxRevDict[f]]
                n += 1
        for i in range(n - 2):
            tr_arr[c][0] = poly[0]
            tr_arr[c][1] = poly[i + 1]
            tr_arr[c][2] = poly[i + 2]
            c += 1
    return tr_arr[:c]


@njit(cache=True)
def build_repair_polygons(single_edge_dict):
    ac = List()
    while len(single_edge_dict) > 0:
        f = List()
        e = List(single_edge_dict.keys())[0]
        while e in single_edge_dict:
            en = single_edge_dict[e]
            f.append(e)
            del single_edge_dict[e]
            e = en
        ac.append(f)
    return ac


@njit(cache=True)
def time_it():
    with objmode(t="f8"):
        t = time.perf_counter()
    return t


@njit(cache=True)
def print_state(state, t0c):
    t2 = time_it()
    t0, t1, fn_name = state
    with objmode():
        print(
            "{:36s} {:9.4f} s {:9.4f} s {:9.4f} s".format(
                fn_name, t0 - t0c, t1 - t0, t2 - t1
            )
        )
    return time_it()


@njit(cache=True)
def print_summary(summary, indent=0):
    for k, v in summary.items():
        with objmode():
            print("{:s}{:22s} {:20d}".format(" " * indent, k, v))

    return time_it()


@njit(cache=True)
def log_it(t0, text):
    t1 = time_it()
    with objmode():
        print("[{:9.4f}] {}".format(t1 - t0, text))


@njit(cache=True)
def repair_surface(poly_list):
    poly_edge_list = [
        (e[(i + 1) % len(e)], e[i]) for e in poly_list for i, f in enumerate(e)
    ]
    s1 = set(poly_edge_list)
    s2 = set([(e[1], e[0]) for e in poly_edge_list])
    singleEdgeSet = s1.difference(s2)
    singleEdgeDict = {k: v for k, v in singleEdgeSet}
    ac = build_repair_polygons(singleEdgeDict)
    return ac


@njit(cache=True)
def calc_polygons(c2e, cvList):
    tlt = TLT()
    return List(
        [List([c2e[i][k] for k in t]) for i, c in enumerate(cvList) for t in tlt[c]]
    )


@njit(cache=True)
def calc_closed_surface(c2e, cvList):
    polyList = calc_polygons(c2e, cvList)
    rep = repair_surface(polyList)
    polyList.extend(rep)
    return polyList, len(rep)


@njit(cache=True)
def all_njit_func(func, res, t0):
    summary = {}
    log_it(t0, "Searching initial point on surface")
    p = findSurfacePnt(func)
    log_it(t0, "Walking over entire surface")
    corners = getSurface(func, p, res)
    summary["cubes"] = len(corners)
    log_it(t0, "Converting corners into points")
    ptsKeys, pv = convert_corners2pts(corners, res)
    summary["cube points"] = len(pv)
    log_it(t0, "Converting corners into cubes")
    cubesArray, cvList = convert_corners2cubes(corners)
    log_it(t0, "Converting coordinates into relations")
    c2p, c2e, e2p = coords2relations(cubesArray, ptsKeys, res)
    summary["cube edges"] = len(e2p)
    log_it(t0, "Searching all marching cubes edges, cut by surface")
    cCeI = cutCedgeIdx(e2p, pv)
    summary["surface points"] = len(cCeI)
    log_it(t0, "Approximating exact coordinates of the cuts")
    precTrPtsList = precTrPnts(func, cCeI, e2p, ptsKeys)
    log_it(t0, "Calculating closed surface")
    corCircList, len_rep = calc_closed_surface(c2e, cvList)
    log_it(t0, "Building triangles")
    summary["polygons"] = len(corCircList)
    summary["repaired polygons"] = len_rep
    verticesArray = tridx2triangle(corCircList, cCeI, precTrPtsList)
    log_it(t0, "Calculations done")
    summary["triangles"] = len(verticesArray)
    return verticesArray, summary


def renderAndSave(func, filename, res=1):
    t0 = time.time()
    version_run = __version__
    try:
        version_inst = importlib.metadata.version(__package__ or __name__)
    except importlib.metadata.PackageNotFoundError as e:
        version_inst = None
    t0 = time_it()
    log_it(t0, f"running xyzcad version {version_run} (installed: {version_inst})")
    log_it(t0, "Compiling")
    verticesArray, summary = all_njit_func(func, res, t0)
    print_summary(summary, 14)
    log_it(t0, "Building mesh")
    solid = mesh.Mesh(np.zeros(verticesArray.shape[0], dtype=mesh.Mesh.dtype))
    solid.vectors[:] = verticesArray
    log_it(t0, f"Saving file: {filename}")
    solid.save(filename)
    log_it(t0, "Done.")
