#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - implicit surface function cad software for 3d printing
#    Copyright (c) 2021 - 2025 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


import importlib.metadata
import time

import numpy as np
from numba import njit, objmode, prange, types
from numba.typed import Dict, List
from stl import mesh

from xyzcad import __version__
from xyzcad.export import export_obj, export_obj_printable, export_stl
from xyzcad.tlt import TLT


def get_installed_version():
    try:
        version_inst = importlib.metadata.version(__package__ or __name__)
    except importlib.metadata.PackageNotFoundError as e:
        version_inst = None


@njit(cache=True)
def rnd(x):
    return np.floor(10000.0 * x + 0.5) / 10000.0


@njit(cache=True)
def find_init_pnt(func, min_val=-1000.0, max_val=+1000.0, res_steps=24, params=()):
    s0 = func((0.0, 0.0, 0.0, params))
    for d in range(res_steps):
        for xi in range(2**d):
            x = (xi + 0.5) / (2**d) * (max_val - min_val) + min_val
            for yi in range(2**d):
                y = (yi + 0.5) / (2**d) * (max_val - min_val) + min_val
                for zi in range(2**d):
                    z = (zi + 0.5) / (2**d) * (max_val - min_val) + min_val
                    s = func((x, y, z, params))
                    if s != s0:
                        return np.array([[x, y, z], [0.0, 0.0, 0.0]])
    return np.zeros((3, 2))


@njit(cache=True)
def get_surfacePnt(func, p0, p1, res_steps=24, params=()):
    s0 = func((p0[0], p0[1], p0[2], params))
    u = 0.0
    d = +1
    for i in range(res_steps):
        p = p0 * (1 - u) + p1 * u
        s = func((p[0], p[1], p[2], params))
        if s != s0:
            s0 = s
            d = -d
        u += d * 1 / 2**i
    return p


@njit(cache=True)
def find_surface_pnt(func, min_val=-1000.0, max_val=+1000.0, res=1.3, params=()):
    res_steps = int(np.floor(np.log2((max_val - min_val) / res)) + 2)
    p0, p1 = find_init_pnt(func, min_val, max_val, res_steps, params)
    p = get_surfacePnt(func, p0, p1, res_steps, params)
    return p


@njit(cache=True)
def get_surface(func, start_pnt, res=1.3, params=()):
    x, y, z = start_pnt
    r = res
    p_nxt = [(rnd(x - r / 2), rnd(y - r / 2), rnd(z - r / 2), 0, 0, 0, 0, 0)]
    corners = Dict()
    v = np.zeros(8, dtype=np.int64)
    vo = np.zeros(4, dtype=np.int64)
    while p_nxt:
        x, y, z, d, vo[0], vo[1], vo[2], vo[3] = p_nxt.pop()
        xh = rnd(x + r)
        yh = rnd(y + r)
        zh = rnd(z + r)
        xl = rnd(x - r)
        yl = rnd(y - r)
        zl = rnd(z - r)
        j = 0
        for i in range(8):
            if (0 == (i & abs(d))) == (1 == np.sign(d)):
                v[i] = vo[j]
                j += 1
            else:
                v[i] = func(
                    (
                        xh if i & 4 else x,
                        yh if i & 2 else y,
                        zh if i & 1 else z,
                        params,
                    )
                )
        for i2 in range(7):
            if v[7] != v[i2]:
                break
        else:
            continue
        if not d == -4:
            if not (v[4] == v[6] and v[4] == v[5] and v[4] == v[7]):
                if (xh, y, z) not in corners:
                    p_nxt.append((xh, y, z, 4, v[4], v[5], v[6], v[7]))
        if not d == -2:
            if not (v[2] == v[6] and v[2] == v[3] and v[2] == v[7]):
                if (x, yh, z) not in corners:
                    p_nxt.append((x, yh, z, 2, v[2], v[3], v[6], v[7]))
        if not d == -1:
            if not (v[1] == v[5] and v[1] == v[3] and v[1] == v[7]):
                if (x, y, zh) not in corners:
                    p_nxt.append((x, y, zh, 1, v[1], v[3], v[5], v[7]))
        if not d == 4:
            if not (v[0] == v[2] and v[0] == v[1] and v[0] == v[3]):
                if (xl, y, z) not in corners:
                    p_nxt.append((xl, y, z, -4, v[0], v[1], v[2], v[3]))
        if not d == 2:
            if not (v[0] == v[4] and v[0] == v[1] and v[0] == v[5]):
                if (x, yl, z) not in corners:
                    p_nxt.append((x, yl, z, -2, v[0], v[1], v[4], v[5]))
        if not d == 1:
            if not (v[0] == v[4] and v[0] == v[2] and v[0] == v[6]):
                if (x, y, zl) not in corners:
                    p_nxt.append((x, y, zl, -1, v[0], v[2], v[4], v[6]))
        corners[(x, y, z)] = (v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0])

    return corners


# removed convert to set() - may trigger numba bug
@njit(cache=True)
def convert_corners2cubes(cubes_coord2cornersval_dict):
    return np.asarray(list(cubes_coord2cornersval_dict.keys())), np.asarray(
        list(cubes_coord2cornersval_dict.values())
    )


@njit(cache=True)
def convert_corners2pts(corners, r):
    pts_res_dict = {}
    for k, v in corners.items():
        x, y, z = k
        xh = rnd(x + r)
        yh = rnd(y + r)
        zh = rnd(z + r)
        pts_res_dict[(x, y, z)] = v[7]
        pts_res_dict[(xh, y, z)] = v[3]
        pts_res_dict[(x, yh, z)] = v[5]
        pts_res_dict[(x, y, zh)] = v[6]
        pts_res_dict[(xh, y, zh)] = v[2]
        pts_res_dict[(x, yh, zh)] = v[4]
        pts_res_dict[(xh, yh, z)] = v[1]
        pts_res_dict[(xh, yh, zh)] = v[0]
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
        xh = rnd(x + r)
        yh = rnd(y + r)
        zh = rnd(z + r)
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
def precTrPnts(func, cutCedgeIdxArray, edge2ptIdxArray, ptCoordArray, params):
    lcceil = len(cutCedgeIdxArray)
    r = np.zeros((lcceil, 3))
    for i in prange(lcceil):
        p0, p1 = edge2ptIdxArray[cutCedgeIdxArray[i]]
        r[i] = get_surfacePnt(func, ptCoordArray[p0], ptCoordArray[p1], 24, params)
    return r


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


@njit
def conv_cube_edge_2_vrtx_idx(poly_cube_edge_idx, cut_edges):
    cut_edges_rev = {e: i for i, e in enumerate(cut_edges)}
    poly_vrtx_idx = List(
        [
            List([cut_edges_rev[e] for e in f if e in cut_edges_rev])
            for f in poly_cube_edge_idx
        ]
    )
    return poly_vrtx_idx


@njit
def count_clss(clss_arr, poly_vrtx_idx):
    clss_poly_arr = np.zeros((len(poly_vrtx_idx), 256))
    for i in range(len(poly_vrtx_idx)):
        for v in poly_vrtx_idx[i]:
            for c in clss_arr[v]:
                if c == 0:
                    continue
                clss_poly_arr[i, c] += 1
    return clss_poly_arr


@njit(cache=True)
def mesh_surface_function(func, res, params, t0):
    summary = {}
    log_it(t0, "Searching initial point on surface")
    p = find_surface_pnt(func, res=res, params=params)
    log_it(t0, "Walking over entire surface")
    corners = get_surface(func, p, res, params)
    summary["cubes"] = len(corners)
    log_it(t0, "Converting corners into points")
    materials = list(set([c for v in corners.values() for c in v]))
    summary["materials"] = len(materials) - 1
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
    precTrPtsList = precTrPnts(func, cCeI, e2p, ptsKeys, params)
    poly_vrtx_idx_grpd = []
    mats = []
    summary["filtered cubes"] = 0
    summary["polygons"] = 0
    summary["repaired polygons"] = 0
    for imat, mat in enumerate(materials[1:]):
        log_it(t0, f"Material {mat}")
        filtered_cv = np.asarray(
            [
                np.sum((v == mat) * np.array([128, 64, 32, 16, 8, 4, 2, 1]))
                for v in cvList
            ]
        )
        summary["filtered cubes"] += len(filtered_cv)
        log_it(t0, "  Calculating closed surface")
        corCircList, len_rep = calc_closed_surface(c2e, filtered_cv)
        summary["polygons"] += len(corCircList)
        summary["repaired polygons"] += len_rep
        log_it(t0, "  Prepare polygons")
        poly_vrtx_idx = conv_cube_edge_2_vrtx_idx(corCircList, cCeI)
        log_it(t0, "  Meshing done")
        poly_vrtx_idx_grpd.append(poly_vrtx_idx)
        mats.append(mat)
    return precTrPtsList, poly_vrtx_idx_grpd, mats, summary


def save_files(name, vertices, faces_grpd, mats, t0):
    export_formats = {"stl", "obj", "obj_printable"}
    if len(name) > 4:
        if name[-4:] == ".stl":
            name = name[:-4]
            export_formats = {"stl"}
        elif name[-4:] == ".obj":
            name = name[:-4]
            export_formats = {"obj", "obj_printable"}
    if "obj" in export_formats:
        log_it(t0, f"Saving {name}_not_printable.obj")
        export_obj(f"{name}_not_printable", vertices, faces_grpd, mats)
    if "obj_printable" in export_formats:
        log_it(t0, f"Saving {name}.obj")
        export_obj_printable(f"{name}", vertices, faces_grpd, mats)
    if "stl" in export_formats:
        log_it(t0, f"Saving {name}.stl")
        export_stl(name, vertices, [f for e in faces_grpd for f in e])
    if "stl_parts" in export_formats:
        for i, faces in enumerate(faces_grpd):
            if len(faces) == 0:
                continue
            log_it(t0, f"Saving {name}_prt{i:03d}.stl")
            export_stl(f"{name}_prt{i:03d}", vertices, List(faces))


def renderAndSave(func, filename, res=1, params=()):
    t0 = time.time()
    version_run = __version__
    version_inst = get_installed_version()
    t0 = time_it()
    log_it(t0, f"running xyzcad version {version_run} (installed: {version_inst})")
    log_it(t0, "Compiling")
    vertices, faces_grpd, mats, summary = mesh_surface_function(func, res, params, t0)
    # faces_grpd_cln = [[e for e in f if len(e) > 0] for f in faces_grpd]
    save_files(filename, vertices, faces_grpd, mats, t0)
    log_it(t0, "Done.")
    print_summary(summary, 14)
