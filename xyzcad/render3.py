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
from stl import mesh


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
    ptsList = [(round(x), #np.floor(1000*x+0.5)/1000,
                round(y), #np.floor(1000*y+0.5)/1000,
                round(z)) ] #np.floor(1000*z+0.5)/1000) ]
    x,y,z = ptsList[0]
    ptsResDict = dict()
    ptsResDict[(x,y,z)] = func(x,y,z)
    r = res

    while ptsList:
        x,y,z = ptsList.pop()

        xl = round(x-r) #np.floor(1000*(x-r)+0.5)/1000
        xu = func(xl,y,z)
        xh = round(x+r) #np.floor(1000*(x+r)+0.5)/1000
        xo = func(xh,y,z)
        yl = round(y-r) #np.floor(1000*(y-r)+0.5)/1000
        yu = func(x,yl,z)
        yh = round(y+r) #np.floor(1000*(y+r)+0.5)/1000
        yo = func(x,yh,z)
        zl = round(z-r) #np.floor(1000*(z-r)+0.5)/1000
        zu = func(x,y,zl)
        zh = round(z+r) #np.floor(1000*(z+r)+0.5)/1000
        zo = func(x,y,zh)

        s = xu + xo + yu + yo + zu + zo
        if s == 6 or s == 0:
            continue


        #if s == 5:
        #    if (xl,y,z) not in ptsResDict:
        #        if xo:
        #            ptsResDict[(xl,y,z)] = xu
        #            ptsList.append((xl,y,z))
        #    if (xh,y,z) not in ptsResDict:
        #        if xu:
        #            ptsResDict[(xh,y,z)] = xo
        #            ptsList.append((xh,y,z))
        #    if (x,yl,z) not in ptsResDict:
        #        if yo:
        #            ptsResDict[(x,yl,z)] = yu
        #            ptsList.append((x,yl,z))
        #    if (x,yh,z) not in ptsResDict:
        #        if yu:
        #            ptsResDict[(x,yh,z)] = yo
        #            ptsList.append((x,yh,z))
        #    if (x,y,zl) not in ptsResDict:
        #        if zo:
        #            ptsResDict[(x,y,zl)] = zu
        #            ptsList.append((x,y,zl))
        #    if (x,y,zh) not in ptsResDict:
        #        if zu:
        #            ptsResDict[(x,y,zh)] = zo
        #            ptsList.append((x,y,zh))
        #elif s == 2:
        #    if (xl,y,z) not in ptsResDict:
        #        if not xo:
        #            ptsResDict[(xl,y,z)] = xu
        #            ptsList.append((xl,y,z))
        #    if (xh,y,z) not in ptsResDict:
        #        if not xu:
        #            ptsResDict[(xh,y,z)] = xo
        #            ptsList.append((xh,y,z))
        #    if (x,yl,z) not in ptsResDict:
        #        if not yo:
        #            ptsResDict[(x,yl,z)] = yu
        #            ptsList.append((x,yl,z))
        #    if (x,yh,z) not in ptsResDict:
        #        if not yu:
        #            ptsResDict[(x,yh,z)] = yo
        #            ptsList.append((x,yh,z))
        #    if (x,y,zl) not in ptsResDict:
        #        if not zo:
        #            ptsResDict[(x,y,zl)] = zu
        #            ptsList.append((x,y,zl))
        #    if (x,y,zh) not in ptsResDict:
        #        if not zu:
        #            ptsResDict[(x,y,zh)] = zo
        #            ptsList.append((x,y,zh))
        #else:
        if (xl,y,z) not in ptsResDict:
            ptsResDict[(xl,y,z)] = xu
            ptsList.append((xl,y,z))
        if (xh,y,z) not in ptsResDict:
            ptsResDict[(xh,y,z)] = xo
            ptsList.append((xh,y,z))
        if (x,yl,z) not in ptsResDict:
            ptsResDict[(x,yl,z)] = yu
            ptsList.append((x,yl,z))
        if (x,yh,z) not in ptsResDict:
            ptsResDict[(x,yh,z)] = yo
            ptsList.append((x,yh,z))
        if (x,y,zl) not in ptsResDict:
            ptsResDict[(x,y,zl)] = zu
            ptsList.append((x,y,zl))
        if (x,y,zh) not in ptsResDict:
            ptsResDict[(x,y,zh)] = zo
            ptsList.append((x,y,zh))

    return ptsResDict


#@jit(nopython=True,cache=True)
def pts2Edges(ptsResDict, res):
    r = res
    edgeDict = dict()
    for p, v in ptsResDict.items():
        x, y, z = p
        xh = round(x+r) #np.floor(1000*(x+r)+0.5)/1000
        try:
            if ptsResDict[(xh,y,z)] != v:
                edgeDict[(x,y,z,xh,y,z)] = v
        except:
            pass
        yh = round(y+r) #np.floor(1000*(y+r)+0.5)/1000
        try:
            if ptsResDict[(x,yh,z)] != v:
                edgeDict[(x,y,z,x,yh,z)] = v
        except:
            pass
        zh = round(z+r) #np.floor(1000*(z+r)+0.5)/1000
        try:
            if ptsResDict[(x,y,zh)] != v:
                edgeDict[(x,y,z,x,y,zh)] = v
        except:
            pass
    return edgeDict

#@jit(nopython=True,cache=True)
def edgePrec(func, edgeDict, res):
    r = res
    edgeDictPrec = dict()
    for e, v in edgeDict.items():
        p0 = e[:3]
        p1 = e[3:]
        pp = getSurfacePnt(func, p0, p1)
        #pp = (np.array(p0) + np.array(p1))/2
        #print('{} - {} - {}'.format(p0,pp,p1))
        edgeDictPrec[e] = pp #(pp[0], pp[1], pp[2])
    return edgeDictPrec

@jit(inline='always')
def pts2dir(x1,y1,z1,x2,y2,z2,v):
    if v:
        return (x2-x1,y2-y1,z2-z1)
    else:
        return (x1-x2,y1-y2,z1-z2)


#@jit(nopython=True,cache=True)
def edge2cube(ptsDict, edgeDictPrec, res):
    r = res
    cntList = []
    cubeList = []
    histo = 8* [0]
    for p, v in ptsDict.items():
        x,y,z = p
        xh = round(x+r) #np.floor(1000*(x+r)+0.5)/1000
        yh = round(y+r) #np.floor(1000*(y+r)+0.5)/1000
        zh = round(z+r) #np.floor(1000*(z+r)+0.5)/1000
        i = 0
        try:
            vx = ptsDict[(xh,y,z)]
        except:
            pass
        try:
            vy = ptsDict[(x,yh,z)]
        except:
            pass
        try:
            vz = ptsDict[(x,y,zh)]
        except:
            pass
        try:
            vxy = ptsDict[(xh,yh,z)]
        except:
            pass
        try:
            vxz = ptsDict[(xh,y,zh)]
        except:
            pass
        try:
            vyz = ptsDict[(x,yh,zh)]
        except:
            pass
        try:
            vxyz = ptsDict[(xh,yh,zh)]
        except:
            pass
        cube = []
        try:
            px = edgeDictPrec[(x,y,z,xh,y,z)]
            #cube.append((px,y,z))
            cube.append((px, pts2dir(x,y,z,xh,y,z,v), (0,-1,-1)))
            i += 1
        except:
            pass
        try:
            pxy = edgeDictPrec[(xh,y,z,xh,yh,z)]
            cube.append((pxy, pts2dir(xh,y,z,xh,yh,z,vx), (+1,0,-1)))

            #cube.append((xh,pxy,z))
            i += 1
        except:
            pass
        try:
            py = edgeDictPrec[(x,y,z,x,yh,z)]
            cube.append((py, pts2dir(x,y,z,x,yh,z,v), (-1,0,-1)))
            #cube.append((x,py,z))
            i += 1
        except:
            pass
        try:
            pyx = edgeDictPrec[(x,yh,z,xh,yh,z)]
            cube.append((pyx, pts2dir(x,yh,z,xh,yh,z,vy), (0,+1,-1)))
            #cube.append((px,yh,z))
            i += 1
        except:
            pass
        try:
            pz = edgeDictPrec[(x,y,z,x,y,zh)]
            cube.append((pz, pts2dir(x,y,z,x,y,zh,v), (-1,-1,0)))
            i += 1
        except:
            pass
        try:
            pzx = edgeDictPrec[(x,y,zh,xh,y,zh)]
            cube.append((pzx, pts2dir(x,y,zh,xh,y,zh,vz), (0,-1,+1)))
            i += 1
        except:
            pass
        try:
            pzxy = edgeDictPrec[(xh,y,zh,xh,yh,zh)]
            cube.append((pzxy, pts2dir(xh,y,zh,xh,yh,zh,vxy), (+1,0,+1)))
            i += 1
        except:
            pass
        try:
            pzy = edgeDictPrec[(x,y,zh,x,yh,zh)]
            cube.append((pzy, pts2dir(x,y,zh,x,yh,zh,vz), (-1,0,+1)))
            i += 1
        except:
            pass
        try:
            pzyx = edgeDictPrec[(x,yh,zh,xh,yh,zh)]
            cube.append((pzyx, pts2dir(x,yh,zh,xh,yh,zh,vyz), (0,+1,+1)))
            i += 1
        except:
            pass
        try:
            pxz = edgeDictPrec[(xh,y,z,xh,y,zh)]
            cube.append((pxz, pts2dir(xh,y,z,xh,y,zh,vx), (+1,-1,0)))
            i += 1
        except:
            pass
        try:
            pyz = edgeDictPrec[(x,yh,z,x,yh,zh)]
            cube.append((pyz, pts2dir(x,yh,z,x,yh,zh,vy), (-1,+1,0)))
            i += 1
        except:
            pass
        try:
            pxyz = edgeDictPrec[(xh,yh,z,xh,yh,zh)]
            cube.append((pxyz, pts2dir(xh,yh,z,xh,yh,zh,vxy), (+1,+1,0)))
            i += 1
        except:
            pass
        histo[i] += 1
        if i > 2:
            cntList.append(i)
            cubeList.append(cube)
    print(histo)
    return cubeList



def triangleNorm(cornerA, cornerB, cornerC):
    a = np.array(cornerA)
    b = np.array(cornerB)
    c = np.array(cornerC)
    n = np.cross(b-a,c-a)
    return n

def dirNorm(edgeA, edgeB, edgeC):
    a = np.array(edgeA)
    b = np.array(edgeB)
    c = np.array(edgeC)
    n = a + b + c
    return n

def isConvex(cornerA, cornerB, cornerC):
    nd = dirNorm(cornerA[1], cornerB[1], cornerC[1])
    nt = triangleNorm(cornerA[0], cornerB[0], cornerC[0])
    return np.dot(nd, nt) > 0


def isInnerEdge(pt1, pt2):
    d = np.array(pt2[2]) - np.array(pt1[2])
    #print('{} - {} - {} - {} - {}'.format(pt1,pt2,d,np.sum(np.abs(d)),np.sum(np.abs(d)) != 2))
    return np.sum(np.abs(d)) != 2

def findInnerEdges(pts):
    return [(i,k+i+1) for i, p1 in enumerate(pts) for k, p2 in
            enumerate(pts[(i+1):]) if isInnerEdge(p1, p2)]

def findOuterEdges(pts):
    #return [(i,k+i+1) for i, p1 in enumerate(pts) for k, p2 in
    #        enumerate(pts[(i+1):])]
    return [(i,k+i+1) for i, p1 in enumerate(pts) for k, p2 in
            enumerate(pts[(i+1):]) if not isInnerEdge(p1, p2)]


def repairOuterCirc(oe):
    x = oe
    y = [f for k in x for f in k]
    bc = np.bincount(y)
    idx = np.where(bc==1)[0]
    if len(idx) == 2:
        oe.append((idx[0], idx[1]))
    return oe

def findOuterCirc(pts):
    oe = findOuterEdges(pts)
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



#@jit(nopython=True,cache=True)
def cubes2vertices(cubes):
    vertices = []
    for cubev in cubes:
        cube = [e[0] for e in cubev]
        v = [e[1] for e in cubev]
        try:
            oc = findOuterCirc(cubev)
        except:
            print("outerCirc not closed!")
            continue
        for i in range(len(oc)-2):
            #vertices.append([cube[oc[0]], cube[oc[i+1]], cube[oc[i+2]]])
            #vertices.append([cube[oc[0]], cube[oc[i+2]], cube[oc[i+1]]])
            if isConvex(cubev[oc[0]], cubev[oc[i+1]], cubev[oc[i+2]]):
                vertices.append([cube[oc[0]], cube[oc[i+1]], cube[oc[i+2]]])
            else:
                vertices.append([cube[oc[0]], cube[oc[i+2]], cube[oc[i+1]]])
    return vertices


def renderAndSave(func, filename, res=1):
    t0 = time.time()
    ptsResDict = getSurface(func, None, res)
    print('getSurface time: {}'.format(time.time()-t0))
    print(len(ptsResDict))
    t0 = time.time()
    edgeResDict = pts2Edges(ptsResDict, res)
    print('pts2Edges time: {}'.format(time.time()-t0))
    print(len(edgeResDict))
    t0 = time.time()
    edgeResDictPrec = edgePrec(func, edgeResDict, res)
    #edgeResDictPrec = edgeResDict
    print('edgePrec time: {}'.format(time.time()-t0))
    print(len(edgeResDictPrec))
    t0 = time.time()
    cubeList = edge2cube(ptsResDict, edgeResDictPrec, res)
    print('edge2cube time: {}'.format(time.time()-t0))
    print(len(cubeList))
    #c = [sum([1 for e in cntList if e == n]) for n in range(12)]
    #print(c)
    t0 = time.time()
    vertices = cubes2vertices(cubeList)
    print('cubes2vertices time: {}'.format(time.time()-t0))
    print(len(vertices))

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




