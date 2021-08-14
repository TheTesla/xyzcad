import numpy as np
import open3d as o3d
from stl import mesh
from numba import jit, prange
import time

import matplotlib.pyplot as plt





#@jit(nopython=True)
def f(x,y=None,z=None):
    if y is None:
        z = x[2]
        y = x[1]
        x = x[0]
    r = 3
    #return 1 if r**2 > ((1.+(x+3)/10)*(x+3)**2 + y**2 + z**2) else 0
    return 1 if r**2 > (x**2 + y**2 + z**2) else 0


#@jit(nopython=True)
def findSurfacePoint(fun, startPnt, direction, r, res):
    t = 0
    d = 1
    suc = 0
    dr = 0
    normDir = direction/np.sum(direction**2)
    for i in range(res):
        step = r/(2.**i)
        p = startPnt + t*normDir
        s = fun(p[0],p[1],p[2])
        if i == 0:
            so = s
        if 0 != s - so:
            dr = d * np.sign(s - so)
            suc = 1
            d *= -1
        t += step*d
        so = s
    return p, suc, dr

@jit(nopython=True)
def findNorm(fun, startPnt, d, r, res):
    ps = np.zeros((10,3))
    cr = np.zeros((10,3))
    i = 0
    for delta in np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]):
        p, suc, dr = findSurfacePoint(fun,startPnt+300000*r*delta/(2**res),d,r,res)
        if 1 == suc:
            ps[i] = p
            i += 1
    k = 0
    if i > 3:
        for a in range(i):
            for b in range(a):
                for c in range(b):
                    cr[k] = np.cross(-ps[a]+ps[b],-ps[a]+ps[c])
                    k += 1

        return np.sum(cr,axis=0), 1
    return np.zeros(3), 0

@jit(nopython=True)
def getPoints(fun,pnts,nrms):
    i = 0
    suc = 0
    dr = 0
    for x in np.arange(-10,10,0.5):
        for y in np.arange(-10,10,0.5):
            for z in np.arange(-10,10,0.5):
                for d in np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]]):
                    p, suc, dr = findSurfacePoint(f,np.array([x,y,z]),0.5*d,2,20)
                    if 1 == suc:
                        nrm, t = findNorm(f,np.array([x,y,z]),d,2,20)
                        if 1 == t:
                            nrms[i] = dr*nrm
                            pnts[i] = p
                            i += 1
    return i


p = np.zeros((100000,3), dtype=np.dtype('f8'))
nrm = np.zeros((100000,3), dtype=np.dtype('f8'))
t0 = time.time()
#n = getPoints(f,p,nrm)

sp = findSurfacePoint(f, np.array([-10,0,0]), np.array([1,0,0]),10, 1000)
print(sp)

p = sp[0]
r = 0.3
ptsList = [p]
ptsResDict = {}
for i in range(1000000):
    if len(ptsList) == 0:
        break
    p = ptsList.pop()
    p = np.floor(1000*p+0.5)/1000
    #print(p)
    s = sum([f(p+np.array([-r,0,0])), f(p+np.array([+r,0,0])),
             f(p+np.array([0,-r,0])), f(p+np.array([0,+r,0])),
             f(p+np.array([0,0,-r])), f(p+np.array([0,0,+r]))
        ])

    if s == 6 or s == 0:
        continue
    if p[0] not in ptsResDict:
        ptsResDict[p[0]] = {}
    if p[1] not in ptsResDict[p[0]]:
        ptsResDict[p[0]][p[1]] = {}
    if p[2] not in ptsResDict[p[0]][p[1]]:
        n = np.array([0,0,0])
        n[0] = f(p+np.array([-r,0,0])) - f(p+np.array([+r,0,0]))
        n[1] = f(p+np.array([0,-r,0])) - f(p+np.array([0,+r,0]))
        n[2] = f(p+np.array([0,0,-r])) - f(p+np.array([0,0,+r]))
        ptsResDict[p[0]][p[1]][p[2]] = n
    else:
        continue
    #print (f(p+np.array([+r,0,0])))
    #if f(p+np.array([-r,0,0])) == f(p+np.array([+r,0,0])):
    #    print('test')
    n = np.array([0,0,0])
    ptsList.append(p+np.array([-r,0,0]))
    ptsList.append(p+np.array([+r,0,0]))
    #if f(p+np.array([0,-r,0])) == f(p+np.array([0,+r,0])):
    ptsList.append(p+np.array([0,-r,0]))
    ptsList.append(p+np.array([0,+r,0]))
    #if f(p+np.array([0,0,-r])) == f(p+np.array([0,0,+r])):
    ptsList.append(p+np.array([0,0,-r]))
    ptsList.append(p+np.array([0,0,+r]))
#print(ptsList)
#print(ptsResDict)

ptsResArray = np.array([np.array([x,y,z,n[0],n[1],n[2]]) for x in ptsResDict.keys() for y in
    ptsResDict[x].keys() for z, n in
        ptsResDict[x][y].items()])

print(time.time() - t0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ptsResArray[:,:3])
n = len(ptsResArray)
pcd.colors = o3d.utility.Vector3dVector(np.array(n*[np.array([1,0,0])]))
pcd.normals = o3d.utility.Vector3dVector(ptsResArray[:,3:])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.2, linear_fit=False)[0]
mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
bbox = pcd.get_axis_aligned_bounding_box()
meshcrp = mesh.crop(bbox)

o3d.io.write_triangle_mesh("sphere3.stl", meshcrp)

ax = plt.axes(projection='3d')
ax.scatter(ptsResArray[:,0], ptsResArray[:,1], ptsResArray[:,2], s=0.1)
plt.show()




