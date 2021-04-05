import numpy as np
from stl import mesh
from numba import jit, prange
import time

def fun(x):
    return 1 if -1<x else 0

def findSurfacePoint(fun, startPnt, direction, r, res):
    t = 0
    d = 1
    normDir = direction/np.sum(direction**2)
    for i in range(res):
        step = r/(2.**i)
        p = startPnt + t*normDir
        s = fun(p)
        print("i={} step={} d={} p={} s={}".format(i,step,d,p,s))
        if i == 0:
            so = s
        if 0 != s - so:
            d *= -1
            pass
        else:
            pass
        t += step*d
        so = s
    return p


print(findSurfacePoint(fun, 0, 1, 10, 20))

