#!/usr/bin/env python3

from xyzcad import render
from numba import jit, prange
import math
import numpy as np
import time

@jit
def screwprofile(x):
    x = x / (2*math.pi)
    return min(max(3*(x if x < 0.5 else 1-x), 0.3), 1.2)

@jit
def conv(f, k, s, r, xm, ym, zm):
    v = 0
    for xi in range(s[0]):
        x = r * (xi-(s[0]-1)/2)
        for yi in range(s[1]):
            y = r * (yi-(s[1]-1)/2)
            for zi in range(s[2]):
                z = r * (zi-(s[2]-1)/2)
                v += f(x+xm, y+ym, z+zm) * k[xi, yi, zi]
    return v

@jit
def f(x,y,z):
    rc = 5
    rf = 15
    zf = 30
    zn = 7
    yf = 15
    w = 75
    wi = 50
    rg = 10
    h = 40
    if x < -w:
        return False
    if x  > w:
        return False
    if y < -rc:
        return False
    if y > h:
        return False
    ang = +math.atan2((z-zf),x)
    r = 2*screwprofile((1.5*y+ang+math.pi)%(2*math.pi)) + (x**2 + (z-zf)**2)**0.5
    if r < rg + 0.1:
        return False
    if (rf**2 > (z-zf)**2 + x**2):
        return True
    if (rg**2 > (z-zf)**2 + (x+w/2)**2):
        return False
    if (rf**2 > (z-zf)**2 + (x+w/2)**2):
        return True
    if y - abs(z-zn) > 0:
        return False
    if (1 > (z/(zf-rf))**2 + (x/(wi))**2):
        return False
    if z > 0:
        if (1 > (z/(zf+rf))**2 + (x/w)**2):
            if y > -rc:
                if y < rc:
                    return True
#    if (rc**2 > z**2 + y**2):
#        return True
    return False


res = 0.1

k = np.ones((7,7,7), dtype=np.float64)
s = k.shape
n = 0
for xi in range(s[0]):
    xk = (xi-(s[0]-1)/2)
    for yi in range(s[1]):
        yk = (yi-(s[1]-1)/2)
        for zi in range(s[2]):
            zk = (zi-(s[2]-1)/2)
            tmp = 2**(-(xk**2 + yk**2 + zk**2))
            k[xi,yi,zi] = tmp
            n += tmp
k = k / n

@jit
def filtered(x,y,z):
    x = float(x)
    y = float(y)
    z = float(z)
    if 0.5 < conv(f, k, s, res, x, y, z):
    #if 0.5 < conv(f, k, res, x, y, z):
        return True
    return False


t0 = time.time()


render.renderAndSave(f, 'clamp.stl', res)

print(time.time() - t0)
