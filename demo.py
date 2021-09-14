#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


from numba import jit
import math
from xyzcad import render2
import time


@jit(nopython=True)
def f(x,y,z):
    r = 70/2
    rb = 34/2
    hb = 40
    h = 50
    hbl = 30
    hm = 5
    mh = r - 4
    mhr = 3/2
    if z < 0:
        return 0
    ang = math.atan2(x,y)/math.pi*180 + 180
    rad = math.sqrt(x**2 + y**2)
    rkh = z/h
    rk = (1-rkh) * 0 + rkh * (math.cos((4*(ang)+180)/180*math.pi)/2+0.5)
    if rb**2 > ((x)**2 + (y)**2 + (z*rb**2/hb)):
        return 1
    if z > h:
        return 0
    if z < hm:
        if x < r and x > -r:
            if y < r and y > -r:
                if (x+mh)**2 + (y+mh)**2 < mhr**2:
                    return 0
                if (x-mh)**2 + (y+mh)**2 < mhr**2:
                    return 0
                if (x-mh)**2 + (y-mh)**2 < mhr**2:
                    return 0
                if (x+mh)**2 + (y-mh)**2 < mhr**2:
                    return 0
                if x**2 + y**2 > r**2:
                    return 1
    if z < hbl:
        if rad < (rk * 2**0.5 + (1-rk) * 1)**0.5 * r:
            if (ang + 360 - z*2) % 120 < 10:
                return 1
    rr = (x**2 + y**2) #* (rk / 2**0.5 + (1-rk) / 1)
    if rr < (rk * 2**0.5 + (1-rk) * 1) * r**2:
        if rr > (rk * 2**0.5 + (1-rk) * 1) * (r-3)**2:
            return 1
    return 0

t0 = time.time()

render2.renderAndSave(f, 'demo.stl', 0.1)

print(time.time() - t0)
