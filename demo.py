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
from xyzcad import render
import time


@jit(nopython=True,cache=True)
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

@jit(nopython=True,cache=True)
def g(x,y,z):
    if 50.1 > x**2 + y**2 + z**2:
        return True
    if 50.1 > (x-7)**2 + y**2 + z**2:
        return True
    if 50.1 > x**2 + (y-7)**2 + z**2:
        return True
    if 50.1 > x**2 + y**2 + (z-7)**2:
        return True
    return False

@jit(nopython=True,cache=True)
def h(x,y,z):
    if 12**2 > x**2 + y**2 + z**2:
        return True
    return False
t0 = time.time()

@jit
def k(x,y,z):
    prd = 15.
    r = (((x+0)%33.3 -16.67)**2 + ((y+16.67)%33.3 -16.67)**2)**0.5
    rd = r - prd
    base =12.**2> (x - ((x if x < 100. else 100.) if x > -100. else -100.))**2\
                 +(y - ((y if y <  50. else  50.) if y >  -50. else  -50.))**2\
                 +(z - ((z if z <   0. else   0.) if z >   -0. else   -0.))**2
    hole = (12**2 < rd**2 + z**2) and (prd**2 > r**2)
    return base and not hole

render.renderAndSave(f, 'demo.stl', 1)

print(time.time() - t0)
