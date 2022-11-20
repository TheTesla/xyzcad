#!/usr/bin/env python3

from xyzcad import render
from numba import jit
import math
import numpy as np

@jit
def screwprofile(x):
    x = x / (2*math.pi)
    return min(max(3*(x if x < 0.5 else 1-x), 0.3), 1.2)


@jit
def f(x,y,z):
    l = 170
    rg = 10
    ra = 10
    rr = 14
    rf = rg+3
    hh = 12
    rgf = rg -4
    lgf = 11
    if z < 0:
        return False
    if z < hh:
        r = (x**2 + y**2)**0.5
        if r < rr - (hh - z):
            return True
        phi = math.atan2(y, x)
        if r * math.cos(((phi*180/math.pi) %60 -30)/180*math.pi) < ra:
            return True
        else:
            return False
    if z > l:
        r = (x**2 + y**2)**0.5
        if r > rg - (z-l):
            return False

    if z > l+3:
        return False
    if z > l - lgf and rgf**2 < x**2 + y**2:
        return False

    ang = -math.atan2(y,x)
    r = 2*screwprofile((1.5*z+ang+math.pi)%(2*math.pi)) + (x**2 + y**2)**0.5
    if r < rg:
        return True


    return False

render.renderAndSave(f, 'screw6longclamp.stl', 0.1)

