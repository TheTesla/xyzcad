#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


import math
import time

from numba import jit, njit

from xyzcad import render


@njit
def sphere_clss(x, y, z):
    if not sphere(x, y, z):
        return 0
    k = 1
    if x > 0:
        k += 1
    if y > 0:
        k += 2
    if z > 0:
        k += 4
    return k


@njit
def sphere(x, y, z):
    return 3**2 > x**2 + y**2 + z**2


t0 = time.time()

render.renderAndSave(sphere_clss, "demo_clss", 1)

print(time.time() - t0)
