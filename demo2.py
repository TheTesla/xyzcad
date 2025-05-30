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
def sphere(p):
    x, y, z = p[:3]
    return 30**2 > x**2 + y**2 + z**2


t0 = time.time()

render.renderAndSave(sphere, "demo.stl", 0.1)

print(time.time() - t0)
