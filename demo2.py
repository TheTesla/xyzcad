#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


from numba import jit, njit
import math
from xyzcad import render
import time


@njit
def sphere(x,y,z):
    return 13**2 > x**2 + y**2 + z**2

t0 = time.time()

render.renderAndSave(sphere, 'demo.stl', 1)

print(time.time() - t0)
