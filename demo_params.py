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
    x, y, z, params = p
    a, b, c, r = params
    return r**2 > (a * x) ** 2 + (b * y) ** 2 + (c * z) ** 2


t0 = time.time()

for a in range(8):
    render.renderAndSave(
        sphere, f"build/demo_params_{a}", 0.1, (1.0 + a, 5.0, 3.0, 20.0)
    )

print(time.time() - t0)
