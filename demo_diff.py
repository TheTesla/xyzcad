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
def old_shape(x, y, z):
    if x > 2:
        return False
    if x < -2:
        return False
    if y > 2:
        return False
    if y < -2:
        return False
    if z > 10:
        return False
    if z < 0:
        return False
    return True


@njit
def new_shape(x, y, z):
    if x > 1:
        return False
    if x < -2:
        return False
    if y > 2:
        return False
    if y < -2:
        return False
    if z > 12:
        return False
    if z < 0:
        return False
    return True


@njit
def full_shape(x, y, z):
    return old_shape(x, y, z) or new_shape(x, y, z)


@njit
def shape_diff(x, y, z):
    if old_shape(x, y, z) and new_shape(x, y, z):
        return 1
    if old_shape(x, y, z) and not new_shape(x, y, z):
        return 2
    if not old_shape(x, y, z) and new_shape(x, y, z):
        return 3
    return 0


t0 = time.time()

render.renderAndSave(shape_diff, "demo_diff", 0.1)

print(time.time() - t0)
