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
def old_shape(p):
    x, y, z = p[:3]
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
def new_shape(p):
    x, y, z = p[:3]
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
def full_shape(p):
    return old_shape(p) or new_shape(p)


@njit
def shape_diff(p):
    if old_shape(p) and new_shape(p):
        return 1
    if old_shape(p) and not new_shape(p):
        return 2
    if not old_shape(p) and new_shape(p):
        return 3
    return 0


t0 = time.time()

render.renderAndSave(shape_diff, "demo_diff", 0.1)

print(time.time() - t0)
