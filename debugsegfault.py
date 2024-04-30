#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################

import pickle

from numba.typed import List

with open('circList.pkl', 'rb') as inp:
    circList = pickle.load(inp)

circList = List(circList)
poly_edge_list = [(e[(i+1)%len(e)], e[i]) for e in circList for i, f in enumerate(e)]




