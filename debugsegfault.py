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

def repair_surface(poly_list):
    print("1")
    poly_edge_list = [(e[(i+1)%len(e)], e[i]) for e in poly_list for i, f in enumerate(e)]
    print("2")

def main():
    with open('circList.pkl', 'rb') as inp:
        circList = pickle.load(inp)

    circList = List(circList)
    repair_surface(circList)

main()



