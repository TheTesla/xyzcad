#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - implicit surface function cad software for 3d printing
#    Copyright (c) 2021 - 2024 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


import numpy as np
import time
from numba import njit, objmode, prange, types
from numba.typed import List, Dict

from stl import mesh

tlt = [[[0]]] * 256
tlt[1] = [[0, 1, 2]]
tlt[254] = [[0, 2, 1]]
tlt[2] = [[2, 10, 11]]
tlt[253] = [[2, 11, 10]]
tlt[3] = [[0, 1, 10, 11]]
tlt[252] = [[0, 11, 10, 1]]
tlt[4] = [[1, 8, 9]]
tlt[251] = [[1, 9, 8]]
tlt[5] = [[0, 8, 9, 2]]
tlt[250] = [[0, 2, 9, 8]]
tlt[6] = [[1, 8, 9], [2, 10, 11]]
tlt[249] = [[9, 8, 1], [11, 10, 2]]
tlt[7] = [[0, 8, 9, 10, 11]]
tlt[248] = [[0, 11, 10, 9, 8]]
tlt[8] = [[3, 10, 9]]
tlt[247] = [[3, 9, 10]]
tlt[9] = [[0, 1, 2], [3, 10, 9]]
tlt[246] = [[2, 1, 0], [9, 10, 3]]
tlt[10] = [[2, 9, 3, 11]]
tlt[245] = [[2, 11, 3, 9]]
tlt[11] = [[0, 1, 9, 3, 11]]
tlt[244] = [[0, 11, 3, 9, 1]]
tlt[12] = [[1, 8, 3, 10]]
tlt[243] = [[1, 10, 3, 8]]
tlt[13] = [[0, 8, 3, 10, 2]]
tlt[242] = [[0, 2, 10, 3, 8]]
tlt[14] = [[1, 8, 3, 11, 2]]
tlt[241] = [[1, 2, 11, 3, 8]]
tlt[15] = [[0, 8, 3, 11]]
tlt[240] = [[0, 11, 3, 8]]
tlt[16] = [[0, 6, 7]]
tlt[239] = [[0, 7, 6]]
tlt[17] = [[1, 2, 6, 7]]
tlt[238] = [[1, 7, 6, 2]]
tlt[18] = [[0, 6, 7], [2, 10, 11]]
tlt[237] = [[7, 6, 0], [11, 10, 2]]
tlt[19] = [[1, 10, 11, 6, 7]]
tlt[236] = [[1, 7, 6, 11, 10]]
tlt[20] = [[0, 6, 7], [1, 8, 9]]
tlt[235] = [[7, 6, 0], [9, 8, 1]]
tlt[21] = [[2, 6, 7, 8, 9]]
tlt[234] = [[2, 9, 8, 7, 6]]
tlt[22] = [[2,10,11],[0,6,7],[1,8,9]]
tlt[233] = [[11,10,2],[0,7,6],[1,9,8]]
tlt[23] = [[6, 7, 8, 9, 10, 11]]
tlt[232] = [[11, 10, 9, 8, 7, 6]]
tlt[24] = [[3,10,9],[0,6,7]]
tlt[231] = [[3,9,10],[0,7,6]]
tlt[25] = [[1,2,6,7],[3,10,9]]
tlt[230] = [[7,6,2,1],[3,9,10]]
tlt[26] = [[2, 9, 3, 11],[0,6,7]]
tlt[229] = [[11, 3, 9, 2],[0,7,6]]
tlt[27] = [[1, 9, 3, 11, 6, 7]]
tlt[228] = [[7, 6, 11, 3, 9, 1]]
tlt[28] = [[1, 8, 3, 10],[0, 6, 7]]
tlt[227] = [[10, 3, 8, 1],[0, 7, 6]]
tlt[29] = [[0, 8, 3, 10, 2],[0, 6, 7]]
tlt[226] = [[2, 10, 3, 8, 0],[0, 7, 6]]
tlt[30] = [[0,6,7],[1, 8, 3, 11, 2]]
tlt[225] = [[0,7,6],[2, 11, 3, 8, 1]]
tlt[31] = [[3, 11, 6, 7, 8]]
tlt[224] = [[3, 8, 7, 6, 11]]
tlt[32] = [[6, 11, 4]]
tlt[223] = [[6, 4, 11]]
tlt[33] = [[0, 1, 2], [4, 6, 11]]
tlt[222] = [[2, 1, 0], [11, 6, 4]]
tlt[34] = [[2, 10, 4, 6]]
tlt[221] = [[2, 6, 4, 10]]
tlt[35] = [[0, 1, 10, 4, 6]]
tlt[220] = [[0, 6, 4, 10, 1]]
tlt[36] = [[4,6,11],[1,8,9]]
tlt[219] = [[4,11,6],[1,9,8]]
tlt[37] = [[4,6,11],[0,8,9,2]]
tlt[218] = [[4,11,6],[2,9,8,0]]
tlt[38] = [[2,10,4,6],[1,8,9]]
tlt[217] = [[6,4,10,2],[1,9,8]]

tlt[39] = [[0, 8, 9, 10, 4, 6]]
tlt[216] = [[6, 4, 10, 9, 8, 0]]

tlt[40] = [[4,6,11],[3,10,9]]
tlt[215] = [[4,11,6],[3,9,10]]

tlt[41] = [[0,1,2],[4,6,11],[3,10,9]]
tlt[214] = [[0,2,1],[4,11,6],[3,9,10]]

tlt[42] = [[2, 9, 3, 4, 6]]
tlt[213] = [[2, 6, 4, 3, 9]]

tlt[43] = [[0, 1, 9, 3, 4, 6]]
tlt[212] = [[6, 4, 3, 9, 1, 0]]

tlt[44] = [[6, 11, 4],[1,8,3,10]]
tlt[211] = [[6, 4, 11],[10,3,8,1]]

tlt[45] = [[0, 8, 3, 10, 2],[6,11,4]]
tlt[210] = [[2, 10, 3, 8, 0],[6,4,11]]

tlt[46] = [[1, 8, 3, 4, 6, 2]]
tlt[209] = [[2, 6, 4, 3, 8, 1]]

tlt[47] = [[0, 8, 3, 4, 6]]
tlt[208] = [[0, 6, 4, 3, 8]]

tlt[48] = [[0, 11, 4, 7]]
tlt[207] = [[0, 7, 4, 11]]

tlt[49] = [[1, 2, 11, 4, 7]]
tlt[206] = [[1, 7, 4, 11, 2]]

tlt[50] = [[0, 2, 10, 4, 7]]
tlt[205] = [[0, 7, 4, 10, 2]]

tlt[51] = [[1, 10, 4, 7]]
tlt[204] = [[1, 7, 4, 10]]

tlt[52] = [[0, 11, 4, 7],[1,8,9]]
tlt[203] = [[7, 4, 11, 0],[1,9,8]]

tlt[53] = [[4, 7, 8, 9, 2, 11]]
tlt[202] = [[11, 2, 9, 8, 7, 4]]

tlt[54] = [[0, 2, 10, 4, 7], [1,8,9]]
tlt[201] = [[0, 7, 4, 10, 2], [1,9,8]]

tlt[55] = [[4, 7, 8, 9, 10]]
tlt[200] = [[4, 10, 9, 8, 7]]

tlt[56] = [[0, 11, 4, 7],[3,10,9]]
tlt[199] = [[7, 4, 11, 0],[3,9,10]]

tlt[57] = [[1, 2, 11, 4, 7],[3,10,9]]
tlt[198] = [[7, 4, 11, 2, 1],[3,9,10]]

tlt[58] = [[0,2,9,3,5,7]]
tlt[197] = [[7,5,3,9,2,0]]

tlt[59] = [[1, 9, 3, 4, 7]]
tlt[196] = [[1, 7, 4, 3, 9]]

tlt[60] = [[0, 11, 4, 7],[1,8,3,10]]
tlt[195] = [[7, 4, 11, 0],[10,3,8,1]]

tlt[61] = [[4, 7, 8, 3],[2,11,10]]
tlt[194] = [[3, 8, 7, 4],[2,10,11]]

tlt[62] = [[4, 7, 8, 3],[0,2,1]]
tlt[193] = [[3, 8, 7, 4],[0,1,2]]

tlt[63] = [[3, 4, 7, 8]]
tlt[192] = [[3, 8, 7, 4]]

tlt[64] = [[5, 8, 7]]
tlt[191] = [[5, 7, 8]]

tlt[65] = [[0,1,2],[5, 8, 7]]
tlt[190] = [[0,2,1],[5, 7, 8]]

tlt[66] = [[2,10,11],[5, 8, 7]]
tlt[189] = [[2,11,10],[5, 7, 8]]

tlt[67] = [[0, 1, 10, 11],[5,8,7]]
tlt[188] = [[11, 10, 1, 0],[5,7,8]]

tlt[68] = [[1, 7, 5, 9]]
tlt[187] = [[1, 9, 5, 7]]

tlt[69] = [[0, 7, 5, 9, 2]]
tlt[186] = [[2, 9, 5, 7, 0]]

tlt[70] = [[1, 7, 5, 9],[2,10,11]]
tlt[185] = [[9, 5, 7, 1],[2,11,10]]

tlt[71] = [[7, 5, 9, 10, 11, 0]]
tlt[184] = [[0, 11, 10, 9, 5, 7]]

tlt[72] = [[5, 8, 7], [3,10,9]]
tlt[183] = [[5, 7, 8], [3,9,10]]

tlt[73] = [[5, 8, 7], [3,10,9],[0,1,2]]
tlt[182] = [[5, 7, 8], [3,9,10],[0,2,1]]

tlt[74] = [[2, 9, 3, 11],[5,7,8]]
tlt[181] = [[11, 3, 9, 2],[5,8,7]]

tlt[75] = [[0, 1, 9, 3, 11],[5,7,8]]
tlt[180] = [[11, 3, 9, 1, 0],[5,8,7]]

tlt[76] = [[1, 7, 5, 3, 10]]
tlt[179] = [[1, 10, 3, 5, 7]]

tlt[77] = [[7, 5, 3, 10, 2, 0]]
tlt[178] = [[0, 2, 10, 3, 5, 7]]

tlt[78] = [[1, 7, 5, 3, 11, 2]]
tlt[177] = [[2, 11, 3, 5, 7, 1]]

tlt[79] = [[0, 7, 5, 3, 11]]
tlt[176] = [[0, 11, 3, 5, 7]]

tlt[80] = [[0, 6, 5, 8]]
tlt[175] = [[0, 8, 5, 6]]

tlt[81] = [[1, 2, 6, 5, 8]]
tlt[174] = [[1, 8, 5, 6, 2]]

tlt[82] = [[0, 6, 5, 8],[2,10,11]]
tlt[173] = [[8, 5, 6, 0],[2,11,10]]

tlt[83] = [[5, 8, 1, 10, 11, 6]]
tlt[172] = [[6, 11, 10, 1, 8, 5]]

tlt[84] = [[0, 6, 5, 9, 1]]
tlt[171] = [[0, 1, 9, 5, 6]]

tlt[85] = [[2, 6, 5, 9]]
tlt[170] = [[2, 9, 5, 6]]

tlt[86] = [[5, 9, 10, 11, 6],[0,2,1]]
tlt[169] = [[6, 11, 10, 9, 5],[0,1,2]]

tlt[87] = [[6, 5, 9, 10, 11]]
tlt[168] = [[6, 11, 10, 9, 5]]

tlt[88] = [[0, 6, 5, 8],[3,10,9]]
tlt[167] = [[8, 5, 6, 0],[3,9,10]]

tlt[89] = [[1, 2, 6, 5, 8],[3,10,9]]
tlt[166] = [[1, 2, 6, 5, 8],[3,10,9]]

tlt[90] = [[0, 6, 5, 8],[2, 9, 3, 11]]
tlt[165] = [[8, 5, 6, 0],[11, 3, 9, 2]]

tlt[91] = [[11, 6, 5, 3],[1,9,8]]
tlt[164] = [[3, 5, 6, 11],[1,8,9]]

tlt[92] = [[6, 5, 3, 10, 1, 0]]
tlt[163] = [[0, 1, 10, 3, 5, 6]]

tlt[93] = [[2, 6, 5, 3, 10]]
tlt[162] = [[2, 10, 3, 5, 6]]

tlt[94] = [[0,2,1],[11, 6, 5, 3]]
tlt[161] = [[0,1,2],[3, 5, 6, 11]]

tlt[95] = [[3, 11, 6, 5]]
tlt[160] = [[3, 5, 6, 11]]

tlt[96] = [[4,6,11], [5,8,7]]
tlt[159] = [[4,11,6], [5,7,8]]

tlt[97] = [[4,6,11], [5,8,7],[0,1,2]]
tlt[158] = [[4,11,6], [5,7,8],[0,2,1]]

tlt[98] = [[2,10,4,6],[5,8,7]]
tlt[157] = [[6,4,10,2],[5,7,8]]

tlt[99] = [[0, 1, 10, 4, 6],[5,8,7]]
tlt[156] = [[6, 4, 10, 1, 0],[5,7,8]]

tlt[100] = [[1,7,5,9],[4,6,11]]
tlt[155] = [[9,5,7,1],[4,11,6]]

tlt[101] = [[0, 7, 5, 9, 2],[4,6,11]]
tlt[154] = [[2, 9, 5, 7, 0],[4,11,6]]

tlt[102] = [[5, 9, 10, 4],[7, 6, 2, 1]]
tlt[153] = [[4, 10, 9, 5],[1, 2, 6, 7]]

tlt[103] = [[5, 9, 10, 4],[0,7,6]]
tlt[152] = [[4, 10, 9, 5],[0,6,7]]

tlt[104] = [[4,6,11],[3,10,9],[5,8,7]]
tlt[151] = [[4,11,6],[3,9,10],[5,7,8]]

tlt[105] = [[3,4,5],[0,7,6],[1,9,8],[2,11,10]]
tlt[150] = [[3,5,4],[0,6,7],[1,8,9],[2,10,11]]

tlt[106] = [[9, 8, 7, 6, 2],[3,4,5]]
tlt[149] = [[2, 6, 7, 8, 9],[3,5,4]]

tlt[107] = [[3,4,5],[0,7,6],[1,9,8]]
tlt[148] = [[3,5,4],[0,6,7],[1,8,9]]

tlt[108] = [[3,4,5],[7,6,11,10,1]]
tlt[147] = [[3,5,4],[1,10,11,6,7]]

tlt[109] = [[3,4,5],[0,7,6],[2,11,10]]
tlt[146] = [[3,5,4],[0,6,7],[2,10,11]]

tlt[110] = [[3,4,5],[7,6,2,1]]
tlt[145] = [[3,5,4],[1,2,6,7]]

tlt[111] = [[3,4,5],[0,7,6]]
tlt[144] = [[3,5,4],[0,6,7]]

tlt[112] = [[0, 11, 4, 5, 8]]
tlt[143] = [[0, 8, 5, 4, 11]]

tlt[113] = [[1, 2, 11, 4, 5, 8]]
tlt[142] = [[8, 5, 4, 11, 2, 1]]

tlt[114] = [[2, 10, 4, 5, 8, 0]]
tlt[141] = [[0, 8, 5, 4, 10, 2]]

tlt[115] = [[1, 10, 4, 5, 8]]
tlt[140] = [[1, 8, 5, 4, 10]]

tlt[116] = [[11, 4, 5, 9, 1, 0]]
tlt[139] = [[0, 1, 9, 5, 4, 11]]

tlt[117] = [[2, 11, 4, 5, 9]]
tlt[138] = [[2, 9, 5, 4, 11]]

tlt[118] = [[0,2,1],[4, 5, 9, 10]]
tlt[137] = [[0,1,2],[4, 10, 9, 5]]

tlt[119] = [[4, 5, 9, 10]]
tlt[136] = [[4, 10, 9, 5]]

tlt[120] = [[3,4,5],[11,10,9,8,0]]
tlt[135] = [[3,5,4],[0,8,9,10,11]]

tlt[121] = [[3,4,5],[1,9,8],[2,11,10]]
tlt[134] = [[3,5,4],[1,8,9],[2,10,11]]

tlt[122] = [[3,4,5],[2,9,8,0]]
tlt[133] = [[3,5,4],[0,8,9,2]]

tlt[123] = [[3,4,5],[1,9,8]]
tlt[132] = [[3,5,4],[1,8,9]]

tlt[124] = [[11,10,1,0],[3,4,5]]
tlt[131] = [[0,1,10,11],[3,5,4]]

tlt[125] = [[3, 4, 5],[2,11,10]]
tlt[130] = [[3, 5, 4],[2,10,11]]

tlt[126] = [[0,2,1],[3, 4, 5]]
tlt[129] = [[0,1,2],[3, 5, 4]]

tlt[127] = [[3, 4, 5]]
tlt[128] = [[3, 5, 4]]





@njit(cache=True)
def round(x):
    return np.floor(10000.*x+0.5)/10000.

@njit(cache=True)
def getInitPnt(func, minVal=-1000., maxVal=+1000., resSteps=24):
    s0 = func(0.,0.,0.)
    for d in range(resSteps):
        for xi in range(2**d):
            x = (xi+0.5)/(2**d)*(maxVal-minVal)+minVal
            for yi in range(2**d):
                y = (yi+0.5)/(2**d)*(maxVal-minVal)+minVal
                for zi in range(2**d):
                    z = (zi+0.5)/(2**d)*(maxVal-minVal)+minVal
                    s = func(x,y,z)
                    if s != s0:
                        return x,y,z,0.,0.,0.
    return 0.,0.,0.,0.,0.,0.





@njit(cache=True)
def getSurfacePnt(func, p0, p1, resSteps=24):
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    x0 = p0[0]
    y0 = p0[1]
    z0 = p0[2]

    s0 = func(x0,y0,z0)
    u = 0
    d = +1
    for i in range(resSteps):
        x = x0 * (1-u) + x1 * u
        y = y0 * (1-u) + y1 * u
        z = z0 * (1-u) + z1 * u
        s = func(x,y,z)
        if s != s0:
            s0 = s
            d = -d
        u += d*1/2**i
    return (x,y,z)




@njit(cache=True)
def findSurfacePnt(func, minVal=-1000., maxVal=+1000., resSteps=24):
    ps = getInitPnt(func, minVal, maxVal, resSteps)
    p =  getSurfacePnt(func, (ps[0],ps[1],ps[2]), (ps[3],ps[4],ps[5]), resSteps)
    return p




@njit(cache=True)
def getSurface(func, startPnt, res=1.3):
    x,y,z = startPnt
    ptsList = List([(round(x-res/2), round(y-res/2), round(z-res/2),0,0)])
    cubeCornerValsDict = Dict()
    r = res
    while ptsList:
        x,y,z,d,c_val_old = ptsList.pop()
        xh = round(x+r)
        yh = round(y+r)
        zh = round(z+r)
        xl = round(x-r)
        yl = round(y-r)
        zl = round(z-r)

        if d == 1:
            v000 = 0 < (c_val_old & 16) # v100 old
            v100 = func(xh , y  , z )
            v010 = 0 < (c_val_old & 64) # v110 old
            v110 = func(xh , yh , z )
            v001 = 0 < (c_val_old & 32) # v101 old
            v101 = func(xh , y  , zh)
            v011 = 0 < (c_val_old & 128) # v111 old
            v111 = func(xh , yh , zh)
        elif d == -1:
            v000 = func(x  , y  , z )
            v100 = 0 < (c_val_old & 1) # v000 old
            v010 = func(x  , yh , z )
            v110 = 0 < (c_val_old & 4) # v010 old
            v001 = func(x  , y  , zh)
            v101 = 0 < (c_val_old & 2) # v001 old
            v011 = func(x  , yh , zh)
            v111 = 0 < (c_val_old & 8) # v011 old
        elif d == 2:
            v000 = 0 < (c_val_old & 4) # v010 old
            v100 = 0 < (c_val_old & 64) # v110 old
            v010 = func(x  , yh , z )
            v110 = func(xh , yh , z )
            v001 = 0 < (c_val_old & 8) # v011 old
            v101 = 0 < (c_val_old & 128) # v111 old
            v011 = func(x  , yh , zh)
            v111 = func(xh , yh , zh)
        elif d == -2:
            v000 = func(x  , y  , z )
            v100 = func(xh , y  , z )
            v010 = 0 < (c_val_old & 1) # v000 old
            v110 = 0 < (c_val_old & 16) # v100 old
            v001 = func(x  , y  , zh)
            v101 = func(xh , y  , zh)
            v011 = 0 < (c_val_old & 2) # v001 old
            v111 = 0 < (c_val_old & 32) # v101 old
        elif d == 4:
            v000 = 0 < (c_val_old & 2) # v001 old
            v100 = 0 < (c_val_old & 32) # v101 old
            v010 = 0 < (c_val_old & 8) # v011 old
            v110 = 0 < (c_val_old & 128) # v111 old
            v001 = func(x  , y  , zh)
            v101 = func(xh , y  , zh)
            v011 = func(x  , yh , zh)
            v111 = func(xh , yh , zh)
        elif d == -4:
            v000 = func(x  , y  , z )
            v100 = func(xh , y  , z )
            v010 = func(x  , yh , z )
            v110 = func(xh , yh , z )
            v001 = 0 < (c_val_old & 1) # v000 old
            v101 = 0 < (c_val_old & 16) # v100 old
            v011 = 0 < (c_val_old & 4) # v010 old
            v111 = 0 < (c_val_old & 64) # v110 old
        else:
            v000 = func(x  , y  , z )
            v100 = func(xh , y  , z )
            v010 = func(x  , yh , z )
            v110 = func(xh , yh , z )
            v001 = func(x  , y  , zh)
            v101 = func(xh , y  , zh)
            v011 = func(x  , yh , zh)
            v111 = func(xh , yh , zh)
        cVal = 128*v111+64*v110+32*v101+16*v100+8*v011+4*v010+2*v001+1*v000
        if cVal == 255 or cVal == 0:
            continue
        if (not (v100 and v110 and v101 and v111)) and \
                (v100 or v110 or v101 or v111):
            if not d == -1:
                if (xh,y,z) not in cubeCornerValsDict:
                    ptsList.append((xh,y,z,+1,cVal))
        if (not (v010 and v110 and v011 and v111)) and \
                (v010 or v110 or v011 or v111):
            if not d == -2:
                if (x,yh,z) not in cubeCornerValsDict:
                    ptsList.append((x,yh,z,+2,cVal))
        if (not (v001 and v101 and v011 and v111)) and \
                (v001 or v101 or v011 or v111):
            if not d == -4:
                if (x,y,zh) not in cubeCornerValsDict:
                    ptsList.append((x,y,zh,+4,cVal))
        if (not (v000 and v010 and v001 and v011)) and \
                (v000 or v010 or v001 or v011):
            if not d == 1:
                if (xl,y,z) not in cubeCornerValsDict:
                    ptsList.append((xl,y,z,-1,cVal))
        if (not (v000 and v100 and v001 and v101)) and \
                (v000 or v100 or v001 or v101):
            if not d == 2:
                if (x,yl,z) not in cubeCornerValsDict:
                    ptsList.append((x,yl,z,-2,cVal))
        if (not (v000 and v100 and v010 and v110)) and \
                (v000 or v100 or v010 or v110):
            if not d == 4:
                if (x,y,zl) not in cubeCornerValsDict:
                    ptsList.append((x,y,zl,-4,cVal))
        cubeCornerValsDict[(x,y,z)] = np.uint8(cVal)

    return cubeCornerValsDict

@njit(cache=True,parallel=True)
def convert_corners2pts(cubeCornerValsDict, r):

    ptsResDict = Dict()

    for k, v in cubeCornerValsDict.items():
        x, y, z = k
        xh = round(x+r)
        yh = round(y+r)
        zh = round(z+r)
        ptsResDict[(x,  y, z)] = 0 < (v & 1) #v000
        ptsResDict[(xh, y, z)] = 0 < (v & 16) #v100
        ptsResDict[(x, yh, z)] = 0 < (v & 4) #v010
        ptsResDict[(x, y, zh)] = 0 < (v & 2) #v001
        ptsResDict[(xh, y, zh)] = 0 < (v & 32) #v101
        ptsResDict[(x, yh, zh)] = 0 < (v & 8) #v011
        ptsResDict[(xh, yh, z)] = 0 < (v & 64) #v110
        ptsResDict[(xh, yh, zh)] = 0 < (v & 128) #v111


    # worked in implementation, when convert_corners2pts() was part of
    # getSurface(), but this variant doesn't work good in this separated
    # implementation, it produces missing datapoints:
    #cubesList = list(set(cubeCornerValsDict.keys()))
    # didn't work in implementation, when convert_corners2pts() was part of
    # getSurface(), but this variant works in this separated implementation:
    #cubesList = list(cubeCornerValsDict.keys())

    # much more mess:
    cubesList = list(set(list(set(list(cubeCornerValsDict.keys())))))
    #cubesList = list(set(list(set(list(set(list(cubeCornerValsDict.keys())))))))

    print(f"len(cubesList)={len(cubesList)}")
    cubesArray = np.asarray(cubesList)
    ptCoordDictKeys = np.asarray(list(ptsResDict.keys()))
    ptCoordDictVals = np.asarray(list(ptsResDict.values()))
    cvArr = np.zeros(cubesArray.shape[0],dtype=np.uint8)
    for i in prange(cubesArray.shape[0]):
        c = cubesArray[i]
        cvArr[i] = cubeCornerValsDict[(c[0],c[1],c[2])]
    return cubesArray, ptCoordDictKeys, ptCoordDictVals, cvArr


@njit(cache=True,parallel=True)
def coords2relations(cubeCoordArray, ptCoordArray, ptValueArray, res):
    r = res

    arr_split = 16
    spl_dict = [{(0., 0., 0.):0}]*arr_split
    for k in range(arr_split):
        spl_dict[k].clear()
    for k in prange(arr_split):
        len_arr = int(ptCoordArray.shape[0]/arr_split + 1.0)
        splitted_arr = ptCoordArray[(len_arr*k):min(len_arr*(k+1),ptCoordArray.shape[0])]
        spl_dict[k] = {(e[0], e[1], e[2]): i+len_arr*k for i, e in enumerate(splitted_arr)}

    ptCoordDictRev = spl_dict[0]
    for k in range(1,arr_split):
        ptCoordDictRev.update(spl_dict[k])

    cube2ptIdxArray = np.zeros((cubeCoordArray.shape[0],8),dtype='int')
    for i in prange(cubeCoordArray.shape[0]):
        p = cubeCoordArray[i]
        x, y, z = p
        xh = round(x+r)
        yh = round(y+r)
        zh = round(z+r)
        cube2ptIdxArray[i] = [  ptCoordDictRev[(x, y, z )],
                                ptCoordDictRev[(xh,y, z )],
                                ptCoordDictRev[(x, yh,z )],
                                ptCoordDictRev[(xh,yh,z )],
                                ptCoordDictRev[(x, y, zh)],
                                ptCoordDictRev[(xh,y, zh)],
                                ptCoordDictRev[(x, yh,zh)],
                                ptCoordDictRev[(xh,yh,zh)]]


    cEdgeArray = np.zeros((cube2ptIdxArray.shape[0]*12,2),dtype='int')
    for i in prange(cube2ptIdxArray.shape[0]):
        cube = cube2ptIdxArray[i]
        cEdgeArray[12*i+ 0] = (cube[0], cube[1])
        cEdgeArray[12*i+ 1] = (cube[0], cube[2])
        cEdgeArray[12*i+ 2] = (cube[0], cube[4])
        cEdgeArray[12*i+ 3] = (cube[6], cube[7])
        cEdgeArray[12*i+ 4] = (cube[5], cube[7])
        cEdgeArray[12*i+ 5] = (cube[3], cube[7])
        cEdgeArray[12*i+ 6] = (cube[1], cube[5])
        cEdgeArray[12*i+ 7] = (cube[1], cube[3])
        cEdgeArray[12*i+ 8] = (cube[2], cube[3])
        cEdgeArray[12*i+ 9] = (cube[2], cube[6])
        cEdgeArray[12*i+10] = (cube[4], cube[6])
        cEdgeArray[12*i+11] = (cube[4], cube[5])
    cEdgesSet = set([(e[0], e[1]) for e in cEdgeArray])

    edge2ptIdxArray = np.asarray(list(cEdgesSet))

    edge2ptIdxDict = {(e[0], e[1]): i for i, e in enumerate(edge2ptIdxArray)}

    cube2edgeIdxArray = np.zeros((cube2ptIdxArray.shape[0],12),dtype='int')

    for i in prange(len(cube2ptIdxArray)):
        cube = cube2ptIdxArray[i]
        cube2edgeIdxArray[i] = [edge2ptIdxDict[(cube[0], cube[1])],
                                edge2ptIdxDict[(cube[0], cube[2])],
                                edge2ptIdxDict[(cube[0], cube[4])],
                                edge2ptIdxDict[(cube[6], cube[7])],
                                edge2ptIdxDict[(cube[5], cube[7])],
                                edge2ptIdxDict[(cube[3], cube[7])],
                                edge2ptIdxDict[(cube[1], cube[5])],
                                edge2ptIdxDict[(cube[1], cube[3])],
                                edge2ptIdxDict[(cube[2], cube[3])],
                                edge2ptIdxDict[(cube[2], cube[6])],
                                edge2ptIdxDict[(cube[4], cube[6])],
                                edge2ptIdxDict[(cube[4], cube[5])]]

    return (cube2ptIdxArray, cube2edgeIdxArray, edge2ptIdxArray, ptCoordArray,
            ptValueArray)


@njit(cache=True)
def cutCedgeIdx(edge2ptIdxList, ptValueList):
    return np.asarray([i for i, e in enumerate(edge2ptIdxList) if ptValueList[e[0]]
            != ptValueList[e[1]]])

@njit(cache=True,parallel=True)
def precTrPnts(func, cutCedgeIdxArray, edge2ptIdxArray, ptCoordArray):
    lcceil = len(cutCedgeIdxArray)
    r = np.zeros((lcceil,3))
    for i in prange(lcceil):
        p0, p1 = edge2ptIdxArray[cutCedgeIdxArray[i]]
        r[i] = getSurfacePnt(func, ptCoordArray[p0], ptCoordArray[p1])
    return r


@njit(cache=True)
def calcTrianglesCor(corCircList, invertConvexness=False):
    trList = List()
    if invertConvexness:
        for circ in corCircList:
            n = len(circ)
            trInCubeList = [(circ[0], circ[i+1], circ[i+2]) \
                                for i in range(n-2)]
            trList.extend(trInCubeList)
    else:
        for circ2 in corCircList:
            n = len(circ)
            trInCubeList2 = [(circ2[0], circ2[i2+2], circ2[i2+1]) \
                                for i2 in range(n-2)]
            trList.extend(trInCubeList2)
    return np.asarray([[[p[0],p[1],p[2]] for p in c] for c in trList])



@njit(cache=True)
def TrIdx2TrCoord(trList, cutCedgeIdxList, precTrPnts):
    cutCedgeIdxRevDict = {e: i for i, e in enumerate(cutCedgeIdxList)}
    return List([[precTrPnts[cutCedgeIdxRevDict[f]] for f in e if f in cutCedgeIdxRevDict] for e in trList])

@njit(cache=True)
def filter_single_edge(poly_edge_list):
    single_edge_set = set()
    for e in poly_edge_list:
        if e not in single_edge_set:
            if (e[1], e[0]) not in single_edge_set:
                single_edge_set.add(e)
            else:
                single_edge_set.remove((e[1], e[0]))
    return single_edge_set

@njit(cache=True)
def build_repair_polygons(single_edge_dict):
    ac = List()
    while len(single_edge_dict) > 0:
        f = List()
        e = List(single_edge_dict.keys())[0]
        while e in single_edge_dict:
            en = single_edge_dict[e]
            f.append(e)
            del single_edge_dict[e]
            e = en
        ac.append(f)
    return ac


@njit(cache=True)
def repair_surface(poly_list):
    poly_edge_list = List([(e[(i+1)%len(e)], e[i]) for e in poly_list for i, f
                           in enumerate(e)])
    singleEdgeSet = filter_single_edge(poly_edge_list)
    singleEdgeDict = {k: v for k, v in singleEdgeSet}
    ac = build_repair_polygons(singleEdgeDict)
    return ac

@njit(cache=True)
def calc_polygons(c2e, cvList, tlta):
    return List([List([c2e[i][k] for k in t]) for i, c in enumerate(cvList) for t in
                 tlta[c]])

@njit(cache=True)
def calc_closed_surface(c2e, cvList, tlta):
    circList = calc_polygons(c2e, cvList, tlta)
    #circList2 = List(circList)
    circList2 = circList
    corCircList = circList2
    rep = repair_surface(circList2)
    # corCircList.extend(List(rep))
    corCircList.extend(rep)
    return corCircList


@njit(cache=True)
def all_njit_func(func, res, tlt):
    with objmode(time0='f8'):
        time0 = time.perf_counter()
    p = findSurfacePnt(func)
    with objmode():
        print('findSurfacePnt time: {}'.format(time.perf_counter() - time0))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    corners = getSurface(func, p, res)
    print(f"len(corners)={len(corners)}")
    with objmode():
        print('getSurface time: {}'.format(time.perf_counter() - time1))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    cubesArray, ptsKeys, ptsVals, cvList = convert_corners2pts(corners, res)
    print(f"len(cubesArray, ptsKeys, ptsVals, cvList)={len(cubesArray)}, "
          +f"{len(ptsKeys)}, {len(ptsVals)},{len(cvList)}")
    with objmode():
        print('convert_corners2pts time: {}'.format(time.perf_counter() - time1))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    c2p, c2e, e2p, pc, pv = coords2relations(cubesArray, ptsKeys, ptsVals, res)
    print(f"len(c2p, c2e, e2p, pc, pv)={len(c2p)}, {len(c2e)}, {len(e2p)}, "
          +f"{len(pc)}, {len(pv)}")
    with objmode():
        print('coords2relations time: {}'.format(time.perf_counter() - time1))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    cCeI = cutCedgeIdx(e2p, pv)
    print(f"len(cCeI)={len(cCeI)}")
    with objmode():
        print('cutCedgeIdx time: {}'.format(time.perf_counter() - time1))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    precTrPtsList = precTrPnts(func, cCeI, e2p, pc)
    print(f"len(precTrPtsList)={len(precTrPtsList)}")
    with objmode():
        print('precTrPnts time: {}'.format(time.perf_counter() - time1))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    corCircList = calc_closed_surface(c2e, cvList, tlt)
    print(f"len(corCircList)={len(corCircList)}")
    with objmode():
        print('calc_closed_surface time: {}'.format(time.perf_counter() - time1))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    circPtsCoordList = TrIdx2TrCoord(corCircList, cCeI, precTrPtsList)
    print(f"len(circPtsCoordList)={len(circPtsCoordList)}")
    with objmode():
        print('TrIdx2TrCoord time: {}'.format(time.perf_counter() - time1))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    verticesArray = calcTrianglesCor(circPtsCoordList, True)
    print(f"len(verticesArray)={len(verticesArray)}")
    with objmode():
        print('calcTrianglesCor time: {}'.format(time.perf_counter() - time1))
    with objmode():
        print('all_njitINTERN time: {}'.format(time.perf_counter() - time0))
    return verticesArray


def renderAndSave(func, filename, res=1):
    t0 = time.time()
    tlt_L = [List(e) for e in tlt]
    verticesArray = all_njit_func(func, res, tlt_L)
    print('all_njit_func time: {}'.format(time.time()-t0))

    t0 = time.time()
    solid = mesh.Mesh(np.zeros(verticesArray.shape[0], dtype=mesh.Mesh.dtype))
    solid.vectors[:] = verticesArray
    print('to mesh time: {}'.format(time.time()-t0))
    t0 = time.time()
    solid.save(filename)
    print('save time: {}'.format(time.time()-t0))





