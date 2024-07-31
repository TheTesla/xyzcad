import numpy as np

# import open3d as o3d
import time
from numba import jit, prange, types
from numba.typed import List, Dict


@jit(nopython=True, cache=True)
def revDict(d):
    return {v: k for k, v in d.items()}


@jit(nopython=True, cache=True, parallel=True)
def toDict(r):
    n = 1
    dl = List()
    d = Dict.empty(key_type=types.float64, value_type=types.float64)
    for k in range(n):
        dl.append(Dict.empty(key_type=types.float64, value_type=types.float64))
    for k in prange(n):
        for i in range(r.shape[0] / n):
            i += int(k * r.shape[0] / n)
            dl[k][r[i, 0]] = r[i, 1]
    # for k in range(n):
    #    d.update(dl[k])
    return {**dl[0]}


t0 = time.time()
r = np.random.rand(10**7, 2)
print(f"random time: {time.time()-t0}")
t0 = time.time()
# d = dict(r)
# d = Dict.empty(key_type=types.float64, value_type=types.float64)
# for i in range(r.shape[0]):
#    d[r[i,0]] = r[i,1]
d = toDict(r)

print(f"dict() time: {time.time()-t0}")


t0 = time.time()
y = revDict(d)
print(f"revDict() time: {time.time()-t0}")
print(len(y))
