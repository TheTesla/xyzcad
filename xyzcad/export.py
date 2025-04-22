
import numpy as np
from numba import njit, objmode, prange, types
from numba.typed import Dict, List
from stl import mesh



#@njit(cache=True)
def poly2triangle_idx(polygons):
    tr_arr = np.zeros((len(polygons) * 8, 3), dtype=np.uint64)
    c = 0
    poly = np.zeros((6))
    for k in range(len(polygons)):
        poly = polygons[k]
        n = len(poly)
        for i in range(n - 2):
            tr_arr[c][0] = poly[0]
            tr_arr[c][1] = poly[i + 1]
            tr_arr[c][2] = poly[i + 2]
            c += 1
    return tr_arr[:c]

@njit(cache=True)
def poly2triangle_coord(polygons, vertices):
    polys_coord = [[vertices[idx] for idx in p] for p in polygons]
    tr_arr = np.zeros((len(polys_coord) * 8, 3, 3))
    c = 0
    poly = np.zeros((6, 3))
    for k in range(len(polys_coord)):
        poly = polys_coord[k]
        n = len(poly)
        for i in range(n - 2):
            tr_arr[c][0] = poly[0]
            tr_arr[c][1] = poly[i + 1]
            tr_arr[c][2] = poly[i + 2]
            c += 1
    return tr_arr[:c]


def export_stl(stl_filename, vertices, poly):
    verticesArray = poly2triangle_coord(poly, vertices)
    solid = mesh.Mesh(np.zeros(verticesArray.shape[0], dtype=mesh.Mesh.dtype))
    solid.vectors[:] = verticesArray
    solid.save(stl_filename+".stl")


def write_obj(obj_filename, vertices, poly_grpd, mtl_filename="", mtl_lst=[]):
    if mtl_filename == "":
        mtl_filename = obj_filename
    with open(obj_filename+".obj", 'w') as obj_file:
        obj_file.write(f"mtllib {mtl_filename}.mtl\n")
        for v in vertices:
            obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for i, faces in enumerate(poly_grpd):
            mtl_name = mtl_lst[i] if i < len(mtl_lst) else f"mtl{i}"
            obj_file.write(f"usemtl {mtl_name}\n")
            for f in faces:
                obj_file.write(f"f {' '.join(str(np.uint64(k+1)) for k in f)}\n")


def write_mtl(mtl_filename, color_mat, mtl_lst=[]):
    mtl_content = ""
    for i, c in enumerate(color_mat):
        mtl_name = mtl_lst[i] if i < len(mtl_lst) else f"mtl{i}"
        mtl_content += f"newmtl {mtl_name}\n"
        mtl_content += f"Ka {c[0]} {c[1]} {c[2]}\n"
        if len(c) == 3:
            mtl_content += f"Kb {c[0]} {c[1]} {c[2]}\n"
            mtl_content += f"Ks {c[0]} {c[1]} {c[2]}\n"
            mtl_content += f"d 1.0\n"
            mtl_content += f"illum 1.0\n"
        elif len(c) == 5:
            mtl_content += f"Kb {c[0]} {c[1]} {c[2]}\n"
            mtl_content += f"Ks {c[0]} {c[1]} {c[2]}\n"
            mtl_content += f"d {c[3]}\n"
            mtl_content += f"illum {c[4]}\n"
        elif len(c) == 9:
            mtl_content += f"Kb {c[3]} {c[4]} {c[5]}\n"
            mtl_content += f"Ks {c[6]} {c[7]} {c[8]}\n"
            mtl_content += f"d 1.0\n"
            mtl_content += f"illum 1.0\n"
        elif len(c) == 11:
            mtl_content += f"Kb {c[3]} {c[4]} {c[5]}\n"
            mtl_content += f"Ks {c[6]} {c[7]} {c[8]}\n"
            mtl_content += f"d {c[9]}\n"
            mtl_content += f"illum {c[10]}\n"
    with open(mtl_filename+".mtl", 'w') as mtl_file:
        mtl_file.write(mtl_content)

def create_gen_color(n):
    color_mat = np.zeros((n,3))
    for i in range(n):
        color_mat[i,2] = 1/2 * (int(i/1) % 2) + 1/4 * (int(i/8 ) % 2) + 1/8 * \
        (int(i/64) % 2)
        color_mat[i,1] = 1/2 * (int(i/2) % 2) + 1/4 * (int(i/16) % 2) + 1/8 * \
        (int(i/128) % 2)
        color_mat[i,0] = 1/2 * (int(i/4) % 2) + 1/4 * (int(i/32) % 2) + 1/8 * \
        (int(i/256) % 2)
    color_mat[0] = 3/4, 3/4, 3/4
    return color_mat

def export_wavefront_obj_simple(obj_filename, vertices, poly_grpd):
    write_mtl(obj_filename, create_gen_color(len(poly_grpd)))
    write_obj(obj_filename, vertices, poly_grpd)

def export_obj_printable(obj_filename, vertices, poly_grpd):
    export_wavefront_obj_simple(obj_filename, vertices, \
            [[] if len(e) == 0 else poly2triangle_idx(e) for e in poly_grpd])


