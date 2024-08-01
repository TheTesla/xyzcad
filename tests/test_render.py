import pytest
import admesh
from numba import njit
from xyzcad import render

@njit
def sphere(x, y, z):
    return 20**2 > x**2 + y**2 + z**2

def test_render_sphere():
    render.renderAndSave(sphere, "build/sphere.stl", 0.5)
    stl = admesh.Stl('build/sphere.stl')
    assert stl.stats["max"]["x"] > 19
    assert stl.stats["max"]["x"] < 21
    assert stl.stats["min"]["x"] < -19
    assert stl.stats["min"]["x"] > -21
    assert stl.stats["max"]["y"] > 19
    assert stl.stats["max"]["y"] < 21
    assert stl.stats["min"]["y"] < -19
    assert stl.stats["min"]["y"] > -21
    assert stl.stats["max"]["z"] > 19
    assert stl.stats["max"]["z"] < 21
    assert stl.stats["min"]["z"] < -19
    assert stl.stats["min"]["z"] > -21




