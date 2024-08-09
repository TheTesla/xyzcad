import subprocess

import pytest
from numba import njit

from xyzcad import render


@njit
def sphere(x, y, z):
    return 20**2 > x**2 + y**2 + z**2


def test_render_sphere():
    render.renderAndSave(sphere, "build/sphere.stl", 0.5)
    rv = subprocess.run(["admesh", "build/sphere.stl"], stdout=subprocess.PIPE)
    lines = rv.stdout.decode(errors="ignore").split("\n")
    lc = [e for e in lines if ":" in e]
    dcs = {
        e.split(":")[0].strip(): e.split(":")[1].strip().split(" ", 1)[0] for e in lc
    }
    dc = {k: int(v) if v.isdigit() else v for k, v in dcs.items()}
    ld = [f.strip() for e in lines if "," in e and "=" in e for f in e.split(",")]
    dd = {
        e.split("=")[0].strip(): float(e.split("=")[1].strip().split(" ", 1)[0])
        for e in ld
    }
    stats = {**dc, **dd}
    assert stats["Max X"] > 19
    assert stats["Max X"] < 21
    assert stats["Min X"] < -19
    assert stats["Min X"] > -21
    assert stats["Max Y"] > 19
    assert stats["Max Y"] < 21
    assert stats["Min Y"] < -19
    assert stats["Min Y"] > -21
    assert stats["Max Z"] > 19
    assert stats["Max Z"] < 21
    assert stats["Min Z"] < -19
    assert stats["Min Z"] > -21
    assert stats["Number of parts"] == 1
    assert stats["Degenerate facets"] == 0
    assert stats["Edges fixed"] == 0
    assert stats["Facets removed"] == 0
    assert stats["Facets added"] == 0
    assert stats["Facets reversed"] == 0
    assert stats["Backwards edges"] == 0
