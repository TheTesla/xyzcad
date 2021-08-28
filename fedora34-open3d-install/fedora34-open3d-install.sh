#!/bin/bash

yum -y install python3-pip git cmake g++ python3-devel libcxx-devel SDL2-devel libXrandr-devel libXinerama-devel libXcursor-devel libcxxabi-devel tbb-devel libXi-devel ninja-build libX11-devel mesa-libGLU-devel

pip3 install wheel

git clone --recursive https://github.com/intel-isl/Open3D

cd Open3D

git checkout v0.13.0

git submodule update --init --recursive

mkdir build

cd build

cmake -DPYTHON_EXECUTABLE=/usr/bin/python3.9 -DCMAKE_INSTALL_PREFIX=/usr/lib/open3d -DCMAKE_CXX_FLAGS=-O ..

make -j32

make install

make install-pip-package


