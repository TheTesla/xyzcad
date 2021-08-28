FROM fedora:34

#RUN yum -y install python3-pip 
#
#RUN yum -y install git cmake
#
#RUN git clone --recursive https://github.com/intel-isl/Open3D
#
#RUN cd Open3D && git checkout v0.13.0 && git submodule update --init --recursive
#
#RUN cd Open3D && mkdir build
#
#RUN yum -y install g++
#
#RUN yum -y install python3-devel
#
#RUN yum -y install libcxx-devel
#
#RUN yum -y install SDL2-devel
#
#RUN yum -y install libXrandr-devel
#
#RUN yum -y install libXinerama-devel
#
#RUN yum -y install libXcursor-devel
#
## ^ cmake completes
#
#RUN yum -y install libcxxabi-devel tbb-devel libXi-devel ninja-build
#
#RUN yum -y install libX11-devel
#
#RUN yum -y install mesa-libGLU-devel
#
#
#
#RUN cd Open3D/build && cmake -DPYTHON_EXECUTABLE=/usr/bin/python3.9 -DCMAKE_INSTALL_PREFIX=/usr/lib/open3d -DCMAKE_CXX_FLAGS=-O ..
#
#RUN cd Open3D/build && make -j32
#
#RUN cd Open3D/build && make install
#
#RUN pip3 install wheel
#
#RUN cd Open3D/build && make install-pip-package
#

COPY fedora34-open3d-install.sh /

RUN chmod +x /fedora34-open3d-install.sh

RUN /fedora34-open3d-install.sh


CMD /bin/bash

