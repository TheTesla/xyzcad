#!/usr/bin/env python3

import setuptools
import xyzcad


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xyzcad",
    version=xyzcad.__version__,
    author="Stefan Helmert",
    author_email="stefan.helmert@t-online.de",
    description="Software renders f(x,y,z) into a printable STL.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheTesla/xyzcad",
    packages=setuptools.find_packages(exclude=['test*']),
    #include_package_data=True,
    install_requires=[
        "numpy",
        "numba",
        "open3d"
    ],
    license = 'https://www.fsf.org/licensing/licenses/agpl-3.0.html',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)

