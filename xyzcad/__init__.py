#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################################
#
#    xyzCad - functional cad software for 3d printing
#    Copyright (c) 2021 Stefan Helmert <stefan.helmert@t-online.de>
#
#######################################################################


import os

# VERSION
__version__ = "0.2.3"


if __name__ == '__main__':
    import git
    repo = git.Repo(search_parent_directories=False)
    version = '-'.join(repo.git.describe('--tags').split('-')[:2])
    if __version__ != version:
        print("Version in git repo changed!")
        print("  -> Change version in {} from {} to {}".format(__file__,
                __version__, version))
        __version__ = version
        with open(os.path.realpath(__file__), 'rt') as v:
            initfile = v.read()
            lines = initfile.splitlines()
            bg = [i for i, e in enumerate(lines) if 'VERSION' in e][0]
            lines[bg+1] = "__version__ = \"{}\"".format(__version__)
            initfile = '\n'.join(lines)
        with open(os.path.realpath(__file__), 'wt') as v:
            v.write(initfile)