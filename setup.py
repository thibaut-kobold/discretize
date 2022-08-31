#!/usr/bin/env python

"""discretize

Discretization tools for finite volume and inverse problems.
"""

import os
import sys

from setuptools import find_packages, setup

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

build_requires = [
    "numpy>=1.8",
    "Cython>=0.2",
    "setuptools_scm",
]

install_requires = [
    "numpy>=1.8",
    "scipy>=0.13",
]

metadata = dict(
    name="discretize",
    packages=find_packages(include=["discretize", "discretize.*"]),
    python_requires=">=3.7",
    # setup_requires=build_requires,
    install_requires=install_requires,
    author="SimPEG developers",
    author_email="rowanc1@gmail.com",
    description="Discretization tools for finite volume and inverse problems",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="finite volume, discretization, pde, ode",
    url="http://simpeg.xyz/",
    download_url="http://github.com/simpeg/discretize",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_scm_version={
        "write_to": os.path.join("discretize", "version.py"),
    },
)
if len(sys.argv) >= 2 and (
    "--help" in sys.argv[1:]
    or sys.argv[1] in ("--help-commands", "egg_info", "--version", "clean")
):
    # For these actions, build_requires are not required.
    #
    # They are required to succeed without Numpy/Cython, for example when
    # pip is used to install discretize when Numpy/Cython is not yet
    # present in the system.
    pass
else:
    import numpy as np
    from Cython.Build import cythonize
    from setuptools.extension import Extension

    ext_kwargs = {}
    if os.environ.get("DISC_COV", None) is not None:
        ext_kwargs["define_macros"] = [("CYTHON_TRACE_NOGIL", 1)]
        ext_kwargs["language_level"] = 3

    extensions = [
        Extension(
            "discretize._extensions.interputils_cython",
            ["discretize/_extensions/interputils_cython.pyx"],
            include_dirs=[np.get_include()],
            **ext_kwargs
        ),
        Extension(
            "discretize._extensions.tree_ext",
            ["discretize/_extensions/tree_ext.pyx", "discretize/_extensions/tree.cpp"],
            include_dirs=[np.get_include()],
            **ext_kwargs
        ),
        Extension(
            "discretize._extensions.simplex_helpers",
            ["discretize/_extensions/simplex_helpers.pyx"],
            include_dirs=[np.get_include()],
            **ext_kwargs
        ),
    ]

    metadata["ext_modules"] = cythonize(extensions)

setup(**metadata)
