#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension



class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, ".")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": ['-O3', '-Wall', '-shared', '-std=c++14', '-fPIC', '-fopenmp']}
    define_macros = []

    sources = [os.path.join(extensions_dir, s) for s in sources]

    eigen_include_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'eigen3'))

    include_dirs = [
        get_pybind_include(),
        get_pybind_include(user=True),
        eigen_include_dir,
        extensions_dir
    ]

    ext_modules = [
        extension(
            "MVRegC",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="MVRegC",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)