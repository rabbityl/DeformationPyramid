//
// Created by liyang on 2022/3/8.
//

#pragma once

#include <iostream>
#include <tuple>

#include <torch/extension.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace image_proc {

    std::tuple< py::array_t<int>, py::array_t<int> > printArray(   );

    std::tuple< py::array_t<float>, py::array_t<int>, py::array_t<int> >  depthToMesh( const py::array_t<float>& pointImage, float maxTriangleEdgeDistance  );


}