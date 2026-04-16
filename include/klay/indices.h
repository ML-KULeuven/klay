#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>

#include <klay/circuit.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace klay {

using Array = nb::ndarray<nb::numpy, long int, nb::shape<-1>>;
using Arrays = std::vector<Array>;

std::pair<Arrays, Arrays> get_indices(Circuit& c);

}  // namespace klay
