#include <pybind11/pybind11.h>
namespace py = pybind11;


#include "layerize.h"

PYBIND11_MODULE(__lib, m) {

    m.def("brr", &brr);

}
