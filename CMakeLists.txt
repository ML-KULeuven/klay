cmake_minimum_required(VERSION 3.18...3.27)
project(klaycircuits LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT SKBUILD)
    message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# -----------------------------------------------
# DEPENDENCIES 
# -----------------------------------------------

include(GNUInstallDirs)

find_package(Python 3.10
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# -----------------------------------------------
# LIBRARY KLAY
# -----------------------------------------------

add_library(klay SHARED
  src/circuit.cpp
  src/node.cpp
)

target_include_directories(klay
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:klay/include>
)

install(TARGETS klay
    EXPORT klayTargets
    LIBRARY DESTINATION klay
)

install(DIRECTORY include/
    DESTINATION klay/include
)

install(EXPORT  klayTargets
    FILE        klayConfig.cmake
    DESTINATION klay/cmake/klay
)

# -----------------------------------------------
# PYTHON KLAY EXTENSION
# -----------------------------------------------

nanobind_add_module(klay_ext
  STABLE_ABI
  NOMINSIZE
  NB_STATIC
  src/klay_bindings.cpp
)

target_link_libraries(klay_ext PRIVATE klay)

set_target_properties(klay_ext PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS klay_ext LIBRARY DESTINATION klay)

# -----------------------------------------------
# PYTHON KLAY EXTENSION STUBS
# -----------------------------------------------

# nanobind_add_stub(
#   klay_ext_stub
#   INSTALL_TIME
#   MODULE klay_ext
#   OUTPUT klay/klay_ext.pyi
#   PYTHON_PATH "klay"
#   MARKER_FILE klay/py.typed
#   DEPENDS klay_ext
# )
# install(FILES "${CMAKE_BINARY_DIR}/klay_ext.pyi" DESTINATION klay)
