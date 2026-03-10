cmake_minimum_required(VERSION 3.18...3.27)
project(klaycircuits LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)

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
        $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------
# INSTALL LIBRARY KLAY (build tree)
# -----------------------------------------------
# FIXME(Ibrahim):
# https://stackoverflow.com/a/66412641/14633299

set(KLAY_CONFIG_FILENAME "klayConfig.cmake")
set(KLAY_TARGETS_FILENAME "klayTargets.cmake")
configure_file(klayConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${KLAY_CONFIG_FILENAME}" @ONLY)

export(
    TARGETS klay
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${KLAY_TARGETS_FILENAME}"
)
export(PACKAGE klay)

# -----------------------------------------------
# INSTALL LIBRARY KLAY (install)
# -----------------------------------------------
# NOTE(Ibrahim):
# to install lbikay system wide

# set(KLAY_EXPORT_NAME "klayTargets")
# set(DEF_INSTALL_CMAKE_DIR lib/cmake/klay)
# set(KLAY_INSTALL_CMAKE_DIR
#   ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
#   "Installation directory for klay CMake files")

# install(
#     TARGETS klay
#     EXPORT ${KLAY_EXPORT_NAME}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

# install(DIRECTORY include/klay DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# # Install-tree export
# set(CONF_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/klay")
# configure_file(klayConfig.cmake.in
#     "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${KLAY_CONFIG_FILENAME}" @ONLY)
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${KLAY_CONFIG_FILENAME}"
#     DESTINATION "${KLAY_INSTALL_CMAKE_DIR}"
# )
# install(
#     EXPORT ${KLAY_EXPORT_NAME}
#     DESTINATION "${KLAY_INSTALL_CMAKE_DIR}"
# )

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

# NOTE(Ibrahim):
# puts klay_ext.abi3.so into the klay/ Python package directory
# (inside the wheel)
# also install (lib)klay in Python pakcage directory 
# to avoid installing libklay system wide
install(TARGETS klay klay_ext LIBRARY DESTINATION klay)

# -----------------------------------------------
# PYTHON KLAY EXTENSION STUBS
# -----------------------------------------------

# https://nanobind.readthedocs.io/en/latest/typing.html#cmake-interface
nanobind_add_stub(
  klay_ext_stub
  MODULE klay_ext
  OUTPUT klay_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:klay_ext>
  DEPENDS klay_ext
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/klay_ext.pyi"
  DESTINATION klay
)
