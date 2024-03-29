# cmake_policy(SET CMP0048 NEW)  # project_VERSION* variables populated from project(... VERSION x.x.x) string
project(ocdft CXX)
set(ocdft_AUTHORS      "Francesco A. Evangelista and Group")
set(ocdft_DESCRIPTION  "An orthogonality constrained DFT plugin to Psi4")
set(ocdft_URL          "https://github.com/evangelistalab/ocdft")
set(ocdft_LICENSE      "GPL-3.0+")

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#include(psi4OptionsTools)

find_package(psi4 1.1 REQUIRED)

#find_package(TargetLAPACK REQUIRED)

## Including custom cmake rules
#include(cmake/clang-cxx-dev-tools.cmake)
#include(cmake/git-version.cmake)

# List of CC files
add_psi4_plugin(ocdft
src/plugin.cc
src/aosubspace.cc
src/constraint.cc
src/determinant.cc
src/helpers.cc
src/iao_builder.cc
src/ocdft.cc
src/ocdft_analyze_excitations.cc
src/livvo.cc
src/ucks_determinant.cc
src/fmt/format.cc
#
#src/rcks.cc
#src/fasnocis.cc
#src/fasnocis.h
#src/junk.h
#src/noci.h
#src/noci_mat.cc
#src/noci_mat.h
)

target_include_directories(ocdft PRIVATE src/fmt)

