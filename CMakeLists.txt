cmake_minimum_required(VERSION 3.18)
project(py_fricp)

add_subdirectory(dependencies/pybind11)

find_package(Eigen3 REQUIRED)
#--- OpenMP (cannot work in clang)
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

pybind11_add_module(py_fricp models/py_fricp.cpp)
target_include_directories(py_fricp PRIVATE dependencies/Fast-Robust-ICP
                                            dependencies/Fast-Robust-ICP/include
                                            ${EIGEN3_INCLUDE_DIRS}
                                            OpenMP::OpenMP_CXX)
target_compile_definitions(py_fricp
                           PRIVATE VERSION_INFO=${PY_FRICP_VERSION_INFO})