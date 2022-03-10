cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(pysmme)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

find_package(Armadillo "10.6" CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

#add_subdirectory(extern/pybind11)
add_subdirectory(extern/carma)
pybind11_add_module(_smme src/bindings.cpp)

target_link_libraries(_smme PRIVATE ${ARMADILLO_LIBARIES} carma::carma)

# define (VERSION_INFO) here.
target_compile_definitions(_smme PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(_smme PRIVATE carma::carma)
