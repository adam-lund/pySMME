cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(pysmme)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

#without this next command we need the armadillo folder. seems always needs access
#to github to update
#find_package(Armadillo "10.6" CONFIG REQUIRED) 
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

#we dont need to add pybind11 probably because pybind11 is also in the carma dir  
#add_subdirectory(extern/pybind11)
add_subdirectory(extern/carma)
pybind11_add_module(_smme src/bindings.cpp)

target_link_libraries(_smme PRIVATE ${ARMADILLO_LIBARIES} carma::carma)

# define (VERSION_INFO) here.
target_compile_definitions(_smme PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(_smme PRIVATE carma::carma)
