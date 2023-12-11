cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

project(cw02 LANGUAGES CXX CUDA)

file(GLOB_RECURSE CUDA_SOURCES src/cuda/*.cu src/cuda/*.h)
add_executable(nqueens-cuda ${CUDA_SOURCES})
set_property(TARGET nqueens-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
file(GLOB_RECURSE SERIAL_SOURCES src/serial/*.cpp src/serial/*.h)
add_executable(nqueens-serial ${SERIAL_SOURCES})

#### OPEN MP PACKAGES ####
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

file(GLOB_RECURSE OPENMP_SOURCES src/openmp/*.cpp src/openmp/*.h)
add_executable(nqueens-openmp ${OPENMP_SOURCES})