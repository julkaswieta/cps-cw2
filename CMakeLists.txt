cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

project(cw02 LANGUAGES CXX CUDA)

add_executable(nqueens-cuda src/nqueens.cu)
set_property(TARGET nqueens-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
add_executable(nqueens-serial src/nqueens.cpp)

#Main output directory
SET(OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
# Ouput all DLLs from all libs into main build folder
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})

#### OPEN MP PACKAGES ####
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

file(GLOB_RECURSE SOURCES src/openmp/*.cpp src/openmp/*.h)
add_executable(nqueens-openmp ${SOURCES})