
cmake_minimum_required(VERSION 3.2)
project(GRNS)

set( CMAKE_VERBOSE_MAKEFILE on )

#####################################
# Paths section
#####################################
set(GRNS_HOME_DIR "${PROJECT_SOURCE_DIR}") # RNS-Lib home directory, should be fixed since PROJECT_SOURCE_DIR is changed
list(APPEND CMAKE_MODULE_PATH "${GRNS_HOME_DIR}/cmake/") # CMake modules
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc") # Path to nvcc

#####################################
# CUDA
#####################################
set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")
set(CMAKE_CUDA_STANDARD 14) # Default value for CUDA_STANDARD property of targets

#####################################
# Compile options
#####################################
#set(GLOBAL_COMPILE_OPTIONS ${GLOBAL_COMPILE_OPTIONS} -Xcompiler=-O3,-fopenmp,-ffast-math -use_fast_math)
set(GLOBAL_COMPILE_OPTIONS ${GLOBAL_COMPILE_OPTIONS} -Xcompiler=-fopenmp)

####################################
# Directories
#####################################
include_directories(src/) # Source code
include_directories(tests/) # Benchmarks

#####################################
# Recurse into benchmarks directory
#####################################
add_subdirectory(tests/)