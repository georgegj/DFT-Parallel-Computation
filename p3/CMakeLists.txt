cmake_minimum_required(VERSION 3.9)
project(p3)
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package (Threads)
set(CMAKE_CXX_FLAGES "${CMAKE_CXX_FLAGES}" -std=c++11)
set(CUDA_NVCC_FLAGES ${CUDA_NVCC_FLAGES};
 -03 -gencode arch=compute_22, code=sm_22
 )
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread")

include_directories(${MPI_INCLUDE_PATH})
include_directories(
 src
)
set(SOURCE_FILES
 src/cuda.cu
 src/inputimage.cu
 src/inputimage.cuh
 src/complex.cuh
 src/complex.cu
 src/threads.cc
 src/mpi.cc
 src/inputimage.cc
 src/inputimage.h
 src/complex.cc
 src/complex.h

 )
cuda_add_executable(
 p33
 src/cuda.cu
)

add_executable(p32 src/mpi.cc)
target_link_libraries(p32 ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(p32 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(p32 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

add_executable (p31 src/threads.cc)
target_link_libraries (p31 ${CMAKE_THREAD_LIBS_INIT})







