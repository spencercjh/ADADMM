cmake_minimum_required(VERSION 3.9)
project(ADADMM)

set(CMAKE_CXX_STANDARD 11)
#SET(CMAKE_CXX_COMPILER "mpic++")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_C_COMPILER  mpicc)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/optimizer SRC_FILE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/data SRC_FILE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/math SRC_FILE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/other SRC_FILE)

add_executable(adadmm ${SRC_FILE})
