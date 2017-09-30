cmake_minimum_required(VERSION 3.0)
project(boo)

set(ATen_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ATen)
set(ATen_BINARY_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/ATen/build)
set(MNIST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mnist)

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(CUDA 5.5)
find_package( OpenCV REQUIRED )

#find_package(HDF5 COMPONENTS C HL NO_MODULE REQUIRED static)

include_directories(
  # dense
  ${ATen_SOURCE_DIR}/lib/TH
  ${ATen_SOURCE_DIR}/lib/THC
  ${ATen_BINARY_DIR}/lib/TH
  ${ATen_BINARY_DIR}/lib/THC
  # sparse
  ${ATen_SOURCE_DIR}/lib/THS
  ${ATen_SOURCE_DIR}/lib/THCS
  ${ATen_BINARY_DIR}/lib/THS
  ${ATen_BINARY_DIR}/lib/THCS

  ${ATen_SOURCE_DIR}/lib
  ${ATen_BINARY_DIR}/lib)

include_directories(
${ATen_SOURCE_DIR}/lib/THNN
${ATen_SOURCE_DIR}/lib/THCUNN)

include_directories(
${ATen_SOURCE_DIR}/src
${ATen_BINARY_DIR}/src/ATen)

include_directories(
${MNIST_SOURCE_DIR}/include/mnist)

# Uncomment upon building on GPU machine
# INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/CMake-hdf5-1.8.19/hdf5-1.8.19/src
${CMAKE_CURRENT_SOURCE_DIR}/CMake-hdf5-1.8.19/hdf5-1.8.19/c++/src
${CMAKE_CURRENT_SOURCE_DIR}/CMake-hdf5-1.8.19/build
)

set(_hdf5_libs
${CMAKE_CURRENT_SOURCE_DIR}/CMake-hdf5-1.8.19/build/bin/libhdf5_cpp.a
${CMAKE_CURRENT_SOURCE_DIR}/CMake-hdf5-1.8.19/build/bin/libhdf5.a
${CMAKE_CURRENT_SOURCE_DIR}/CMake-hdf5-1.8.19/build/bin/libz.a
${CMAKE_CURRENT_SOURCE_DIR}/CMake-hdf5-1.8.19/build/bin/libszip.a
-ldl
)

# TODO: structure project in a better way
# Temporary solution -- change to a normal cpp project structure later
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

ADD_EXECUTABLE(perceptron examples/perceptron.cpp)
TARGET_LINK_LIBRARIES(perceptron ${ATen_BINARY_DIR}/src/ATen/libATen.dylib ${_hdf5_libs} ${OpenCV_LIBS})

ADD_EXECUTABLE(vanilla examples/vanilla.cpp)
TARGET_LINK_LIBRARIES(vanilla ${ATen_BINARY_DIR}/src/ATen/libATen.dylib ${_hdf5_libs} ${OpenCV_LIBS})

ADD_EXECUTABLE(deep examples/deep.cpp)
TARGET_LINK_LIBRARIES(deep ${ATen_BINARY_DIR}/src/ATen/libATen.dylib ${_hdf5_libs} ${OpenCV_LIBS})

ADD_EXECUTABLE(numpy examples/numpy.cpp)
TARGET_LINK_LIBRARIES(numpy ${ATen_BINARY_DIR}/src/ATen/libATen.dylib ${_hdf5_libs} ${OpenCV_LIBS})