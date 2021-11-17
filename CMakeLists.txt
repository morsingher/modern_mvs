cmake_minimum_required (VERSION 3.17)
project (mvs LANGUAGES CUDA CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ffast-math -march=native -mtune=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --use_fast_math -arch=sm_37 -Wno-deprecated-gpu-targets -w")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED)

include_directories(
  src/utils
  src/math
  src/cuda
  src/core
  src/fusion
  ${OpenCV_INCLUDE_DIRS} 
  ${Eigen3_INCLUDE_DIR} 
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} -Wno-unused-result -Wno-dev)

add_executable(mvs 
  src/utils/io_utils.cc 
  src/utils/options.cc 
  src/math/camera.cu 
  src/cuda/texture_array.cu
  src/core/cuda_kernels.cu 
  src/core/patchmatch.cu 
  src/fusion/point_cloud.cu
  src/main.cu
)

set_target_properties(mvs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(cuda-api-wrappers STATIC IMPORTED)
set_target_properties(cuda-api-wrappers PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcuda-nvtx-wrappers.a)
set_target_properties(cuda-api-wrappers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/lib/)

target_link_libraries(mvs ${OpenCV_LIBS} Eigen3::Eigen ${PCL_LIBRARIES} cuda-api-wrappers)