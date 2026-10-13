# CPU reference implementation of the Gaussian blur. Builds on any OS.
#   cmake -S cpu_reference -B build-ref && cmake --build build-ref
#   ./build-ref/blur_ref test_4k.ppm ref.ppm 4.0
cmake_minimum_required(VERSION 3.20)
project(bhs_blur_reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(blur_reference STATIC gaussian_blur_ref.cpp gaussian_blur_ref.h ppm_io.h)
target_include_directories(blur_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(blur_ref blur_ref_main.cpp)
target_link_libraries(blur_ref PRIVATE blur_reference)

if(MSVC)
  target_compile_options(blur_reference PRIVATE /W4)
else()
  target_compile_options(blur_reference PRIVATE -Wall -Wextra)
endif()
