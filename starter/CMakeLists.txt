# Metal compute skeleton, test task 03, "Mac" track.
#   cmake -S starter -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#   ./build/blur_metal --list
# kernels.metal is compiled at runtime: Command Line Tools are enough;
# full Xcode is optional (Instruments, GPU Capture).
cmake_minimum_required(VERSION 3.20)
project(blur_metal LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "This skeleton is macOS only. For the no-Mac track use starter_cuda/ (see README).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CPU reference (shared by all tracks).
add_subdirectory(../cpu_reference "${CMAKE_CURRENT_BINARY_DIR}/cpu_reference")

add_executable(blur_metal src/main.mm src/kernels.metal)
set_source_files_properties(src/kernels.metal PROPERTIES HEADER_FILE_ONLY ON)
target_compile_options(blur_metal PRIVATE -fobjc-arc -Wall -Wextra)
get_filename_component(BHS_ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets" ABSOLUTE)
target_compile_definitions(blur_metal PRIVATE
  BHS_KERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.metal"
  BHS_ASSET_DIR="${BHS_ASSET_DIR}")
target_link_libraries(blur_metal PRIVATE blur_reference
  "-framework Metal" "-framework Foundation" "-framework ImageIO" "-framework CoreGraphics")

# Plus (iOS / iPadOS, not required): needs an Xcode project with an app target.
