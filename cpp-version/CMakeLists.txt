
cmake_minimum_required(VERSION 3.10)
project(PoseTracker)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
add_executable(PoseTracker src/main.cpp)

# Link OpenCV
target_link_libraries(PoseTracker ${OpenCV_LIBS})
