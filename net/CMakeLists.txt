cmake_minimum_required(VERSION 3.9)
project(cursovaya)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
find_package( OpenCV REQUIRED )

set(SOURCE_FILES generator.cpp)
add_subdirectory(FastNoise)
add_executable(cursovaya ${SOURCE_FILES})



include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cursovaya ${OpenCV_LIBS})