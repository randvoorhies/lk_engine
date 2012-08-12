cmake_minimum_required(VERSION 2.8)
project(lk_engine)

set(CMAKE_CXX_COMPILER g++-4.6)

add_executable(lk_engine test_lk_engine.cpp)

find_package(PkgConfig)

pkg_check_modules(OPENCV opencv)
include_directories(${OPENCV_INCLUDE_DIRS})
target_link_libraries(lk_engine ${OPENCV_LIBRARIES})

include_directories(.)

add_definitions(-std=c++0x)


