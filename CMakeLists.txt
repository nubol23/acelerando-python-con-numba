cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories("include")

add_executable(main main.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 17)