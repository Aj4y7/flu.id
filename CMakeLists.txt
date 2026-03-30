cmake_minimum_required(VERSION 3.16)
project(fluid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -ffast-math -march=native")

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)
find_package(OpenMP REQUIRED)

add_executable(fluid main.cpp fluid.cpp)
target_link_libraries(fluid PRIVATE SFML::Graphics SFML::Window SFML::System OpenMP::OpenMP_CXX)