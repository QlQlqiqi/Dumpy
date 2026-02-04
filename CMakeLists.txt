cmake_minimum_required(VERSION 3.20)
project(Dumpy)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -g -O0 -Wall")
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
find_package(Threads REQUIRED)
add_compile_options(-march=haswell)

SET(ROOT_DIR ${CMAKE_SOURCE_DIR})
include_directories(${ROOT_DIR}/src)
include_directories(${ROOT_DIR}/include)

add_subdirectory(src)

add_executable(Dumpy src/main.cpp)

target_link_libraries(Dumpy Threads::Threads libboost_serialization.so)
target_link_libraries(Dumpy dumpy_lib search_engine_lib src_utils_lib index_construction_lib)
