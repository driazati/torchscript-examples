cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(ivalues_demo ivalues.cpp)
target_compile_options(ivalues_demo PRIVATE -g -Wall -Wextra -pedantic -fdiagnostics-color)
target_link_libraries(ivalues_demo "${TORCH_LIBRARIES}")
set_property(TARGET ivalues_demo PROPERTY CXX_STANDARD 11)
