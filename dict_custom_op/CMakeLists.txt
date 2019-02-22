cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dictionary_op)

find_package(Torch REQUIRED)

# Define our library target
add_library(dictionary_op SHARED op.cpp)

# Enable C++11
target_compile_features(dictionary_op PRIVATE cxx_range_for)

# Set gcc options
target_compile_options(dictionary_op PRIVATE -g -Wall -Wextra -pedantic -fdiagnostics-color)

# Link against LibTorch
target_link_libraries(dictionary_op "${TORCH_LIBRARIES}")
