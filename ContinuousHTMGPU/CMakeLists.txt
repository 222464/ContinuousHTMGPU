cmake_minimum_required(VERSION 2.8)

project(CHTMGPU)

include_directories("${PROJECT_SOURCE_DIR}/source")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
find_package(OpenCL REQUIRED)
 
include_directories(${OPENCL_INCLUDE_DIRS})
 
if(OPENCL_HAS_CPP_BINDINGS)
    message("OpenCL has CPP bindings. Full include is: " ${OPENCL_INCLUDE_DIRS})
else(OPENCL_HAS_CPP_BINDINGS)
    message("No OpenCL CPP bindings found")
endif(OPENCL_HAS_CPP_BINDINGS)

find_package(SFML 2 REQUIRED system window graphics)
 
include_directories(${SFML_INCLUDE_DIR})
 
add_executable(CHTMGPU "${PROJECT_SOURCE_DIR}/source/Main.cpp")

target_link_libraries(CHTMGPU ${OPENCL_LIBRARIES})
target_link_libraries(CHTMGPU ${SFML_LIBRARIES})