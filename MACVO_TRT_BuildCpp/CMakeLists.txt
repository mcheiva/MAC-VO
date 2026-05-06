cmake_minimum_required(VERSION 3.21)
project(macvo_trt_builder
    VERSION 0.1.0
    DESCRIPTION "MACVO FlowFormerCov ONNX -> TensorRT engine builder"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
endif()

find_package(CLI11 REQUIRED)
find_package(TensorRT 10 REQUIRED COMPONENTS nvinfer nvonnxparser)

message(STATUS "TensorRT ${TensorRT_VERSION} -> ${TensorRT_INCLUDE_DIR}")

# Header-only library: downstream C++ projects do `target_link_libraries(...
# macvo_trt::builder)` and call macvo_trt::Build(opts) directly.
add_library(macvo_trt_builder INTERFACE)
add_library(macvo_trt::builder ALIAS macvo_trt_builder)
target_include_directories(macvo_trt_builder INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(macvo_trt_builder INTERFACE
    TensorRT::nvinfer
    TensorRT::nvonnxparser
)
target_compile_features(macvo_trt_builder INTERFACE cxx_std_20)

add_executable(macvo_trt_build src/main.cpp)
target_link_libraries(macvo_trt_build PRIVATE
    macvo_trt::builder
    CLI11::CLI11
)

include(GNUInstallDirs)
install(TARGETS macvo_trt_build
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
