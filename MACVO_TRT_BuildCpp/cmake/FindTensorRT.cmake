# FindTensorRT.cmake
#
# Locates an installed NVIDIA TensorRT SDK and exposes modern imported
# targets. Synthesised from three upstream references:
#   * OlivierLDff's community gist (component model, version parsing)
#       https://gist.github.com/OlivierLDff/aafaee8373a2337fef8cd5d2ece973c7
#   * NVIDIA TensorRT-OSS find_library_create_target (Windows IMPLIB pattern)
#       https://github.com/NVIDIA/TensorRT/blob/main/cmake/modules/find_library_create_target.cmake
#   * PRBonn rangenet_lib's tensorrt-config.cmake (multi-component support)
#
# Inputs (any one of these is enough to locate a non-default install):
#   TensorRT_ROOT          (cache var or env, CMake-standard <Package>_ROOT)
#   TENSORRT_ROOT / TRT_ROOT / TENSORRT_DIR (env var aliases)
#   CUDAToolkit_ROOT / CUDA_PATH (Windows CUDA installer co-installs TRT)
#
# Output variables (CMake-convention 4-component version):
#   TensorRT_FOUND
#   TensorRT_VERSION                              (e.g. "10.13.2.6")
#   TensorRT_VERSION_{MAJOR,MINOR,PATCH,TWEAK}
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARY_DIRS
#   TensorRT_LIBRARIES
#
# Imported targets (transitively link to TensorRT::nvinfer):
#   TensorRT::nvinfer
#   TensorRT::nvonnxparser
#   TensorRT::nvinfer_plugin
#
# Usage:
#   list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
#   find_package(TensorRT 10 REQUIRED COMPONENTS nvinfer nvonnxparser)
#   target_link_libraries(my_target PRIVATE TensorRT::nvonnxparser)

include(FindPackageHandleStandardArgs)

if(NOT TensorRT_FIND_COMPONENTS)
    set(TensorRT_FIND_COMPONENTS nvinfer nvonnxparser)
endif()

# ---------------------------------------------------------------------------
# Build the search hint list. The CMake-standard <Package>_ROOT variable is
# already consulted by find_path/find_library automatically (CMake >= 3.12),
# so we only have to prepend the legacy NVIDIA aliases plus the CUDA root.
# ---------------------------------------------------------------------------
set(_TRT_SEARCH_HINTS "")
foreach(_var
    ENV{TENSORRT_ROOT}
    ENV{TENSORRT_DIR}
    ENV{TRT_ROOT}
    CUDAToolkit_ROOT
    CUDA_TOOLKIT_ROOT_DIR
    ENV{CUDA_PATH}
)
    if(DEFINED ${_var} OR DEFINED ENV{${_var}})
        list(APPEND _TRT_SEARCH_HINTS "${${_var}}")
    endif()
endforeach()
if(WIN32)
    list(APPEND _TRT_SEARCH_HINTS
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
        "C:/Program Files/NVIDIA/TensorRT"
    )
else()
    list(APPEND _TRT_SEARCH_HINTS
        "/usr/local/tensorrt" "/opt/tensorrt" "/usr"
    )
endif()
list(REMOVE_DUPLICATES _TRT_SEARCH_HINTS)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TRT_SEARCH_HINTS}
    PATH_SUFFIXES include include/x86_64-linux-gnu
    DOC "TensorRT include directory (containing NvInfer.h)"
)

# ---------------------------------------------------------------------------
# Version parsing. NvInferVersion.h ships in two flavours:
#   * Public:     #define NV_TENSORRT_MAJOR 10
#   * Enterprise: #define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE
#                 #define TRT_MAJOR_ENTERPRISE 10
# ---------------------------------------------------------------------------
function(_trt_extract_version HEADER_TEXT OUT_VAR DIRECT INDIRECT)
    set(_value "")
    string(REGEX MATCH "#define[ \t]+${DIRECT}[ \t]+([0-9]+)" _ "${HEADER_TEXT}")
    if(CMAKE_MATCH_1)
        set(_value "${CMAKE_MATCH_1}")
    else()
        string(REGEX MATCH "#define[ \t]+${INDIRECT}[ \t]+([0-9]+)" _ "${HEADER_TEXT}")
        if(CMAKE_MATCH_1)
            set(_value "${CMAKE_MATCH_1}")
        endif()
    endif()
    set(${OUT_VAR} "${_value}" PARENT_SCOPE)
endfunction()

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_h)
    _trt_extract_version("${_trt_h}" TensorRT_VERSION_MAJOR
        NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE)
    _trt_extract_version("${_trt_h}" TensorRT_VERSION_MINOR
        NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE)
    _trt_extract_version("${_trt_h}" TensorRT_VERSION_PATCH
        NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE)
    _trt_extract_version("${_trt_h}" TensorRT_VERSION_TWEAK
        NV_TENSORRT_BUILD TRT_BUILD_ENTERPRISE)
    if(TensorRT_VERSION_MAJOR AND DEFINED TensorRT_VERSION_MINOR
       AND DEFINED TensorRT_VERSION_PATCH AND DEFINED TensorRT_VERSION_TWEAK)
        set(TensorRT_VERSION
            "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_TWEAK}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Per-component finder. Mirrors NVIDIA TensorRT-OSS' find_library_create_target
# pattern: SHARED IMPORTED with IMPORTED_IMPLIB on Windows so downstream
# $<TARGET_FILE:TensorRT::nvinfer> returns the DLL (for runtime-copy steps).
# Falls back to UNKNOWN IMPORTED if the DLL cannot be located alongside the
# .lib (NVIDIA's standard install puts both in <root>/bin on Windows).
# ---------------------------------------------------------------------------
set(_TRT_LIB_SUFFIXES
    lib lib/x64 lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu bin)

function(_trt_find_component COMP)
    set(_names "")
    if(TensorRT_VERSION_MAJOR)
        list(APPEND _names "${COMP}_${TensorRT_VERSION_MAJOR}")
    endif()
    list(APPEND _names "${COMP}")

    find_library(TensorRT_${COMP}_LIBRARY
        NAMES ${_names}
        HINTS ${_TRT_SEARCH_HINTS}
        PATH_SUFFIXES ${_TRT_LIB_SUFFIXES}
        DOC "TensorRT ${COMP} library"
    )

    if(NOT TensorRT_${COMP}_LIBRARY)
        set(TensorRT_${COMP}_FOUND FALSE PARENT_SCOPE)
        return()
    endif()
    set(TensorRT_${COMP}_FOUND TRUE PARENT_SCOPE)

    get_filename_component(_lib_dir "${TensorRT_${COMP}_LIBRARY}" DIRECTORY)
    if(NOT TensorRT_LIBRARY_DIRS)
        set(TensorRT_LIBRARY_DIRS "${_lib_dir}" CACHE INTERNAL "")
    endif()

    if(TARGET TensorRT::${COMP})
        return()
    endif()

    if(WIN32)
        # On Windows find_library returns the .lib import library; the .dll
        # lives next to it (NVIDIA's bin/ co-install) or in a sibling lib/.
        # Strip the trailing `_<MAJOR>` is already in the name; just swap ext.
        get_filename_component(_lib_stem "${TensorRT_${COMP}_LIBRARY}" NAME_WE)
        find_file(TensorRT_${COMP}_DLL
            NAMES "${_lib_stem}.dll"
            HINTS "${_lib_dir}" ${_TRT_SEARCH_HINTS}
            PATH_SUFFIXES bin lib lib/x64
            NO_DEFAULT_PATH
        )
        if(TensorRT_${COMP}_DLL)
            add_library(TensorRT::${COMP} SHARED IMPORTED)
            set_target_properties(TensorRT::${COMP} PROPERTIES
                IMPORTED_IMPLIB   "${TensorRT_${COMP}_LIBRARY}"
                IMPORTED_LOCATION "${TensorRT_${COMP}_DLL}"
            )
        else()
            add_library(TensorRT::${COMP} UNKNOWN IMPORTED)
            set_target_properties(TensorRT::${COMP} PROPERTIES
                IMPORTED_LOCATION "${TensorRT_${COMP}_LIBRARY}"
            )
        endif()
    else()
        add_library(TensorRT::${COMP} UNKNOWN IMPORTED)
        set_target_properties(TensorRT::${COMP} PROPERTIES
            IMPORTED_LOCATION "${TensorRT_${COMP}_LIBRARY}"
        )
    endif()

    set_target_properties(TensorRT::${COMP} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
endfunction()

# Always probe nvinfer first; everything else links to it transitively.
_trt_find_component(nvinfer)
foreach(_c IN LISTS TensorRT_FIND_COMPONENTS)
    if(NOT _c STREQUAL "nvinfer")
        _trt_find_component(${_c})
    endif()
endforeach()

# Wire transitive dependency: every parser/plugin links nvinfer.
foreach(_dep nvonnxparser nvinfer_plugin)
    if(TARGET TensorRT::${_dep} AND TARGET TensorRT::nvinfer)
        set_property(TARGET TensorRT::${_dep} APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
    endif()
endforeach()

set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
set(TensorRT_LIBRARIES "")
foreach(_c IN LISTS TensorRT_FIND_COMPONENTS)
    if(TensorRT_${_c}_LIBRARY)
        list(APPEND TensorRT_LIBRARIES "${TensorRT_${_c}_LIBRARY}")
    endif()
endforeach()

find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY
    VERSION_VAR   TensorRT_VERSION
    HANDLE_COMPONENTS
)

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvonnxparser_LIBRARY
    TensorRT_nvinfer_plugin_LIBRARY
    TensorRT_nvinfer_DLL
    TensorRT_nvonnxparser_DLL
    TensorRT_nvinfer_plugin_DLL
)
