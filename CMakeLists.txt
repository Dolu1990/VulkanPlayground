cmake_minimum_required(VERSION 3.22)


project(
    VulkanPlayground
    VERSION 0.0.0
    LANGUAGES
        ASM
        C
        CXX
)

include(cmake/tools/compile-shader.cmake)
include(cmake/tools/base.cmake)

file(GLOB_RECURSE SHADERS "${CMAKE_SOURCE_DIR}/shaders/*.vert" "${CMAKE_SOURCE_DIR}/shaders/*.frag")
compile_shaders(TARGETS ${SHADERS})

add_subdirectory(src)
