add_subdirectory(commons)
add_subdirectory(hello_world)
add_subdirectory(hello_vulkan)

set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
add_subdirectory(tinygltf)
