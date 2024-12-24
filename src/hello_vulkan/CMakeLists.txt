vulkan_base()

set(TARGET hello_vulkan)
add_executable (${TARGET} main.cpp)
target_link_libraries (${TARGET} Vulkan::Vulkan glfw commons)
target_include_directories (${TARGET} PRIVATE ${STB_INCLUDEDIR})
add_dependencies(${TARGET} default_frag default_vert)

target_include_directories(${TARGET} PRIVATE "../tinygltf")

#set(TEXTURES ${CMAKE_SOURCE_DIR}/images/texture.jpg)
#set(SHADERS ${CMAKE_SOURCE_DIR}/shaders/default.frag.spv ${CMAKE_SOURCE_DIR}/shaders/default.vert.spv)

#file (COPY ${TEXTURES} DESTINATION ${CMAKE_BINARY_DIR}/src/hello_vulkan/textures)
#file (COPY ${SHADERS} DESTINATION ${CMAKE_BINARY_DIR}/src/hello_vulkan/shaders)
