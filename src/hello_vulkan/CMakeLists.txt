vulkan_base()

add_executable ( hello_vulkan main.cpp )
target_link_libraries ( hello_vulkan Vulkan::Vulkan glfw )
target_include_directories (hello_vulkan PRIVATE ${STB_INCLUDEDIR})
add_dependencies(hello_vulkan default_frag default_vert)

#set(TEXTURES ${CMAKE_SOURCE_DIR}/images/texture.jpg)
#set(SHADERS ${CMAKE_SOURCE_DIR}/shaders/default.frag.spv ${CMAKE_SOURCE_DIR}/shaders/default.vert.spv)

#file (COPY ${TEXTURES} DESTINATION ${CMAKE_BINARY_DIR}/src/hello_vulkan/textures)
#file (COPY ${SHADERS} DESTINATION ${CMAKE_BINARY_DIR}/src/hello_vulkan/shaders)
