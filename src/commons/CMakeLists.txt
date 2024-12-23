set(TARGET commons)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
add_library(${TARGET} WindowVulkan.cpp)
target_include_directories(${TARGET}  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET}  PUBLIC Vulkan::Vulkan glfw)