function(vulkan_base)
    find_package (glfw3 REQUIRED)
    find_package (glm REQUIRED)
    find_package (Vulkan REQUIRED)
    find_package (tinyobjloader REQUIRED)

    find_package (PkgConfig)
    # pkg_get_variable (STB_INCLUDEDIR stb includedir)
    if (NOT STB_INCLUDEDIR)
      unset (STB_INCLUDEDIR)
      find_path (STB_INCLUDEDIR stb_image.h PATH_SUFFIXES stb)
    endif ()
    if (NOT STB_INCLUDEDIR)
      message (FATAL_ERROR "stb_image.h not found")
    endif ()

    #add_executable (glslang::validator IMPORTED)
    #find_program (GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
    #set_property (TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")
endfunction(vulkan_base)
