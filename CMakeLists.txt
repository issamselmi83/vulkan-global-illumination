cmake_minimum_required(VERSION 3.5.0)
project(vulkan_ray_tracer CXX)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# === SHADER COMPILATION ===
set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/shaders)
set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB SHADERS 
    ${SHADER_SRC_DIR}/*.vert 
    ${SHADER_SRC_DIR}/*.frag 
    ${SHADER_SRC_DIR}/*.comp
)

# Trouve glslc (nécessite que Vulkan SDK soit bien installé)
find_program(GLSLC_EXECUTABLE glslc HINTS ENV PATH)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Please install Vulkan SDK or shaderc.")
endif()

foreach(SHADER IN LISTS SHADERS)
    get_filename_component(FILENAME ${SHADER} NAME)
    set(SPV_OUTPUT ${SHADER_OUT_DIR}/${FILENAME}.spv)

    add_custom_command(
        OUTPUT ${SPV_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
        COMMAND ${GLSLC_EXECUTABLE} ${SHADER} -o ${SPV_OUTPUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${FILENAME}"
    )

    list(APPEND SPV_SHADERS ${SPV_OUTPUT})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_SHADERS})

# === INCLUDES ===
include_directories(src/)
include_directories(third_party/stb/)
include_directories(third_party/tinyobjloader/)

# === EXECUTABLE ===
add_executable(vulkan_ray_tracer 
    src/main.cpp
    src/vrt_camera.cpp
    src/vrt_ray_tracer.cpp
    src/vrt_window.cpp
    src/mesh_loader.cpp
)

target_link_libraries(vulkan_ray_tracer Vulkan::Vulkan glfw)
