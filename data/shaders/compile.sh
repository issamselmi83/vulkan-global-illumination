#!/bin/sh
# Shaders for the normal (color and depth) pipeline.
glslc shader.vert -o rendering.vert.spv
glslc shader.frag -o rendering.frag.spv
glslc ray_tracing.comp -o ray_tracing.comp.spv
echo "Graphics Shaders compiled..."
