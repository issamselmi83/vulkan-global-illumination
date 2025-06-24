#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>

#include "vrt_ray_tracer.hpp"

struct Triangle {
    glm::vec4 v0, v1, v2;   // position, w = 1
    glm::vec4 normal;       // w = 0
    glm::vec4 albedo;       // colour, w = 0
    glm::vec4 specular;     // w = 0
};

namespace mesh {
    /** Load an OBJ, triangulate quads, paint it `baseColour`, translate by `offset`. */
    using Triangle = vrt::Triangle;
    std::vector<Triangle> loadOBJModel(const std::string& path,
                                  const glm::vec3& baseColour = {1,1,1},
                                  const glm::vec3& offset     = {0,0,0});
}
