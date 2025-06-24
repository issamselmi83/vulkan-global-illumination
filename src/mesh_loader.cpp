// mesh_loader.cpp
#define TINYOBJLOADER_IMPLEMENTATION         // << exactly once in the whole project
#include "tiny_obj_loader.h"

#include "mesh_loader.hpp"                    // declares mesh::Triangle & loadOBJModel
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>   // <-- add this     length2(vec3) becomes visible


#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>                            // std::accumulate

using mesh::Triangle;                         // shorthand

namespace mesh
{
// ----------------------------------------------------------------------------------
//  loadOBJModel : “good citizen” version
// ----------------------------------------------------------------------------------
//  * triangulates **any** polygon fan-style            (fv ≥ 3, no longer tri/quad only)
//  * keeps per-vertex normals if present, otherwise a flat normal is generated
//  * chooses a per-material diffuse colour (fallback = baseColour arg)
//  * skips zero-area faces to avoid NaNs in cross() / normalize()
//  * reserves exact triangle count, thus zero extra reallocations
// ----------------------------------------------------------------------------------
std::vector<Triangle> loadOBJModel(const std::string& path,
                                   const glm::vec3&   baseColour,
                                   const glm::vec3&   offset)
{
    // ---------------- tinyobj load ----------------
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t>     shapes;
    std::vector<tinyobj::material_t>  materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials,
                          &warn, &err, path.c_str(), /*mtl path*/ nullptr,
                          /*triangulate*/ false, /*default_vcol*/ false))
    {
        throw std::runtime_error("OBJ load failed: " + path + " – " + err);
    }
    if (!warn.empty()) std::cerr << "[tinyobj] " << warn << '\n';

    // -------- estimate triangle count & reserve ----
    size_t triTotal = 0;
    for (const auto& sh : shapes)
        for (int fv : sh.mesh.num_face_vertices)
            triTotal += std::max(0, fv - 2);         // (N-2) fan triangles

    std::vector<Triangle> tris;
    tris.reserve(triTotal);

    // ---------------- helper lambdas ---------------
    auto fetchPos = [&](int vIdx) -> glm::vec3 {
        return {
            attrib.vertices[3 * vIdx + 0] + offset.x,
            attrib.vertices[3 * vIdx + 1] + offset.y,
            attrib.vertices[3 * vIdx + 2] + offset.z
        };
    };
    auto fetchNrm = [&](int nIdx) -> glm::vec3 {
        return (nIdx >= 0) ? glm::vec3{
            attrib.normals[3 * nIdx + 0],
            attrib.normals[3 * nIdx + 1],
            attrib.normals[3 * nIdx + 2]
        } : glm::vec3{0};
    };

    // ------------------- main loop -----------------
    for (const auto& shape : shapes)
    {
        size_t idxOff = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
        {
            const int fv     = shape.mesh.num_face_vertices[f];
            const int matIdx = (f < shape.mesh.material_ids.size())
                               ? shape.mesh.material_ids[f] : -1;

            glm::vec3 faceColour = baseColour;
            if (matIdx >= 0 && matIdx < static_cast<int>(materials.size()))
                faceColour = {
                    materials[matIdx].diffuse[0],
                    materials[matIdx].diffuse[1],
                    materials[matIdx].diffuse[2]
                };

            // produce fan triangles (v0, v[i], v[i+1]) for i = 1 .. fv-2
            for (int i = 1; i + 1 < fv; ++i)
            {
                const tinyobj::index_t idx0 = shape.mesh.indices[idxOff + 0];
                const tinyobj::index_t idx1 = shape.mesh.indices[idxOff + i];
                const tinyobj::index_t idx2 = shape.mesh.indices[idxOff + i + 1];

                Triangle t{};
                // positions
                t.v0 = glm::vec4(fetchPos(idx0.vertex_index), 1.0f);
                t.v1 = glm::vec4(fetchPos(idx1.vertex_index), 1.0f);
                t.v2 = glm::vec4(fetchPos(idx2.vertex_index), 1.0f);

                // normals: keep vertex normals if OBJ has them, else flat
                glm::vec3 n0 = fetchNrm(idx0.normal_index);
                glm::vec3 n1 = fetchNrm(idx1.normal_index);
                glm::vec3 n2 = fetchNrm(idx2.normal_index);

                glm::vec3 faceN = glm::normalize(
                    glm::cross(glm::vec3(t.v1) - glm::vec3(t.v0),
                               glm::vec3(t.v2) - glm::vec3(t.v0)));

                // guard: skip degenerate faces
                if (!std::isfinite(faceN.x)) continue;
                if (glm::length2(faceN) < 1e-12f) continue;

                // if any vertex normal is missing, use flat
                if (n0 == glm::vec3(0) || n1 == glm::vec3(0) || n2 == glm::vec3(0))
                    t.normal = glm::vec4(faceN, 0.0f);
                else
                    t.normal = glm::vec4(glm::normalize((n0 + n1 + n2) / 3.0f), 0.0f);

                t.albedo   = glm::vec4(faceColour, 0.0f);
                t.specular = glm::vec4(0);                 // fill as needed

                tris.emplace_back(t);
            }
            idxOff += fv;
        }
    }

    std::cout << "[MeshLoader] " << tris.size()
              << " triangles from " << path << '\n';
    return tris;
}
} // namespace mesh
