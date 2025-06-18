// material_presets.hpp
#pragma once
#include <glm/vec3.hpp>
#include <random>

namespace vrt {
  struct Material { glm::vec3 albedo, specular; };

  enum class MaterialPreset {
    Gold,
    Silver,
    Copper,
    Bronze,
    Brass,
    Aluminum,
    Iron,
    Chrome,
    GlassClear,
    GlassBlue,
    Emerald,
    Ruby,
    Obsidian,
    Rubber,
    WoodOak,
    WoodPine,
    PlasticRed,
    PlasticGreen,
    PlasticBlue,
    PlasticYellow
  };

  inline Material get(MaterialPreset m) {
    switch (m) {
      // ⬐ Metals – specular = base colour (Metallic = 1)
      case MaterialPreset::Gold:    return {{1.000f,0.766f,0.336f}, {1.000f,0.766f,0.336f}};
      case MaterialPreset::Silver:  return {{0.972f,0.960f,0.915f}, {0.972f,0.960f,0.915f}};
      case MaterialPreset::Copper:  return {{0.955f,0.637f,0.538f}, {0.955f,0.637f,0.538f}};
      case MaterialPreset::Bronze:  return {{0.714f,0.428f,0.181f}, {0.714f,0.428f,0.181f}};
      case MaterialPreset::Brass:   return {{0.780f,0.568f,0.113f}, {0.780f,0.568f,0.113f}};
      case MaterialPreset::Aluminum:return {{0.913f,0.921f,0.925f}, {0.913f,0.921f,0.925f}};
      case MaterialPreset::Iron:    return {{0.560f,0.570f,0.580f}, {0.560f,0.570f,0.580f}};
      case MaterialPreset::Chrome:  return {{0.550f,0.550f,0.550f}, {0.550f,0.550f,0.550f}};

      // ⬐ Dielectrics – specular ≈ 0.04 (all channels)
      case MaterialPreset::GlassClear:return {{0.80f,0.90f,1.00f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::GlassBlue: return {{0.30f,0.50f,0.80f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::Emerald:   return {{0.075f,0.614f,0.075f}, {0.05f,0.05f,0.05f}}; // IOR≈1.57
      case MaterialPreset::Ruby:      return {{0.614f,0.041f,0.041f}, {0.08f,0.08f,0.08f}}; // IOR≈1.76
      case MaterialPreset::Obsidian:  return {{0.053f,0.056f,0.066f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::Rubber:    return {{0.02f,0.02f,0.02f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::WoodOak:   return {{0.71f,0.40f,0.11f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::WoodPine:  return {{0.89f,0.81f,0.61f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::PlasticRed:   return {{0.50f,0.00f,0.00f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::PlasticGreen: return {{0.00f,0.50f,0.00f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::PlasticBlue:  return {{0.00f,0.00f,0.50f}, {0.04f,0.04f,0.04f}};
      case MaterialPreset::PlasticYellow:return {{0.50f,0.50f,0.00f}, {0.04f,0.04f,0.04f}};
      default:
        return {{0.8f,0.8f,0.8f}, {0.04f,0.04f,0.04f}};
    }
  }

  inline vrt::MaterialPreset randomPreset()
    {
        // One PRNG per thread is inexpensive and avoids locking
        static thread_local std::mt19937 rng{ std::random_device{}() };

        // Assumes the enum values are contiguous and begin at 0
        constexpr int first = 0;
        constexpr int last  = static_cast<int>(vrt::MaterialPreset::PlasticYellow);

        std::uniform_int_distribution<int> dist(first, last);
        return static_cast<vrt::MaterialPreset>(dist(rng));
    }
}
