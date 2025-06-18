// material_presets.hpp
#pragma once
#include <glm/vec3.hpp>

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
      case MaterialPreset::Gold:
        return { {1.000f,0.766f,0.336f}, {0.628f,0.555f,0.366f} };
      case MaterialPreset::Silver:
        return { {0.972f,0.960f,0.915f}, {0.915f,0.926f,0.945f} };
      case MaterialPreset::Copper:
        return { {0.955f,0.637f,0.538f}, {0.880f,0.790f,0.680f} };
      case MaterialPreset::Bronze:
        return { {0.714f,0.428f,0.181f}, {0.393f,0.271f,0.166f} };
      case MaterialPreset::Brass:
        return { {0.780f,0.568f,0.113f}, {0.992f,0.941f,0.807f} };
      case MaterialPreset::Aluminum:
        return { {0.913f,0.921f,0.925f}, {0.950f,0.960f,0.960f} };
      case MaterialPreset::Iron:
        return { {0.560f,0.570f,0.580f}, {0.580f,0.590f,0.600f} };
      case MaterialPreset::Chrome:
        return { {0.550f,0.550f,0.550f}, {0.950f,0.950f,0.950f} };
      case MaterialPreset::GlassClear:
        return { {0.8f,0.9f,1.0f}, {0.04f,0.04f,0.04f} };
      case MaterialPreset::GlassBlue:
        return { {0.3f,0.5f,0.8f}, {0.04f,0.04f,0.04f} };
      case MaterialPreset::Emerald:
        return { {0.075f,0.614f,0.075f}, {0.633f,0.727f,0.633f} };
      case MaterialPreset::Ruby:
        return { {0.614f,0.041f,0.041f}, {0.727f,0.626f,0.626f} };
      case MaterialPreset::Obsidian:
        return { {0.053f,0.056f,0.066f}, {0.182f,0.17f,0.225f} };
      case MaterialPreset::Rubber:
        return { {0.02f,0.02f,0.02f}, {0.4f,0.4f,0.4f} };
      case MaterialPreset::WoodOak:
        return { {0.71f,0.40f,0.11f}, {0.2f,0.2f,0.2f} };
      case MaterialPreset::WoodPine:
        return { {0.89f,0.81f,0.61f}, {0.15f,0.15f,0.15f} };
      case MaterialPreset::PlasticRed:
        return { {0.5f,0.0f,0.0f}, {0.7f,0.7f,0.7f} };
      case MaterialPreset::PlasticGreen:
        return { {0.0f,0.5f,0.0f}, {0.7f,0.7f,0.7f} };
      case MaterialPreset::PlasticBlue:
        return { {0.0f,0.0f,0.5f}, {0.7f,0.7f,0.7f} };
      case MaterialPreset::PlasticYellow:
        return { {0.5f,0.5f,0.0f}, {0.7f,0.7f,0.7f} };
      default:
        return { {0.8f,0.8f,0.8f}, {0.2f,0.2f,0.2f} };
    }
  }
}
