option(CONFIG_ENABLE "Enable Configuration" ON)
if(CONFIG_ENABLE)
  set(MODELS_DIR "${CMAKE_SOURCE_DIR}/Models")
  set(SHADERS_DIR "${CMAKE_SOURCE_DIR}/data/shaders")
  set(TEXTURES_DIR "${CMAKE_SOURCE_DIR}/data/skybox")
endif()
configure_file(config.h.in config.h @ONLY)

# Specify the configuration file from which the header file will be generated
configure_file (config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h @ONLY)

