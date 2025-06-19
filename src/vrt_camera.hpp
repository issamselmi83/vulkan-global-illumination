#ifndef __VULKAN_RAY_TRACING_CAMERA_HPP__
#define __VULKAN_RAY_TRACING_CAMERA_HPP__

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <iostream>

namespace vrt {
	class Camera {
	public:
		Camera(float fov, float aspect);
		~Camera();

		void move(GLFWwindow* window, float dt);

		void setPerspective(float fov, float aspect);

		const glm::mat4 getWorldTransform() const;
		const glm::mat4& getProjectionMatrix() const;

	private:
		const float MOVE_SPEED = 2.0f;
		const float LOOK_SPEED = 1.0f;

		glm::mat4 _projection;

		glm::vec3 _position;
		glm::vec3 _rotation;
	};
}

#endif