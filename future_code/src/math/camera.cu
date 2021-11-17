#include "camera.h"

bool Camera::load()
{
  std::ifstream file(filename);
  if (!file) {
    return false;
  }

  std::string line;
  std::getline(file, line);
  for (int i = 0; i < 3; i++) {
    file >> R(i, 0) >> R(i, 1) >> R(i, 2) >> t[i];
  }
  for (int i = 0; i < 4; i++) {
    std::getline(file, line);  // Skip dummy lines
  }
  for (int i = 0; i < 3; i++) {
    file >> K(i, 0) >> K(i, 1) >> K(i, 2);
  }
  K_inv = K.inverse();

  float depth_num, interval;  // Dummy, just for compatibility with MVSNet
  file >> min_depth >> interval >> depth_num >> max_depth;
  min_depth *= 0.6;  // TODO: probably better to make this configurable?
  max_depth *= 1.2;  // TODO: probably better to make this configurable?

  return true;
}

void Camera::rescale(const int new_width, const int new_height)
{
  const float s_x = new_width / static_cast<float>(width);
  const float s_y = new_height / static_cast<float>(height);
  K(0, 0) *= s_x;
  K(0, 2) *= s_x;
  K(1, 1) *= s_y;
  K(1, 2) *= s_y;
  K_inv = K.inverse();
  width = new_width;
  height = new_height;
}