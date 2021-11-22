#ifndef MVS_CAMERA_H
#define MVS_CAMERA_H

#include "common.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <cuda/runtime_api.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#define CUDA_FHD __forceinline__ __host__ __device__
#define CUDA_FD __forceinline__ __device__

class Camera {
 public:
  Camera(){};
  Camera(const std::string& path) : filename(path) {
    R.setIdentity();
    t.setZero();
    K.setIdentity();
    K_inv.setIdentity();
  };

  bool load();
  void rescale(const int new_width, const int new_height);

  std::string filename;

  Eigen::Matrix3d K, K_inv, R;
  Eigen::Vector3d t;

  int height, width;
  float min_depth, max_depth;

  CUDA_FHD Eigen::Vector3d observePoint(const Eigen::Vector3d& point) const {
    const Eigen::Vector3d p_cam = R * point + t;
    const float depth = p_cam.z();
    const float u = K(0, 0) * (p_cam.x() / depth) + K(0, 2);
    const float v = K(1, 1) * (p_cam.y() / depth) + K(1, 2);
    return Eigen::Vector3d(u, v, depth);
  }

  // TODO: I think this could be just return depth * K_inv * Eigen::Vector3d(u, v, 1.0)

  CUDA_FHD Eigen::Vector3d projectPixel(const int u, const int v, const float depth) const {
    Eigen::Vector3d point((u - K(0, 2)) / K(0, 0), (v - K(1, 2)) / K(1, 1), 1.0);
    return depth * point;
  }

  CUDA_FHD Eigen::Vector3d projectPixelToWorld(const int u, const int v, const float depth) const {
    Eigen::Vector3d point_cam = projectPixel(u, v, depth);
    return R.transpose() * (point_cam - t);
  }

  CUDA_FHD Eigen::Vector3d getViewDirection(const int u, const int v) const { return projectPixel(u, v, 1.0); }
  CUDA_FHD bool outsideBounds(const int u, const int v) const { return u < 0 || v < 0 || u >= width || v >= height; }
};

#endif