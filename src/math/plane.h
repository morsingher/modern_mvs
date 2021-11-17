#ifndef MVS_PLANE_H
#define MVS_PLANE_H

#include "camera.h"

using Pixel = float2;

class Plane {
 public:
  Eigen::Vector3d normal;
  float dist;

  CUDA_FHD Plane() : dist(0.0) { normal.setZero(); };
  CUDA_FHD Plane(const Eigen::Vector3d& n, const float d) : normal(n), dist(d){};
  CUDA_FHD Plane(const cv::Vec3f& n, const float d) : dist(d) { normal = Eigen::Vector3d(n[0], n[1], n[2]); };

  CUDA_FD void setRandom(const Camera& cam, const Pixel pix, curandState* rand)
  {
    const float depth = curand_uniform(rand) * (cam.max_depth - cam.min_depth) + cam.min_depth;

    float q1 = 1.0f, q2 = 1.0f, s = 2.0f;
    while (s >= 1.0f) {
      q1 = 2.0f * curand_uniform(rand) - 1.0f;
      q2 = 2.0f * curand_uniform(rand) - 1.0f;
      s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrt(1.0f - s);
    normal = Eigen::Vector3d(2.0f * q1 * sq, 2.0f * q2 * sq, 1.0f - 2.0f * s);
    const Eigen::Vector3d view_dir = cam.getViewDirection(pix.x, pix.y);
    if (normal.dot(view_dir) > 0.0) {
      normal = -normal;
    }
    normal = normal.normalized();

    setDistanceFromDepth(cam, pix, depth);
  }

  CUDA_FHD void setDistanceFromDepth(const Camera& cam, const Pixel pix, const float depth)
  {
    const Eigen::Vector3d point = cam.projectPixel(pix.x, pix.y, depth);
    dist = -normal.dot(point);
  }

  CUDA_FHD void setDistanceAsDepth(const Camera& cam, const Pixel pix)
  {
    const Eigen::Vector3d point = cam.getViewDirection(pix.x, pix.y);
    dist = -dist / normal.dot(point);
  }

  CUDA_FHD float getDepth(const Camera& cam, const Pixel pix) const
  {
    const Eigen::Vector3d point = cam.getViewDirection(pix.x, pix.y);
    return -dist / normal.dot(point);
  }

  CUDA_FHD void rotateNormalToWorld(const Camera& cam) { normal = cam.R.transpose() * normal; }
  CUDA_FHD void rotateNormalToCamera(const Camera& cam) { normal = cam.R * normal; }
};

#endif