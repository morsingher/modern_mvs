#ifndef MVS_PATCHMATCH_H
#define MVS_PATCHMATCH_H

#include "io_utils.h"
#include "mat.h"
#include "options.h"
#include "timer.h"

#include "cuda_kernels.h"

class PatchMatch {
 public:
  PatchMatch(FramePtr frame, OptionsPtr opt) : frame_h(frame), opt_h(opt){};
  ~PatchMatch();

  bool run();
  bool loadData();
  bool setCurrentResult();
  bool moveDataToCuda(cuda::device_t device);
  bool saveResults();

  FramePtr frame_h;

  OptionsPtr opt_h;
  cuda::memory::region_t opt_d;

  int num_images, num_pixels;
  int width, height;

  cuda::memory::device::unique_ptr<curandState[]> rand_d;

  std::vector<Camera> cameras_h;
  cuda::memory::region_t cameras_d;

  std::vector<Plane> planes_h;
  cuda::memory::region_t planes_d;

  std::vector<float> costs_h;
  cuda::memory::device::unique_ptr<float[]> costs_d;

  std::vector<Mat2D> images_h;
  TextureArray images_tex_h;
  cuda::memory::region_t images_tex_d;

  cuda::memory::device::unique_ptr<unsigned int[]> views_d;

  // Geometric consistency

  std::vector<Mat2D> depths_h;
  TextureArray depths_tex_h;
  cuda::memory::region_t depths_tex_d;
};

#endif