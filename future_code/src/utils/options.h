#ifndef MVS_OPTIONS_H
#define MVS_OPTIONS_H

#include "io_utils.h"
#include "mat.h"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

class Options {
 public:
  Options(const std::string& path) : filename(path){};
  bool load();

  std::string filename;
  std::string input_folder, output_folder;
  std::string img_folder, cam_folder;
  std::string depth_folder, normal_folder, cost_folder;
  std::string viz_folder;
  int max_size, min_size;
  int gpu_id;

  // Multi-scale parameters

  int cur_size, num_scales, cur_scale;
  bool upsample;

  // PatchMatch parameters

  bool use_planar_priors, planar_priors;
  int num_planar_iterations;
  bool use_median_filter;
  bool use_geom_cons, geom_cons;
  int num_geom_iterations;
  int num_images, num_photo_iterations, top_k, patch_size, radius_increment;
  float max_cost, sigma_color, sigma_spatial;

  // Point cloud parameters

  bool generate_pcl;
  int min_consistent;
  float max_error, max_diff, max_angle;
};

using OptionsPtr = std::shared_ptr<Options>;

#endif