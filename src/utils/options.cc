#include "options.h"

bool Options::load() {
  std::cout << std::endl << "Reading options from: " << filename << std::endl;

  FILE* fp = fopen(filename.c_str(), "r");
  if (fp == nullptr) {
    std::cout << "Failed to read " << filename << std::endl;
    return false;
  }

  char read_buffer[65536];
  rapidjson::FileReadStream is(fp, read_buffer, sizeof(read_buffer));
  rapidjson::Document d;
  d.ParseStream(is);
  fclose(fp);

  input_folder = d["input_folder"].GetString();
  output_folder = d["output_folder"].GetString();

  img_folder = input_folder + "images/";
  cam_folder = input_folder + "cameras/";
  depth_folder = output_folder + "depth/";
  normal_folder = output_folder + "normal/";
  cost_folder = output_folder + "cost/";
  viz_folder = output_folder + "viz/";

  max_size = d["max_size"].GetInt();
  gpu_id = d["gpu_id"].GetInt();

  // Check the existence of input data and create output directory

  if (!checkInputData(input_folder)) {
    std::cout << "Invalid input data!" << std::endl;
    return false;
  }

  createOutputDir(output_folder);

  // PatchMatch parameters

  use_median_filter = d["use_median_filter"].GetBool();
  use_geom_cons = d["use_geom_cons"].GetBool();
  geom_cons = false;
  num_geom_iterations = d["num_geom_iterations"].GetInt();
  num_photo_iterations = d["num_photo_iterations"].GetInt();
  top_k = d["top_k"].GetInt();
  patch_size = d["patch_size"].GetInt();
  radius_increment = d["radius_increment"].GetInt();
  max_cost = static_cast<float>(d["max_cost"].GetDouble());
  sigma_color = static_cast<float>(d["sigma_color"].GetDouble());
  sigma_spatial = static_cast<float>(d["sigma_spatial"].GetDouble());

  // Point cloud parameters

  generate_pcl = d["generate_pcl"].GetBool();
  min_consistent = d["min_consistent"].GetInt();
  max_error = static_cast<float>(d["max_error"].GetDouble());
  max_diff = static_cast<float>(d["max_diff"].GetDouble());
  max_angle = static_cast<float>(d["max_angle"].GetDouble()) * M_PI / 180.0;

  return true;
}