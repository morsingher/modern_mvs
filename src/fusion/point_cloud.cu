#include "point_cloud.h"

// TODO: not sure it's the right way but it works

cv::Vec3f rotateNormalToWorld(const cv::Vec3f& n, const Eigen::Matrix3d& R) {
  const Eigen::Vector3d n_cam(n[0], n[1], n[2]);
  const Eigen::Vector3d n_world = R.transpose() * n_cam;
  return cv::Vec3f(n_world.x(), n_world.y(), n_world.z());
}

// TODO: maybe try to break this into smaller functions?
// I don't know if it makes sense

bool generatePointCloud(const std::vector<FramePtr>& frames, OptionsPtr opt) {
  // Step 1: load MVS results
  // This is a tradeoff between memory and speed. If memory is a concern,
  // you should load data when they are needed instead of all at once

  std::vector<Mat3D> images;
  std::vector<Camera> cameras;
  std::vector<Mat2D> depths;
  std::vector<Mat3D> normals;
  std::vector<cv::Mat_<uchar>> masks;  // TODO: this can probably be bool

  std::cout << "Loading MVS results..." << std::endl;

  for (int i = 0; i < frames.size(); ++i) {
    const int ref = frames[i]->ref_id;

    const std::string img_path = getFilename(opt->img_folder, ref, ".jpg");
    Mat3D img(img_path);
    if (!img.readColor()) {
      std::cout << "Failed to read: " << img_path << std::endl;
      return false;
    }

    const std::string cam_path = getFilename(opt->cam_folder, ref, ".txt");
    Camera cam(cam_path);
    if (!cam.load()) {
      std::cout << "Failed to read: " << cam_path << std::endl;
      return false;
    }

    cam.width = img.width;
    cam.height = img.height;

    const std::string depth_path = getFilename(opt->depth_folder, ref, ".dmb");
    Mat2D depth(depth_path);
    if (!depth.readBinary()) {
      std::cout << "Failed to read: " << depth_path << std::endl;
      return false;
    }

    const std::string normal_path = getFilename(opt->normal_folder, ref, ".dmb");
    Mat3D normal(normal_path);
    if (!normal.readBinary()) {
      std::cout << "Failed to read: " << normal_path << std::endl;
      return false;
    }

    const int max_size = std::max(depth.width, depth.height);
    if (std::max(img.width, img.height) > max_size) {
      img.rescale(max_size);
      cam.rescale(img.width, img.height);
    }

    depths.push_back(depth);
    normals.push_back(normal);
    cameras.push_back(cam);
    images.push_back(img);
    masks.push_back(cv::Mat::zeros(depth.height, depth.width, CV_8UC1));
  }

  std::cout << "Loaded all the frames!" << std::endl;

  // Step 2: fuse all the frames into a unique point cloud

  PointCloud pcl;

  Timer timer("<PointCloud>");

  for (int i = 0; i < frames.size(); i++) {
    std::cout << "Fusing frame " << i << "..." << std::endl;

    const int cols = depths[i].width;
    const int rows = depths[i].height;
    const int num_src = frames[i]->src_ids.size();
    std::vector<cv::Point2i> used(num_src, cv::Point2i(-1, -1));  // Here or later?

    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        if (masks[i](row, col) != 1)  // If the current pixel has never been used
        {
          const float ref_depth = depths[i].getValue(row, col);
          cv::Vec3f ref_normal = cv::normalize(normals[i].getValue(row, col));
          ref_normal = rotateNormalToWorld(ref_normal, cameras[i].R);

          if (ref_depth > 0.0)  // If the depth is valid
          {
            const Eigen::Vector3d ref_pt = cameras[i].projectPixelToWorld(col, row, ref_depth);
            Eigen::Vector3d pcl_pt = ref_pt;
            cv::Vec3f pcl_normal = ref_normal;
            cv::Vec3f pcl_color = images[i].getValue(row, col);

            int num_consistent = 0;

            for (int j = 0; j < num_src; j++)  // Observe the 3D point in each neighbor
            {
              const int src_id = frames[i]->src_ids[j];
              const Eigen::Vector3d src_pix = cameras[src_id].observePoint(ref_pt);
              const int src_col = static_cast<int>(std::round(src_pix.x()));
              const int src_row = static_cast<int>(std::round(src_pix.y()));

              if (!cameras[src_id].outsideBounds(src_col, src_row) && masks[src_id](src_row, src_col) != 1) {
                const float src_depth = depths[src_id].getValue(src_row, src_col);
                cv::Vec3f src_normal = cv::normalize(normals[src_id].getValue(src_row, src_col));
                src_normal = rotateNormalToWorld(src_normal, cameras[src_id].R);

                if (src_depth > 0.0) {
                  const Eigen::Vector3d src_pt = cameras[src_id].projectPixelToWorld(src_col, src_row, src_depth);
                  const Eigen::Vector3d ref_pix = cameras[i].observePoint(src_pt);

                  const float reproj_error = std::sqrt(std::pow(col - ref_pix.x(), 2) + std::pow(row - ref_pix.y(), 2));
                  const float rel_depth_diff = std::fabs(ref_pix.z() - ref_depth) / ref_depth;
                  const float angle = std::acos(ref_normal.dot(src_normal));

                  if (reproj_error < opt->max_error && rel_depth_diff < opt->max_diff && angle < opt->max_angle) {
                    pcl_pt += src_pt;
                    pcl_normal += src_normal;
                    pcl_color += images[src_id].getValue(src_row, src_col);

                    used[j].x = src_col;
                    used[j].y = src_row;
                    num_consistent++;
                  }
                }
              }
            }

            if (num_consistent >= opt->min_consistent)  // Keep only consistent values
            {
              pcl_pt /= (num_consistent + 1.0);
              pcl_normal /= (num_consistent + 1.0);
              pcl_color /= (num_consistent + 1.0);

              Point final_pt;
              final_pt.x = pcl_pt.x();
              final_pt.y = pcl_pt.y();
              final_pt.z = pcl_pt.z();
              final_pt.r = static_cast<int>(pcl_color[2]);
              final_pt.g = static_cast<int>(pcl_color[1]);
              final_pt.b = static_cast<int>(pcl_color[0]);
              final_pt.normal_x = pcl_normal[0];
              final_pt.normal_y = pcl_normal[1];
              final_pt.normal_z = pcl_normal[2];

              pcl.push_back(final_pt);

              for (int j = 0; j < num_src; j++) {
                if (used[j].x != -1) {
                  masks[frames[i]->src_ids[j]](used[j].y, used[j].x) = 1;
                }
              }
            }
          }
        }
      }
    }
  }

  timer.stop();

  std::cout << "Saving the point cloud with " << pcl.points.size() << " points..." << std::endl;
  const std::string ply_path = opt->output_folder + "/point_cloud.ply";
  pcl::io::savePLYFileBinary(ply_path, pcl);

  return true;
}