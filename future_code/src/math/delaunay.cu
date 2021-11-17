#include "delaunay.h"

Plane Triangle::computePlane(const Camera& cam) const
{
  const Eigen::Vector3d p1 = cam.projectPixel(v1.x, v1.y, d1);
  const Eigen::Vector3d p2 = cam.projectPixel(v2.x, v2.y, d2);
  const Eigen::Vector3d p3 = cam.projectPixel(v3.x, v3.y, d3);

  // Eigen version, not sure if it works

  // Eigen::Matrix<double, 3, 4> A;
  // A << pt1, pt2, pt3, Eigen::Vector3d(1.0, 1.0, 1.0);
  // const Eigen::Vector4d b = Eigen::Vector4d::Zero();
  // const Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

  // OpenCV version, I know it works

  cv::Mat_<float> A = cv::Mat::zeros(3, 4, CV_32FC1);
  cv::Mat_<float> x = cv::Mat::zeros(4, 1, CV_32FC1);
  for (int i = 0; i < 3; i++) {
    A(0, i) = p1[i];
    A(1, i) = p2[i];
    A(2, i) = p3[i];
    A(i, 3) = 1.0;
  }
  cv::SVD::solveZ(A, x);

  const Eigen::Vector3d normal(x(0, 0), x(1, 0), x(2, 0));
  const float dist = x(3, 0);
  const float norm = (dist > 0 ? normal.norm() : -normal.norm());

  return Plane(normal / norm, dist / norm);
}

float Triangle::getMaxLength() const
{
  const float len_1 = std::sqrt(std::pow(v1.x - v2.x, 2) + std::pow(v1.y - v2.y, 2));
  const float len_2 = std::sqrt(std::pow(v1.x - v3.x, 2) + std::pow(v1.y - v3.y, 2));
  const float len_3 = std::sqrt(std::pow(v2.x - v3.x, 2) + std::pow(v2.y - v3.y, 2));
  return std::max(len_1, std::max(len_2, len_3));
}

bool Delaunay::loadData()
{
  const std::string depth_path = getFilename(opt->depth_folder, frame->ref_id, ".dmb");
  depth = Mat2D(depth_path);
  if (!depth.readBinary()) {
    std::cout << "Failed to read: " << depth_path << std::endl;
    return false;
  }

  const std::string cost_path = getFilename(opt->cost_folder, frame->ref_id, ".dmb");
  cost = Mat2D(cost_path);
  if (!cost.readBinary()) {
    std::cout << "Failed to read: " << cost_path << std::endl;
    return false;
  }

  const std::string img_path = getFilename(opt->img_folder, frame->ref_id, ".jpg");
  img = Mat2D(img_path);
  if (!img.readGrayscale()) {
    std::cout << "Failed to read: " << img_path << std::endl;
    return false;
  }

  const std::string cam_path = getFilename(opt->cam_folder, frame->ref_id, ".txt");
  cam = Camera(cam_path);
  if (!cam.load()) {
    std::cout << "Failed to read: " << cam_path << std::endl;
    return false;
  }

  cam.width = img.width;
  cam.height = img.height;

  const int max_size = std::max(depth.width, depth.height);
  if (std::max(img.width, img.height) > max_size) {
    img.rescale(max_size);
    cam.rescale(img.width, img.height);
  }

  width = img.width;
  height = img.height;

  return true;
}

std::vector<cv::Point> Delaunay::findGoodPoints()
{
  std::vector<cv::Point> good_points;
  const int step_size = 5;

  for (int col = 0; col < width; col += step_size) {
    for (int row = 0; row < height; row += step_size) {
      float min_cost = opt->max_cost;
      cv::Point min_cost_point;
      const int col_bound = std::min(width, col + step_size);
      const int row_bound = std::min(height, row + step_size);
      for (int i = col; i < col_bound; i++) {
        for (int j = row; j < row_bound; j++) {
          const float cost_cur = cost.getValue(j, i);
          if (cost_cur < min_cost) {
            min_cost_point = cv::Point(i, j);
            min_cost = cost_cur;
          }
        }
      }
      if (min_cost < 0.1) {
        good_points.push_back(min_cost_point);
      }
    }
  }

  return good_points;
}

std::vector<Triangle> Delaunay::computeTriangulation(const std::vector<cv::Point>& points, const cv::Rect& bounds)
{
  std::vector<cv::Vec6f> tmp;
  cv::Subdiv2D sub(bounds);
  for (const auto& p : points) {
    sub.insert(cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
  }
  sub.getTriangleList(tmp);

  std::vector<Triangle> triangles;
  for (const auto& tri : tmp) {
    cv::Point v1(static_cast<int>(tri[0]), static_cast<int>(tri[1]));
    cv::Point v2(static_cast<int>(tri[2]), static_cast<int>(tri[3]));
    cv::Point v3(static_cast<int>(tri[4]), static_cast<int>(tri[5]));
    const float d1 = depth.getValue(v1.y, v1.x);
    const float d2 = depth.getValue(v2.y, v2.x);
    const float d3 = depth.getValue(v3.y, v3.x);
    triangles.push_back(Triangle(v1, v2, v3, d1, d2, d3));
  }
  return triangles;
}

bool Delaunay::computePlanarPriors()
{
  if (!loadData()) {
    std::cout << "Failed to load data for planar priors!" << std::endl;
    return false;
  }

  const cv::Rect img_bounds(0, 0, img.width, img.height);
  const auto good_points = findGoodPoints();
  const auto triangles = computeTriangulation(good_points, img_bounds);

  // Save triangulation as PNG (for debug)

  std::vector<cv::Mat> tri_out_vec = {img.data, img.data, img.data};
  cv::Mat tri_out;
  cv::merge(tri_out_vec, tri_out);
  for (const auto& t : triangles) {
    if (img_bounds.contains(t.v1) && img_bounds.contains(t.v2) && img_bounds.contains(t.v3)) {
      cv::line(tri_out, t.v1, t.v2, cv::Scalar(0, 0, 255));
      cv::line(tri_out, t.v1, t.v3, cv::Scalar(0, 0, 255));
      cv::line(tri_out, t.v2, t.v3, cv::Scalar(0, 0, 255));
    }
  }
  const std::string tri_path = getFilename(opt->viz_folder, frame->ref_id, "_tri.png");
  cv::imwrite(tri_path, tri_out);

  // Set actual priors

  const std::string mask_path = getFilename(opt->viz_folder, frame->ref_id, "_mask.png");
  mask = Mat2D(mask_path);
  mask.allocate(height, width, 1);

  int idx = 0;
  for (const auto& t : triangles) {
    if (img_bounds.contains(t.v1) && img_bounds.contains(t.v2) && img_bounds.contains(t.v3)) {
      const float max_edge_length = t.getMaxLength();
      const float step = 1.0 / max_edge_length;
      for (float p = 0; p < 1.0; p += step) {
        for (float q = 0; q < 1.0 - p; q += step) {
          int x = p * t.v1.x + q * t.v2.x + (1.0 - p - q) * t.v3.x;
          int y = p * t.v1.y + q * t.v2.y + (1.0 - p - q) * t.v3.y;
          mask.setValue(y, x, idx + 1.0);
        }
      }
      planes.push_back(t.computePlane(cam));
      idx++;
    }
  }

  // Invalidate priors with depth out of range and save for debug

  const std::string depth_prior_path = getFilename(opt->viz_folder, frame->ref_id, "_depth_prior.png");
  Mat2D depth_prior(depth_prior_path);
  depth_prior.allocate(height, width, 1);

  const std::string normal_prior_path = getFilename(opt->viz_folder, frame->ref_id, "_normal_prior.png");
  Mat3D normal_prior(normal_prior_path);
  normal_prior.allocate(height, width, 3);

  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      if (mask.getValue(row, col) > 0) {
        const Plane p = planes[mask.getValue(row, col) - 1];
        const Eigen::Vector3d n = cam.R.transpose() * p.normal;
        const float d = p.getDepth(cam, make_float2(col, row));
        if (d < cam.min_depth || d > cam.max_depth) {
          mask.setValue(row, col, 0);
        }
        else {
          depth_prior.setValue(row, col, d);
          normal_prior.setValue(row, col, cv::Vec3f(n.x(), n.y(), n.z()));
        }
      }
    }
  }

  depth_prior.writePng();
  normal_prior.writePng();
  mask.writePng();

  return true;
}