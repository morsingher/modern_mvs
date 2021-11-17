#ifndef MVS_DELAUNAY_H
#define MVS_DELAUNAY_H

#include "mat.h"
#include "options.h"
#include "plane.h"

class Triangle {
 public:
  Triangle(
      const cv::Point& v1_,
      const cv::Point& v2_,
      const cv::Point& v3_,
      const float d1_,
      const float d2_,
      const float d3_)
      : v1(v1_), v2(v2_), v3(v3_), d1(d1_), d2(d2_), d3(d3_){};

  Plane computePlane(const Camera& cam) const;
  float getMaxLength() const;

  cv::Point v1, v2, v3;
  float d1, d2, d3;
};

class Delaunay {
 public:
  Delaunay(FramePtr f, OptionsPtr o) : frame(f), opt(o){};

  bool loadData();
  std::vector<cv::Point> findGoodPoints();
  std::vector<Triangle> computeTriangulation(const std::vector<cv::Point>& points, const cv::Rect& bounds);
  bool computePlanarPriors();

  inline float getMask(const int row, const int col) const { return mask.getValue(row, col); }
  inline Plane getPlane(const int idx) const { return planes[idx]; }

  FramePtr frame;
  OptionsPtr opt;
  Mat2D mask, cost, depth, img;
  Camera cam;
  std::vector<Plane> planes;
  int width, height;
};

#endif