#ifndef MVS_COMMON_H
#define MVS_COMMON_H

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

struct Frame {
  int ref_id;
  std::vector<int> src_ids;
};

using FramePtr = std::shared_ptr<Frame>;

#endif