#ifndef MVS_POINT_CLOUD_H
#define MVS_POINT_CLOUD_H

// Workaround for old versions of PCL, otherwise building with CUDA doesn't work
// https://github.com/PointCloudLibrary/pcl/issues/2597

#include <boost/lexical_cast.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>

// Actual PCL headers

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include "mat.h"
#include "options.h"
#include "plane.h"
#include "timer.h"

using Point = pcl::PointXYZRGBNormal;
using PointCloud = pcl::PointCloud<Point>;
using PointCloudPtr = pcl::PointCloud<Point>::Ptr;

bool generatePointCloud(const std::vector<FramePtr>& frames, OptionsPtr opt);

#endif