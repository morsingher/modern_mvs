#ifndef MVS_MAT_H
#define MVS_MAT_H

#include "common.h"

// Just a utility for saving PNGs

inline cv::Vec3f operator-(const cv::Vec3f& v, const float s) { return cv::Vec3f(v[0] - s, v[1] - s, v[2] - s); }

template <typename T>
class Mat {
 public:
  Mat(){};
  Mat(const std::string& path) : filename(path){};

  std::string filename;
  int height, width, depth;
  cv::Mat_<T> data;

  template <typename U = T, typename = std::enable_if_t<std::is_same<float, U>::value>>
  bool readGrayscale()
  {
    data = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    height = data.rows;
    width = data.cols;
    depth = data.channels();
    return !data.empty();
  }

  template <typename U = T, typename = std::enable_if_t<std::is_same<cv::Vec3f, U>::value>>
  bool readColor()
  {
    data = cv::imread(filename, cv::IMREAD_COLOR);
    height = data.rows;
    width = data.cols;
    depth = data.channels();
    return !data.empty();
  }

  inline T getValue(const int row, const int col) const { return data(row, col); }
  inline void setValue(const int row, const int col, const T& value) { data(row, col) = value; }
  inline void medianFilter(const int k) { cv::medianBlur(data, data, k); }

  bool readBinary();
  bool writeBinary();
  void writePng();
  void writePng(const std::string& path_out);
  void rescale(const int max_size);
  void allocate(const int h, const int w, const int d);
};

using Mat2D = Mat<float>;
using Mat3D = Mat<cv::Vec3f>;

template <typename T>
void Mat<T>::writePng()
{
  writePng(filename);
}

template <typename T>
void Mat<T>::writePng(const std::string& path_out)
{
  cv::Mat_<T> data_out = cv::Mat::zeros(height, width, CV_MAKETYPE(CV_32F, depth));

  double min, max;
  cv::minMaxLoc(data, &min, &max);
  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      data_out(row, col) = 255.0 * (data(row, col) - min) / (max - min);
    }
  }

  cv::imwrite(path_out, data_out);
}

template <typename T>
bool Mat<T>::readBinary()
{
  std::fstream text_file(filename, std::ios::in);
  if (!text_file) {
    std::cout << "Failed to open text: " << filename << std::endl;
    return false;
  }
  char dummy;
  text_file >> width >> dummy >> height >> dummy >> depth >> dummy;
  std::streampos pos = text_file.tellg();
  text_file.close();

  data = cv::Mat::zeros(height, width, CV_MAKETYPE(CV_32F, depth));
  std::fstream bin_file(filename, std::ios::in | std::ios::binary);
  if (!bin_file) {
    std::cout << "Failed to open bin: " << filename << std::endl;
    return false;
  }
  bin_file.seekg(pos);
  bin_file.read(reinterpret_cast<char*>(data.data), sizeof(float) * depth * height * width);

  return true;
}

template <typename T>
bool Mat<T>::writeBinary()
{
  std::fstream text_file(filename, std::ios::out);
  if (!text_file) {
    std::cout << "Failed to open text: " << filename << std::endl;
    return false;
  }
  text_file << width << "&" << height << "&" << depth << "&";
  text_file.close();

  std::fstream bin_file(filename, std::ios::out | std::ios::binary | std::ios::app);
  if (!bin_file) {
    std::cout << "Failed to open bin: " << filename << std::endl;
    return false;
  }
  bin_file.write(reinterpret_cast<char*>(data.ptr(0)), (data.dataend - data.datastart));
  bin_file.close();

  return true;
}

template <typename T>
void Mat<T>::rescale(const int max_size)
{
  const float s_x = static_cast<float>(max_size) / width;
  const float s_y = static_cast<float>(max_size) / height;
  const float s = std::min(s_x, s_y);
  const int new_cols = std::round(s * width);
  const int new_rows = std::round(s * height);
  cv::resize(data, data, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
  height = new_rows;
  width = new_cols;
}

template <typename T>
void Mat<T>::allocate(const int h, const int w, const int d)
{
  height = h;
  width = w;
  depth = d;
  data = cv::Mat::zeros(h, w, CV_MAKETYPE(CV_32F, d));
}

#endif