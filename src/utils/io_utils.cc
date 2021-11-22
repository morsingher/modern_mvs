#include "io_utils.h"

std::vector<FramePtr> readFrames(const std::string& path) {
  std::cout << "Reading frames from: " << path << std::endl;

  std::vector<FramePtr> frames;
  std::ifstream file(path);

  int num_frames;
  file >> num_frames;

  for (int i = 0; i < num_frames; i++) {
    const auto frame = std::make_shared<Frame>();
    file >> frame->ref_id;

    int num_src_frames;
    file >> num_src_frames;

    for (int j = 0; j < num_src_frames; j++) {
      int id, score;
      file >> id >> score;
      frame->src_ids.push_back(id);
    }

    frames.push_back(frame);
  }

  return frames;
}

bool checkInputData(const std::string& folder) {
  for (const auto& sub : {"", "pair.txt", "images/", "cameras/"}) {
    if (!std::filesystem::exists(folder + sub)) {
      std::cout << folder + sub << " does not exist!" << std::endl;
      return false;
    }
  }

  return true;
}

void createOutputDir(const std::string& folder) {
  if (std::filesystem::exists(folder)) {
    std::filesystem::remove_all(folder);
  }
  std::filesystem::create_directory(folder);

  for (const auto& sub : {"depth/", "normal/", "cost/", "viz/"}) {
    std::filesystem::create_directory(folder + sub);
  }
}

std::string getFilename(const std::string& folder, const int id, const std::string& ext) {
  std::stringstream path;
  path << folder << std::setw(8) << std::setfill('0') << id << ext;
  return path.str();
}