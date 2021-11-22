#include "patchmatch.h"
#include "point_cloud.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: <executable> <config_file>" << std::endl;
    return EXIT_FAILURE;
  }

  Timer timer("<Main>");

  // Parse configuration file

  const std::string config_file(argv[1]);
  OptionsPtr opt = std::make_shared<Options>(config_file);
  if (!opt->load()) {
    std::cout << "Failed to load options!" << std::endl;
    return EXIT_FAILURE;
  }

  // Read pair.txt and build the list of frames

  const std::string frames_file = opt->input_folder + "pair.txt";
  const std::vector<FramePtr> frames = readFrames(frames_file);
  if (frames.empty()) {
    std::cout << "Failed to load frames!" << std::endl;
    return EXIT_FAILURE;
  }

  // For each frame create PatchMatch object with IDs and params, then run()

  std::cout << std::endl << "Running MVS with " << frames.size() << " frames" << std::endl;

  // Iterations with only photometric cost

  for (const auto& frame : frames) {
    std::cout << std::endl << "Running photometric PatchMatch for frame " << frame->ref_id << std::endl;
    PatchMatch mvs(frame, opt);
    if (!mvs.run()) {
      std::cout << "Failed to run MVS for frame " << frame->ref_id << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Iterations with geometric consistency check

  if (opt->use_geom_cons) {
    opt->geom_cons = true;
    for (const auto& frame : frames) {
      std::cout << std::endl << "Running geometric PatchMatch for frame " << frame->ref_id << std::endl;
      PatchMatch mvs(frame, opt);
      if (!mvs.run()) {
        std::cout << "Failed to run MVS for frame " << frame->ref_id << std::endl;
        return EXIT_FAILURE;
      }
    }
    opt->geom_cons = false;
  }

  // Generate point cloud from depths and normals

  if (opt->generate_pcl) {
    std::cout << std::endl << "Generating the point cloud..." << std::endl;
    if (!generatePointCloud(frames, opt)) {
      std::cout << "Failed to create the point cloud!" << std::endl;
      return EXIT_FAILURE;
    }
  }

  timer.stop();

  return EXIT_SUCCESS;
}