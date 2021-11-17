#include "patchmatch.h"

bool PatchMatch::loadData()
{
  std::string img_path = getFilename(opt_h->img_folder, frame_h->ref_id, ".jpg");
  Mat2D img(img_path);
  if (!img.readGrayscale()) {
    std::cout << "Failed to read: " << img_path << std::endl;
    return false;
  }
  images_h.push_back(img);

  std::string cam_path = getFilename(opt_h->cam_folder, frame_h->ref_id, ".txt");
  Camera cam(cam_path);
  if (!cam.load()) {
    std::cout << "Failed to read: " << cam_path << std::endl;
    return false;
  }
  cameras_h.push_back(cam);

  for (int i = 0; i < frame_h->src_ids.size(); i++) {
    img_path = getFilename(opt_h->img_folder, frame_h->src_ids[i], ".jpg");
    Mat2D img(img_path);
    if (!img.readGrayscale()) {
      std::cout << "Failed to read: " << img_path << std::endl;
      return false;
    }
    images_h.push_back(img);

    cam_path = getFilename(opt_h->cam_folder, frame_h->src_ids[i], ".txt");
    Camera cam(cam_path);
    if (!cam.load()) {
      std::cout << "Failed to read: " << cam_path << std::endl;
      return false;
    }
    cameras_h.push_back(cam);
  }

  // Rescale images, if needed

  for (int i = 0; i < images_h.size(); i++) {
    cameras_h[i].width = images_h[i].width;
    cameras_h[i].height = images_h[i].height;

    if (std::max(images_h[i].width, images_h[i].height) > opt_h->max_size) {
      images_h[i].rescale(opt_h->max_size);
      cameras_h[i].rescale(images_h[i].width, images_h[i].height);
    }
  }

  num_images = images_h.size();
  width = images_h[0].width;
  height = images_h[0].height;
  num_pixels = width * height;

  opt_h->num_images = num_images;

  // Read depths if geometric consistency is required

  if (opt_h->geom_cons) {
    std::string depth_path = getFilename(opt_h->depth_folder, frame_h->ref_id, ".dmb");
    Mat2D depth(depth_path);
    if (!depth.readBinary()) {
      std::cout << "Failed to read: " << depth_path << std::endl;
      return false;
    }
    depths_h.push_back(depth);

    for (int i = 0; i < frame_h->src_ids.size(); i++) {
      depth_path = getFilename(opt_h->depth_folder, frame_h->src_ids[i], ".dmb");
      Mat2D depth(depth_path);
      if (!depth.readBinary()) {
        std::cout << "Failed to read: " << depth_path << std::endl;
        return false;
      }
      depths_h.push_back(depth);
    }
  }

  return true;
}

bool PatchMatch::setCurrentResult()
{
  const std::string depth_path = getFilename(opt_h->depth_folder, frame_h->ref_id, ".dmb");
  Mat2D depth_cur(depth_path);
  if (!depth_cur.readBinary()) {
    std::cout << "Failed to read: " << depth_path << std::endl;
    return false;
  }

  const std::string cost_path = getFilename(opt_h->cost_folder, frame_h->ref_id, ".dmb");
  Mat2D cost_cur(cost_path);
  if (!cost_cur.readBinary()) {
    std::cout << "Failed to read: " << cost_path << std::endl;
    return false;
  }

  const std::string normal_path = getFilename(opt_h->normal_folder, frame_h->ref_id, ".dmb");
  Mat3D normal_cur(normal_path);
  if (!normal_cur.readBinary()) {
    std::cout << "Failed to read: " << normal_path << std::endl;
    return false;
  }

  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      const int idx = row * width + col;
      planes_h[idx] = Plane(normal_cur.getValue(row, col), depth_cur.getValue(row, col));
      costs_h[idx] = cost_cur.getValue(row, col);
    }
  }

  return true;
}

bool PatchMatch::moveDataToCuda(cuda::device_t device)
{
  std::cout << "Moving data to CUDA..." << std::endl;

  rand_d = cuda::memory::device::make_unique<curandState[]>(device, num_pixels);

  planes_h = std::vector<Plane>(num_pixels);
  planes_d = cuda::memory::device::allocate(device, num_pixels * sizeof(Plane));

  cameras_d = cuda::memory::device::allocate(device, num_images * sizeof(Camera));
  cuda::memory::copy(cameras_d.get(), cameras_h.data(), num_images * sizeof(Camera));

  costs_h = std::vector<float>(num_pixels);
  costs_d = cuda::memory::device::make_unique<float[]>(device, num_pixels);

  opt_d = cuda::memory::device::allocate(device, sizeof(Options));
  cuda::memory::copy(opt_d.get(), opt_h.get(), sizeof(Options));

  images_tex_h.setDataToTextureMemory(images_h);
  images_tex_d = cuda::memory::device::allocate(device, sizeof(TextureArray));
  cuda::memory::copy(images_tex_d.get(), &images_tex_h, sizeof(TextureArray));

  views_d = cuda::memory::device::make_unique<unsigned int[]>(device, num_pixels);

  if (opt_h->geom_cons) {
    depths_tex_h.setDataToTextureMemory(depths_h);
    depths_tex_d = cuda::memory::device::allocate(device, sizeof(TextureArray));
    cuda::memory::copy(depths_tex_d.get(), &depths_tex_h, sizeof(TextureArray));

    if (!setCurrentResult()) {
      std::cout << "Failed to set current result!" << std::endl;
      return false;
    }

    cuda::memory::copy(planes_d.get(), planes_h.data(), num_pixels * sizeof(Plane));
    cuda::memory::copy(costs_d.get(), costs_h.data(), num_pixels * sizeof(float));
  }

  return true;
}

bool PatchMatch::saveResults()
{
  // Get results

  const std::string cost_path = getFilename(opt_h->cost_folder, frame_h->ref_id, ".dmb");
  Mat2D cost(cost_path);
  cost.allocate(height, width, 1);

  const std::string depth_path = getFilename(opt_h->depth_folder, frame_h->ref_id, ".dmb");
  Mat2D depth(depth_path);
  depth.allocate(height, width, 1);

  const std::string normal_path = getFilename(opt_h->normal_folder, frame_h->ref_id, ".dmb");
  Mat3D normal(normal_path);
  normal.allocate(height, width, 3);

  for (int col = 0; col < width; ++col) {
    for (int row = 0; row < height; ++row) {
      const int idx = row * width + col;
      const Plane p = planes_h[idx];
      cost.setValue(row, col, costs_h[idx]);
      depth.setValue(row, col, p.dist);
      normal.setValue(row, col, cv::normalize(cv::Vec3f(p.normal.x(), p.normal.y(), p.normal.z())));
    }
  }

  if (opt_h->use_median_filter) {
    depth.medianFilter(5);
    normal.medianFilter(5);
  }

  // Write binary matrices

  if (!cost.writeBinary()) {
    std::cout << "Failed to write: " << cost_path << std::endl;
    return false;
  }

  if (!depth.writeBinary()) {
    std::cout << "Failed to write: " << depth_path << std::endl;
    return false;
  }

  if (!normal.writeBinary()) {
    std::cout << "Failed to write: " << normal_path << std::endl;
    return false;
  }

  // Save png for visualization

  const std::string viz_cost_path = getFilename(opt_h->viz_folder, frame_h->ref_id, "_cost.png");
  cost.writePng(viz_cost_path);

  const std::string viz_depth_path = getFilename(opt_h->viz_folder, frame_h->ref_id, "_depth.png");
  depth.writePng(viz_depth_path);

  const std::string viz_normal_path = getFilename(opt_h->viz_folder, frame_h->ref_id, "_normal.png");
  normal.writePng(viz_normal_path);

  return true;
}

// TODO: maybe break this into smaller functions?

bool PatchMatch::run()
{
  // Load input data

  std::cout << "Loading input data..." << std::endl;

  if (!loadData()) {
    std::cout << "Failed to read input data!" << std::endl;
    return false;
  }

  // Set CUDA device and move data

  if (cuda::device::count == 0) {
    std::cout << "Failed to find CUDA devices!" << std::endl;
    return false;
  }

  auto device = cuda::device::get(opt_h->gpu_id);
  std::cout << "Setting CUDA device with ID " << opt_h->gpu_id << " and name " << device.name() << std::endl;
  cuda::device::current::set(device);
  device.reset();

  if (!moveDataToCuda(device)) {
    std::cout << "Failed to move data to CUDA!" << std::endl;
    return false;
  }

  // Run kernels

  constexpr cuda::grid::block_dimension_t block_dim = 16;
  constexpr auto block_dims = cuda::grid::block_dimensions_t::square(block_dim);

  const cuda::grid::dimensions_t grid_dims = {
      cuda::grid::dimension_t((width + block_dim - 1) / block_dim),
      cuda::grid::dimension_t((height + block_dim - 1) / block_dim),
      1};

  const cuda::grid::dimensions_t grid_dims_cb = {
      cuda::grid::dimension_t((width + block_dim - 1) / block_dim),
      cuda::grid::dimension_t(((height / 2) + block_dim - 1) / block_dim),
      1};

  auto planes_ptr = static_cast<Plane*>(planes_d.get());
  auto cameras_ptr = static_cast<Camera*>(cameras_d.get());
  auto opt_ptr = static_cast<Options*>(opt_d.get());
  auto images_ptr = static_cast<TextureArray*>(images_tex_d.get());
  auto depths_ptr = static_cast<TextureArray*>(depths_tex_d.get());

  Timer timer("<PatchMatch>");

  std::cout << "Running initialization kernel..." << std::endl;

  cuda::launch(
      initialize,
      cuda::make_launch_config(grid_dims, block_dims),
      planes_ptr,
      cameras_ptr,
      costs_d.get(),
      rand_d.get(),
      opt_ptr,
      images_ptr,
      views_d.get());

  device.synchronize();

  int num_iterations = opt_h->num_photo_iterations;
  if (opt_h->geom_cons) {
    num_iterations = opt_h->num_geom_iterations;
  }

  for (int iter = 0; iter < num_iterations; iter++) {
    std::cout << "Running black-red kernels for iteration " << iter << "..." << std::endl;

    cuda::launch(
        checkerboardBlack,
        cuda::make_launch_config(grid_dims_cb, block_dims),
        planes_ptr,
        cameras_ptr,
        costs_d.get(),
        rand_d.get(),
        opt_ptr,
        images_ptr,
        depths_ptr,
        views_d.get(),
        iter);

    device.synchronize();

    cuda::launch(
        checkerboardRed,
        cuda::make_launch_config(grid_dims_cb, block_dims),
        planes_ptr,
        cameras_ptr,
        costs_d.get(),
        rand_d.get(),
        opt_ptr,
        images_ptr,
        depths_ptr,
        views_d.get(),
        iter);

    device.synchronize();
  }

  std::cout << "Running finalization kernel..." << std::endl;
  cuda::launch(getResults, cuda::make_launch_config(grid_dims, block_dims), planes_ptr, cameras_ptr);
  device.synchronize();

  timer.stop();

  // Get and save results

  std::cout << "Saving results..." << std::endl;
  cuda::memory::copy(planes_h.data(), planes_d.get(), num_pixels * sizeof(Plane));
  cuda::memory::copy(costs_h.data(), costs_d.get(), num_pixels * sizeof(float));
  if (!saveResults()) {
    std::cout << "Failed to save results!" << std::endl;
    return false;
  }
  std::cout << "PatchMatch done!" << std::endl;

  return true;
}

PatchMatch::~PatchMatch()
{
  cuda::memory::device::free(planes_d);
  cuda::memory::device::free(cameras_d);
  cuda::memory::device::free(opt_d);
  cuda::memory::device::free(images_tex_d);
}