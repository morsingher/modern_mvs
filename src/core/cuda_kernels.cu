#include "cuda_kernels.h"

// Utilities

CUDA_FD void sortVector(float* d, const int n) {
  int j;
  for (int i = 1; i < n; i++) {
    float tmp = d[i];
    for (j = i; j >= 1 && tmp < d[j - 1]; j--) {
      d[j] = d[j - 1];
    }
    d[j] = tmp;
  }
}

CUDA_FD int findMinCostIndex(const float* costs, const int n) {
  float min_cost = costs[0];
  int min_cost_idx = 0;
  for (int idx = 1; idx < n; ++idx) {
    if (costs[idx] <= min_cost) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

CUDA_FD void setBit(unsigned int& input, const unsigned int n) { input |= (unsigned int)(1 << n); }

CUDA_FD int isSet(unsigned int input, const unsigned int n) { return (input >> n) & 1; }

CUDA_FD void transformPDFToCDF(float* probs, const int num_probs) {
  float prob_sum = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    prob_sum += probs[i];
  }
  const float inv_prob_sum = 1.0f / prob_sum;

  float cum_prob = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    const float prob = probs[i] * inv_prob_sum;
    cum_prob += prob;
    probs[i] = cum_prob;
  }
}

// TODO: I have no idea why this does not work when uncommenting the line below
// return plane.getDepth(cam, pix);

CUDA_FD float getDepthFromPlane(const Camera& cam, const Plane& plane, const Pixel pix) {
  const Eigen::Vector3d point = cam.getViewDirection(pix.x, pix.y);
  return -plane.dist / plane.normal.dot(point);
}

CUDA_FD Eigen::Vector3d perturbNormal(const Camera& camera, const Pixel pix, const Plane& plane_cur, curandState* rand,
                                      const float perturbation) {
  const float a1 = (curand_uniform(rand) - 0.5f) * perturbation;
  const float a2 = (curand_uniform(rand) - 0.5f) * perturbation;
  const float a3 = (curand_uniform(rand) - 0.5f) * perturbation;

  const float sin_a1 = sin(a1);
  const float sin_a2 = sin(a2);
  const float sin_a3 = sin(a3);
  const float cos_a1 = cos(a1);
  const float cos_a2 = cos(a2);
  const float cos_a3 = cos(a3);

  Eigen::Matrix3d R_pert;
  R_pert << cos_a2 * cos_a3, cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3, sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2,
      cos_a2 * sin_a3, cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3, cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1, -sin_a2,
      cos_a2 * sin_a1, cos_a1 * cos_a2;

  Eigen::Vector3d n_pert = R_pert * plane_cur.normal;

  const Eigen::Vector3d view_dir = camera.getViewDirection(pix.x, pix.y);
  if (n_pert.dot(view_dir) >= 0.0) {
    n_pert = plane_cur.normal;
  }

  return n_pert.normalized();
}

// TODO: Why can't this be simplified further?
// const Eigen::Vector3d t_rel = src_cam.t - R_rel * ref_cam.t;

CUDA_FD Eigen::Matrix3d computeHomography(const Camera& ref_cam, const Camera& src_cam, const Plane& plane) {
  const Eigen::Vector3d C_ref = -ref_cam.R.transpose() * ref_cam.t;
  const Eigen::Vector3d C_src = -src_cam.R.transpose() * src_cam.t;
  const Eigen::Vector3d C_rel = C_ref - C_src;

  const Eigen::Matrix3d R_rel = src_cam.R * ref_cam.R.transpose();
  const Eigen::Vector3d t_rel = src_cam.R * C_rel;

  const Eigen::Matrix3d homo = src_cam.K * (R_rel - t_rel * plane.normal.transpose() / plane.dist) * ref_cam.K_inv;

  return homo;
}

CUDA_FD Pixel projectWithHomography(const Eigen::Matrix3d& homo, const Pixel pix) {
  const Eigen::Vector3d vec(pix.x, pix.y, 1.0);
  const Eigen::Vector3d proj = homo * vec;
  return make_float2(proj.x() / proj.z(), proj.y() / proj.z());
}

CUDA_FD float computeBilateralWeight(const float color_dist, const float spatial_dist, const Options* opt) {
  const float color_norm = 2.0 * opt->sigma_color * opt->sigma_color;
  const float spatial_norm = 2.0 * opt->sigma_spatial * opt->sigma_spatial;
  return exp(-(spatial_dist / spatial_norm) - (color_dist / color_norm));
}

CUDA_FD float computeBilateralNCC(const cudaTextureObject_t ref_img, const Camera& ref_cam,
                                  const cudaTextureObject_t src_img, const Camera& src_cam, const Pixel pix,
                                  const Plane& plane, const Options* opt) {
  const int radius = opt->patch_size / 2;

  const Eigen::Matrix3d homo = computeHomography(ref_cam, src_cam, plane);
  const Pixel src_pt = projectWithHomography(homo, pix);
  if (src_cam.outsideBounds(src_pt.x, src_pt.y)) {
    return opt->max_cost;
  }

  float sum_ref = 0.0f;
  float sum_ref_ref = 0.0f;
  float sum_src = 0.0f;
  float sum_src_src = 0.0f;
  float sum_ref_src = 0.0f;
  float bilateral_weight_sum = 0.0f;
  const float ref_pix_center = tex2D<float>(ref_img, pix.x + 0.5f, pix.y + 0.5f);

  for (int i = -radius; i < radius + 1; i += opt->radius_increment) {
    float sum_ref_row = 0.0f;
    float sum_src_row = 0.0f;
    float sum_ref_ref_row = 0.0f;
    float sum_src_src_row = 0.0f;
    float sum_ref_src_row = 0.0f;
    float bilateral_weight_sum_row = 0.0f;

    for (int j = -radius; j < radius + 1; j += opt->radius_increment) {
      const Pixel ref_pt = make_float2(pix.x + i, pix.y + j);
      const Pixel src_pt = projectWithHomography(homo, ref_pt);

      if (ref_cam.outsideBounds(ref_pt.x, ref_pt.y) || src_cam.outsideBounds(src_pt.x, src_pt.y)) {
        return opt->max_cost;
      }

      const float ref_pix = tex2D<float>(ref_img, ref_pt.x + 0.5, ref_pt.y + 0.5);
      const float src_pix = tex2D<float>(src_img, src_pt.x + 0.5, src_pt.y + 0.5);

      const float color_dist = fabs(ref_pix - ref_pix_center);
      const float spatial_dist = sqrt((float)(i * i + j * j));
      const float weight = computeBilateralWeight(color_dist, spatial_dist, opt);

      sum_ref_row += weight * ref_pix;
      sum_ref_ref_row += weight * ref_pix * ref_pix;
      sum_src_row += weight * src_pix;
      sum_src_src_row += weight * src_pix * src_pix;
      sum_ref_src_row += weight * ref_pix * src_pix;
      bilateral_weight_sum_row += weight;
    }

    sum_ref += sum_ref_row;
    sum_ref_ref += sum_ref_ref_row;
    sum_src += sum_src_row;
    sum_src_src += sum_src_src_row;
    sum_ref_src += sum_ref_src_row;
    bilateral_weight_sum += bilateral_weight_sum_row;
  }

  const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
  sum_ref *= inv_bilateral_weight_sum;
  sum_ref_ref *= inv_bilateral_weight_sum;
  sum_src *= inv_bilateral_weight_sum;
  sum_src_src *= inv_bilateral_weight_sum;
  sum_ref_src *= inv_bilateral_weight_sum;

  const float var_ref = sum_ref_ref - sum_ref * sum_ref;
  const float var_src = sum_src_src - sum_src * sum_src;

  const float min_var = 1e-5f;
  if (var_ref < min_var || var_src < min_var) {
    return opt->max_cost;
  } else {
    const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
    const float var_ref_src = sqrt(var_ref * var_src);
    return max(0.0f, min(opt->max_cost, 1.0f - covar_src_ref / var_ref_src));
  }
}

CUDA_FD float computeTopKCost(const cudaTextureObject_t* images, const Camera* cameras, const Pixel pix,
                              const Plane& plane, unsigned int* views, const Options* opt) {
  float cost_vector[MAX_IMAGES] = {opt->max_cost};
  float cost_vector_copy[MAX_IMAGES] = {opt->max_cost};
  int cost_count = 0, num_valid_views = 0;

  for (int i = 1; i < opt->num_images; ++i) {
    const float c = computeBilateralNCC(images[0], cameras[0], images[i], cameras[i], pix, plane, opt);
    cost_vector[i - 1] = c;
    cost_vector_copy[i - 1] = c;
    cost_count++;
    if (c < opt->max_cost) {
      num_valid_views++;
    }
  }

  sortVector(cost_vector, cost_count);
  *views = 0;

  const int top_k = min(num_valid_views, opt->top_k);
  if (top_k > 0) {
    float cost = 0.0f;
    for (int i = 0; i < top_k; ++i) {
      cost += cost_vector[i];
    }
    float cost_threshold = cost_vector[top_k - 1];
    for (int i = 0; i < opt->num_images - 1; ++i) {
      if (cost_vector_copy[i] <= cost_threshold) {
        setBit(*views, i);
      }
    }
    return cost / top_k;
  }

  return opt->max_cost;
}

CUDA_FD void computeMultiViewCost(const cudaTextureObject_t* images, const Camera* cameras, const Pixel pix,
                                  const Plane& plane, float* cost_vector, const Options* opt) {
  for (int i = 1; i < opt->num_images; ++i) {
    cost_vector[i - 1] = computeBilateralNCC(images[0], cameras[0], images[i], cameras[i], pix, plane, opt);
  }
}

CUDA_FD float computeGeometricCost(const cudaTextureObject_t src_depth_tex, const Camera& ref_cam,
                                   const Camera& src_cam, const Plane& plane, const Pixel pix) {
  const float max_cost = 5.0f;

  const float ref_depth = getDepthFromPlane(ref_cam, plane, pix);
  const Eigen::Vector3d ref_pt = ref_cam.projectPixelToWorld(pix.x, pix.y, ref_depth);

  const Eigen::Vector3d src_pix = src_cam.observePoint(ref_pt);
  const float src_depth = tex2D<float>(src_depth_tex, src_pix.x() + 0.5, src_pix.y() + 0.5);

  if (src_depth == 0.0) {
    return max_cost;
  }

  const Eigen::Vector3d src_pt = src_cam.projectPixelToWorld(src_pix.x(), src_pix.y(), src_depth);
  const Eigen::Vector3d ref_pix = ref_cam.observePoint(src_pt);

  const float diff_col = pix.x - ref_pix.x();
  const float diff_row = pix.y - ref_pix.y();
  return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

CUDA_FD void refinePlanes(const cudaTextureObject_t* images, const cudaTextureObject_t* depths_tex,
                          const Camera* cameras, Plane* plane, float* depth, float* cost, curandState* rand,
                          const float* view_weights, const float weight_norm, const Pixel pix, const Options* opt) {
  const float perturbation = 0.02f;

  // Random plane

  Plane p_rand;
  p_rand.setRandom(cameras[0], pix, rand);
  const Eigen::Vector3d n_rand = p_rand.normal;
  const float d_rand = getDepthFromPlane(cameras[0], p_rand, pix);

  // Current plane

  const Eigen::Vector3d n_cur = plane->normal;
  const float d_cur = *depth;

  // Perturbed plane

  float d_pert = d_cur;
  const float min_d_pert = (1.0 - perturbation) * d_pert;
  const float max_d_pert = (1.0 + perturbation) * d_pert;
  do {
    d_pert = curand_uniform(rand) * (max_d_pert - min_d_pert) + min_d_pert;
  } while (d_pert < cameras[0].min_depth && d_pert > cameras[0].max_depth);
  const Eigen::Vector3d n_pert = perturbNormal(cameras[0], pix, *plane, rand, perturbation * M_PI);

  // Check all the combinations

  const int num_planes = 5;
  float depths[num_planes] = {d_rand, d_cur, d_rand, d_cur, d_pert};
  Eigen::Vector3d normals[num_planes] = {n_cur, n_rand, n_rand, n_pert, n_cur};

  for (int i = 0; i < num_planes; ++i) {
    Plane new_plane(normals[i], depths[i]);
    new_plane.setDistanceFromDepth(cameras[0], pix, depths[i]);

    float cost_vector[MAX_IMAGES] = {opt->max_cost};
    computeMultiViewCost(images, cameras, pix, new_plane, cost_vector, opt);

    float temp_cost = 0.0f;
    for (int j = 0; j < opt->num_images - 1; ++j) {
      if (view_weights[j] > 0) {
        if (opt->geom_cons) {
          temp_cost += view_weights[j] * (cost_vector[j] + computeGeometricCost(depths_tex[j + 1], cameras[0],
                                                                                cameras[j + 1], new_plane, pix));
        } else {
          temp_cost += view_weights[j] * cost_vector[j];
        }
      }
    }
    temp_cost /= weight_norm;

    const float new_depth = getDepthFromPlane(cameras[0], new_plane, pix);
    if (new_depth >= cameras[0].min_depth && new_depth <= cameras[0].max_depth && temp_cost < *cost) {
      *depth = new_depth;
      *plane = new_plane;
      *cost = temp_cost;
    }
  }
}

// TODO: the code is clear but I feel it could be more compact
// It's not as trivial as it seems, since every case must be handled differently
// A naive idea would be to write 8 separate functions

CUDA_FD void checkerboardPropagation(const cudaTextureObject_t* images, const cudaTextureObject_t* depths_tex,
                                     const Camera* cameras, Plane* planes, float* costs, curandState* rand,
                                     unsigned int* views, const Pixel pix, const Options* opt, const int iter) {
  if (cameras[0].outsideBounds(pix.x, pix.y)) {
    return;
  }

  const int width = cameras[0].width;
  const int height = cameras[0].height;
  const int idx = pix.y * width + pix.x;

  // Step 1: Adaptive Checkerboard Sampling
  // Find the minimum cost plane for each of the regions defined in ACMH paper
  // Legend:
  // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far,
  // 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far

  float cost_array[8][MAX_IMAGES] = {opt->max_cost};
  bool flag[8] = {false};

  float min_cost;
  int min_cost_idx;

  // up_far

  int up_far = idx - 3 * width;
  if (pix.y > 2) {
    flag[1] = true;
    min_cost = costs[up_far];
    min_cost_idx = up_far;
    for (int i = 1; i < 11; ++i) {
      if (pix.y > 2 + 2 * i) {
        const int tmp_idx = up_far - 2 * i * width;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    up_far = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[up_far], cost_array[1], opt);
  }

  // down_far

  int down_far = idx + 3 * width;
  if (pix.y < height - 3) {
    flag[3] = true;
    min_cost = costs[down_far];
    min_cost_idx = down_far;
    for (int i = 1; i < 11; ++i) {
      if (pix.y < height - 3 - 2 * i) {
        const int tmp_idx = down_far + 2 * i * width;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    down_far = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[down_far], cost_array[3], opt);
  }

  // left_far

  int left_far = idx - 3;
  if (pix.x > 2) {
    flag[5] = true;
    min_cost = costs[left_far];
    min_cost_idx = left_far;
    for (int i = 1; i < 11; ++i) {
      if (pix.x > 2 + 2 * i) {
        const int tmp_idx = left_far - 2 * i;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    left_far = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[left_far], cost_array[5], opt);
  }

  // right_far

  int right_far = idx + 3;
  if (pix.x < width - 3) {
    flag[7] = true;
    min_cost = costs[right_far];
    min_cost_idx = right_far;
    for (int i = 1; i < 11; ++i) {
      if (pix.x < width - 3 - 2 * i) {
        const int tmp_idx = right_far + 2 * i;
        if (min_cost < costs[tmp_idx]) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    right_far = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[right_far], cost_array[7], opt);
  }

  // up_near

  int up_near = idx - width;
  if (pix.y > 0) {
    flag[0] = true;
    min_cost = costs[up_near];
    min_cost_idx = up_near;
    for (int i = 0; i < 3; ++i) {
      if (pix.y > 1 + i && pix.x > i) {
        const int tmp_idx = up_near - (1 + i) * width - i;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
      if (pix.y > 1 + i && pix.x < width - 1 - i) {
        const int tmp_idx = up_near - (1 + i) * width + i;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    up_near = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[up_near], cost_array[0], opt);
  }

  // down_near

  int down_near = idx + width;
  if (pix.y < height - 1) {
    flag[2] = true;
    min_cost = costs[down_near];
    min_cost_idx = down_near;
    for (int i = 0; i < 3; ++i) {
      if (pix.y < height - 2 - i && pix.x > i) {
        const int tmp_idx = down_near + (1 + i) * width - i;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
      if (pix.y < height - 2 - i && pix.x < width - 1 - i) {
        const int tmp_idx = down_near + (1 + i) * width + i;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    down_near = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[down_near], cost_array[2], opt);
  }

  // left_near

  int left_near = idx - 1;
  if (pix.x > 0) {
    flag[4] = true;
    min_cost = costs[left_near];
    min_cost_idx = left_near;
    for (int i = 0; i < 3; ++i) {
      if (pix.x > 1 + i && pix.y > i) {
        const int tmp_idx = left_near - (1 + i) - i * width;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
      if (pix.x > 1 + i && pix.y < height - 1 - i) {
        const int tmp_idx = left_near - (1 + i) + i * width;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    left_near = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[left_near], cost_array[4], opt);
  }

  // right_near

  int right_near = idx + 1;
  if (pix.x < width - 1) {
    flag[6] = true;
    min_cost = costs[right_near];
    min_cost_idx = right_near;
    for (int i = 0; i < 3; ++i) {
      if (pix.x < width - 2 - i && pix.y > i) {
        const int tmp_idx = right_near + (1 + i) - i * width;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
      if (pix.x < width - 2 - i && pix.y < height - 1 - i) {
        const int tmp_idx = right_near + (1 + i) + i * width;
        if (costs[tmp_idx] < min_cost) {
          min_cost = costs[tmp_idx];
          min_cost_idx = tmp_idx;
        }
      }
    }
    right_near = min_cost_idx;
    computeMultiViewCost(images, cameras, pix, planes[right_near], cost_array[6], opt);
  }

  const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

  // Step 2 - Multi-hypothesis Joint View Selection
  // Compute the final multi-view weighted cost

  // View selection priors assign a higher probability to views that are selected
  // in immediately neighboring pixels (eq 10 in ACMP paper)

  float view_selection_priors[MAX_IMAGES] = {0.0f};
  int neighbor_positions[4] = {idx - width, idx + width, idx - 1, idx + 1};
  for (int i = 0; i < 4; ++i) {
    if (flag[2 * i]) {
      for (int j = 0; j < opt->num_images - 1; ++j) {
        if (isSet(views[neighbor_positions[i]], j) == 1) {
          view_selection_priors[j] += 0.9f;
        } else {
          view_selection_priors[j] += 0.1f;
        }
      }
    }
  }

  // Analysis of the cost matrix M (eq 1 ACMM paper)
  // Assign a view weight to each image based on multi-hypothesis selection

  float sampling_probs[MAX_IMAGES] = {0.0f};
  float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));  // tau(t), eq 2 ACMM paper

  // For each image
  for (int i = 0; i < opt->num_images - 1; i++) {
    int count_good = 0, count_bad = 0;
    float tmpw = 0;
    // For each new plane
    for (int j = 0; j < 8; j++) {
      if (cost_array[j][i] < cost_threshold) {
        tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));  // C(m_ij), eq 3 ACMM paper
        count_good++;
      }
      if (cost_array[j][i] > 1.2f)  // tau_1
      {
        count_bad++;
      }
    }

    if (count_good > 2 && count_bad < 3) {
      sampling_probs[i] = tmpw / count_good;  // w(I_j) for each I_j in S_t, eq 4 ACMM paper
    } else if (count_bad < 3) {
      sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
    }

    sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];  // eq 11 ACMP paper
  }

  // Monte Carlo sampling to compute weights from probabilities (see below eq 11 in ACMP paper)

  float view_weights[MAX_IMAGES] = {0.0f};
  transformPDFToCDF(sampling_probs, opt->num_images - 1);
  for (int sample = 0; sample < 15; ++sample) {
    const float rand_prob = curand_uniform(&rand[idx]) - FLT_EPSILON;
    for (int image_id = 0; image_id < opt->num_images - 1; ++image_id) {
      const float prob = sampling_probs[image_id];
      if (prob > rand_prob) {
        view_weights[image_id] += 1.0f;
        break;
      }
    }
  }

  // View selection

  unsigned int temp_views = 0;
  float weight_norm = 0;
  for (int i = 0; i < opt->num_images - 1; ++i) {
    if (view_weights[i] > 0) {
      setBit(temp_views, i);
      weight_norm += view_weights[i];
    }
  }

  // Final view-weighted cost for each new plane: m_photo(p, h_i)

  float final_costs[8] = {0.0f};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < opt->num_images - 1; ++j) {
      if (view_weights[j] > 0) {
        if (opt->geom_cons) {
          if (flag[i]) {
            final_costs[i] += view_weights[j] * (cost_array[i][j] +
                                                 0.1 * computeGeometricCost(depths_tex[j + 1], cameras[0],
                                                                            cameras[j + 1], planes[positions[i]], pix));
          } else {
            final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1 * 5.0);
          }
        } else {
          final_costs[i] += view_weights[j] * cost_array[i][j];
        }
      }
    }
    final_costs[i] /= weight_norm;
  }

  min_cost_idx = findMinCostIndex(final_costs, 8);

  // Re-compute the cost of the current hypothesis with the new view weights

  float cost_vector_cur[MAX_IMAGES] = {opt->max_cost};
  computeMultiViewCost(images, cameras, pix, planes[idx], cost_vector_cur, opt);
  float cost_cur = 0.0f;
  for (int i = 0; i < opt->num_images - 1; ++i) {
    if (opt->geom_cons) {
      cost_cur +=
          view_weights[i] * (cost_vector_cur[i] + 0.1f * computeGeometricCost(depths_tex[i + 1], cameras[0],
                                                                              cameras[i + 1], planes[idx], pix));
    } else {
      cost_cur += view_weights[i] * cost_vector_cur[i];
    }
  }
  cost_cur /= weight_norm;
  costs[idx] = cost_cur;
  float depth_cur = getDepthFromPlane(cameras[0], planes[idx], pix);

  if (flag[min_cost_idx]) {
    const float depth_new = getDepthFromPlane(cameras[0], planes[positions[min_cost_idx]], pix);
    const float cost_new = final_costs[min_cost_idx];
    if (depth_new >= cameras[0].min_depth && depth_new <= cameras[0].max_depth && cost_new < cost_cur) {
      depth_cur = depth_new;
      planes[idx] = planes[positions[min_cost_idx]];
      costs[idx] = cost_new;
      views[idx] = temp_views;
    }
  }

  refinePlanes(images, depths_tex, cameras, &planes[idx], &depth_cur, &costs[idx], &rand[idx], view_weights,
               weight_norm, pix, opt);
}

// Actual kernels

__global__ void initialize(Plane* planes, const Camera* cameras, float* costs, curandState* rand, const Options* opt,
                           TextureArray* images, unsigned int* views) {
  const Pixel pix = make_float2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if (cameras[0].outsideBounds(pix.x, pix.y)) {
    return;
  }

  const int idx = pix.y * cameras[0].width + pix.x;
  curand_init(clock64(), pix.y, pix.x, &rand[idx]);

  if (opt->geom_cons) {
    planes[idx].setDistanceFromDepth(cameras[0], pix, planes[idx].dist);
  } else {
    planes[idx].setRandom(cameras[0], pix, &rand[idx]);
  }

  costs[idx] = computeTopKCost(images->data, cameras, pix, planes[idx], &views[idx], opt);
}

__global__ void checkerboardBlack(Plane* planes, const Camera* cameras, float* costs, curandState* rand,
                                  const Options* opt, TextureArray* images, TextureArray* depths, unsigned int* views,
                                  const int iter) {
  Pixel pix = make_float2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  pix.y = (threadIdx.x % 2 == 0) ? (pix.y * 2) : (pix.y * 2 + 1);
  checkerboardPropagation(images->data, depths->data, cameras, planes, costs, rand, views, pix, opt, iter);
}

__global__ void checkerboardRed(Plane* planes, const Camera* cameras, float* costs, curandState* rand,
                                const Options* opt, TextureArray* images, TextureArray* depths, unsigned int* views,
                                const int iter) {
  Pixel pix = make_float2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  pix.y = (threadIdx.x % 2 == 0) ? (pix.y * 2 + 1) : (pix.y * 2);
  checkerboardPropagation(images->data, depths->data, cameras, planes, costs, rand, views, pix, opt, iter);
}

// TODO: this kernel can be removed. When you save results, just get distance from depth.

__global__ void getResults(Plane* planes, const Camera* cameras) {
  const Pixel pix = make_float2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if (cameras[0].outsideBounds(pix.x, pix.y)) {
    return;
  }

  const int idx = pix.y * cameras[0].width + pix.x;

  planes[idx].setDistanceAsDepth(cameras[0], pix);
}