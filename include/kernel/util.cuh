#ifndef UTIL_KERNELS
#define UTIL_KERNELS
#include <assert.h>

struct tmpGrid {
  int width;
  int height;
};

struct Grid {
  int width;
  int height;
  float L_x;
  float L_y;
};

__device__ __forceinline__ int get_flat_index(tmpGrid args) {
  const auto [width, height] = args;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  assert(i < width && j < height);

  return j * width + i;
}

struct Coords {
  float x;
  float y;
};

__device__ __forceinline__ Coords get_normalized_coords(int i, int j,
                                                        Grid grid) {
  const auto [width, height, L_x, L_y] = grid;

  const float x_uc = 2.0f * i / (width - 1.0f) - 1.0f;
  const float y_uc = 1.0f - 2.0f * j / (height - 1.0f);

  return {.x = L_x * x_uc, .y = L_y * y_uc};
}

#endif
