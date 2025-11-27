#include "kernel/testKernel.h"

__host__ __device__ inline int get_flat_index(int x, int y, int gridWidth) {
  int W = gridWidth;
  return y * W + x;
}

__global__ void testKernel(cuFloatComplex *d_array, int gridWidth,
                           int gridHeight, int time) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= gridWidth || y >= gridHeight) {
    return;
  }

  const auto flat_index = get_flat_index(x, y, gridWidth);

  const float center_x = gridWidth / 2.0f;
  const float center_y = gridHeight / 2.0f;
  const float scale = fminf(gridWidth, gridHeight) / 2.0f;

  const float nx = (x - center_x) / scale;
  const float ny = (center_y - y) / scale;

  const float r = sqrtf(nx * nx + ny * ny);

  const float theta = atan2f(ny, nx);

  const float spatial_freq = 15.0f;
  const float temporal_freq = 0.05f;
  const float rotation_speed = 0.5f;

  const float phase =
      (r * spatial_freq) + (theta * rotation_speed) + (time * temporal_freq);

  const float real_part = cosf(phase);
  const float imag_part = sinf(phase);

  d_array[flat_index] = make_cuFloatComplex(real_part, imag_part);
}
