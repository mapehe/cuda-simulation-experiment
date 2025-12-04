#include "kernel/quantum/quantumKernels.cuh"

__global__ void initComplexPotential(cuComplex *d_V_tot, PotentialArgs args,
                                     Grid grid) {
  const auto [width, height, dx, dy, trapFreqSq, V_bias, r_0, sigma,
              absorb_strength, absorb_width] = args;
  int idx = get_flat_index({.width = width, .height = height});

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  auto [nx, ny] = get_normalized_coords(i, j, grid);

  float r = sqrtf(nx * nx + ny * ny);

  float v_harm = 0.5f * trapFreqSq * r * r;
  float v_waterfall = V_bias * tanhf((r - r_0) / sigma);
  float val_real = v_harm + v_waterfall + V_bias;

  float val_imag =
      -1.0f * absorb_strength * expf(-(r * r) / (absorb_width * absorb_width));

  d_V_tot[idx] = make_cuComplex(val_real, val_imag);
}
