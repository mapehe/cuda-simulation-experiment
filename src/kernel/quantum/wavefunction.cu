#include "kernel/math/linalg.cuh"
#include "kernel/quantum/quantumKernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

void normalizePsi(cuFloatComplex *d_psi, dim3 block, dim3 grid,
                  GaussianArgs args, Grid gridArgs) {
  const auto [width, height, L_x, L_y] = gridArgs;
  float dx = L_x / width;
  float dy = L_y / width;

  int numElements = width * height;

  thrust::device_ptr<cuFloatComplex> th_psi(d_psi);
  float sumSq =
      thrust::transform_reduce(th_psi, th_psi + numElements, SquareMagnitude(),
                               0.0f, thrust::plus<float>());

  float currentProbability = sumSq * dx * dy;

  if (currentProbability == 0.0f)
    return;

  float scaleFactor = 1.0f / sqrtf(currentProbability);

  thrust::transform(th_psi, th_psi + numElements, th_psi,
                    [scaleFactor] __device__(cuFloatComplex val) {
                      return make_cuFloatComplex(val.x * scaleFactor,
                                                 val.y * scaleFactor);
                    });

  cudaDeviceSynchronize();
}

__global__ void initGaussian(cuFloatComplex *d_psi, GaussianArgs args,
                             Grid grid) {
  const auto [width, height, L_x, L_y] = grid;
  const auto [x0, y0, sigma, amplitude] = args;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  int idx = get_flat_index({.width = width, .height = height});
  auto [nx, ny] = get_normalized_coords(i, j, grid);

  float2 pos = make_float2(nx, ny);
  float2 center = make_float2(args.x0, args.y0);
  float dist_sq = distanceSq(pos, center);

  float envelope = amplitude * expf(-dist_sq / (2.0f * sigma * sigma));

  d_psi[idx] = make_cuFloatComplex(envelope, 0);
}
