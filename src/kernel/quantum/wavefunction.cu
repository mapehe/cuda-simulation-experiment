#include "kernel/quantum/quantumKernels.cuh"
#include "kernel/util.cuh"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

void normalizePsi(cuFloatComplex *d_psi, dim3 block, dim3 grid,
                  GaussianArgs args) {
  int width = args.width;
  int height = args.height;
  float dx = args.dx;
  float dy = args.dy;

  int numElements = width * height;

  thrust::device_ptr<cuFloatComplex> th_psi(d_psi);
  float sumSq =
      thrust::transform_reduce(th_psi, th_psi + numElements, SquareMagnitude(),
                               0.0f, thrust::plus<float>());

  float currentProbability = sumSq * dx * dy;

  if (currentProbability == 0.0f) return;

  float scaleFactor = 1.0f / sqrtf(currentProbability);

  thrust::transform(th_psi, th_psi + numElements, th_psi,
      [scaleFactor] __device__ (cuFloatComplex val) {
          return make_cuFloatComplex(val.x * scaleFactor, val.y * scaleFactor);
      });

  cudaDeviceSynchronize();
}

__global__ void initGaussian(cuFloatComplex *d_psi, GaussianArgs args) {
  const auto [width, height, dx, dy, x0, y0, sigma, kx, ky, amplitude] = args;

  int idx = get_flat_index({.width = width, .height = height});
  auto [nx, ny] = get_normalized_coords({.width = width, .height = height});

  float dist_sq = (nx - x0) * (nx - x0) + (ny - y0) * (ny - y0);
  float envelope = amplitude * expf(-dist_sq / (2.0f * sigma * sigma));

  float phase_angle = kx * nx + ky * ny;
  float cos_phase, sin_phase;
  sincosf(phase_angle, &sin_phase, &cos_phase);

  d_psi[idx] = make_cuFloatComplex(envelope * cos_phase, envelope * sin_phase);
}
