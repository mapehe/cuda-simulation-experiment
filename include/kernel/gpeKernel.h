#ifndef GPE_KERNEL_CUH
#define GPE_KERNEL_CUH

#include "config.h"
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

template <> struct SimulationData<CUDAKernelMode::GrossPitaevskii> {
  cuFloatComplex *d_psi;
  float *d_V;
  cuFloatComplex *d_expK;
  std::vector<cuFloatComplex> h_data;
  cufftHandle plan;

  dim3 grid;
  dim3 block;

  int width;
  int height;
  int iterations;
  int downloadFrequency;
  int downloadIterator;
  float dt;
  float g;
};

__global__ void initGaussian(cuFloatComplex *d_psi, int width, int height,
                             float dx, float dy, float x0, float y0,
                             float sigma, float kx, float ky, float amplitude);

void normalizePsi(SimulationData<CUDAKernelMode::GrossPitaevskii> &data,
                  int width, int height, float dx, float dy);

__global__ void initHarmonicTrap(float *d_V, int width, int height, float dx,
                                 float dy, float trapFreqSq);

__global__ void initKineticOperator(cuFloatComplex *d_expK, int width,
                                    int height, float dk_x, float dk_y,
                                    float dt);

__global__ void evolveRealSpace(cuFloatComplex *d_psi, float *d_V, int width,
                                int height, float g, float dt);

__global__ void evolveMomentumSpace(cuFloatComplex *d_psi,
                                    cuFloatComplex *d_expK, int width,
                                    int height, float scale);

__global__ void addGaussianObstacle(float *d_V, int width, int height, float dx,
                                    float dy, float x0, float y0, float sigma,
                                    float heightV);

template <> struct MemoryResource<CUDAKernelMode::GrossPitaevskii> {
  static SimulationData<CUDAKernelMode::GrossPitaevskii>
  allocate(const Params &p);

  static void free(SimulationData<CUDAKernelMode::GrossPitaevskii> &data);

  static void
  append_frame(const SimulationData<CUDAKernelMode::GrossPitaevskii> &data,
               std::vector<cuFloatComplex> &history);
};

template <> struct KernelLauncher<CUDAKernelMode::GrossPitaevskii> {
  static void launch(dim3 numBlocks, dim3 threadsPerBlock,
                     SimulationData<CUDAKernelMode::GrossPitaevskii> &data,
                     int t);
};

#endif
