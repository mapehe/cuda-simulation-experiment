#ifndef QUANTUM_KERNELS_H
#define QUANTUM_KERNELS_H

#include "kernel/util.cuh"
#include <cuComplex.h>
#include <cuda_runtime.h>

struct GaussianArgs {
  float x0;
  float y0;
  float sigma;
  float amplitude;
};

struct PotentialArgs {
  int width;
  int height;
  float dx;
  float dy;
  float trapFreqSq;
  float V_bias;
  float r_0;
  float sigma;
  float absorb_strength;
  float absorb_width;
};

struct KineticInitArgs {
  int width;
  int height;
  float dk_x;
  float dk_y;
  float dt;
};

struct SquareMagnitude {
  __host__ __device__ float operator()(const cuFloatComplex &x) const {
    return cuCrealf(x) * cuCrealf(x) + cuCimagf(x) * cuCimagf(x);
  }
};

__global__ void initGaussian(cuFloatComplex *d_psi, GaussianArgs args,
                             Grid grid);

void normalizePsi(cuFloatComplex *d_psi, dim3 block, dim3 grid,
                  GaussianArgs args, Grid gridArgs);

__global__ void initComplexPotential(cuComplex *d_V_tot, PotentialArgs args,
                                     Grid grid);

__global__ void initKineticOperator(cuFloatComplex *d_expK,
                                    KineticInitArgs args);

__global__ void evolveRealSpace(cuFloatComplex *d_psi, cuFloatComplex *d_V,
                                int width, int height, float g, float dt);

__global__ void evolveMomentumSpace(cuFloatComplex *d_psi,
                                    cuFloatComplex *d_expK, int width,
                                    int height, float scale);

#endif
