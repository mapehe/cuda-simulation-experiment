#include "kernel/quantum/quantumKernels.cuh"

__global__ void evolveRealSpace(cuFloatComplex *d_psi, cuFloatComplex *d_V,
                                int width, int height, float g, float dt) {
  int idx = get_flat_index({.width = width, .height = height});

  cuFloatComplex psi = d_psi[idx];
  cuFloatComplex V_c = d_V[idx];

  float V_real = cuCrealf(V_c);
  float V_imag = cuCimagf(V_c);

  float n = cuCrealf(psi) * cuCrealf(psi) + cuCimagf(psi) * cuCimagf(psi);

  float angle = -(V_real + g * n) * dt;
  float c, s;
  sincosf(angle, &s, &c);
  cuFloatComplex phasor = make_cuFloatComplex(c, s);

  float decay_factor = expf(V_imag * dt);

  cuFloatComplex psi_rotated = cuCmulf(psi, phasor);

  d_psi[idx] = make_cuFloatComplex(cuCrealf(psi_rotated) * decay_factor,
                                   cuCimagf(psi_rotated) * decay_factor);
}

__global__ void evolveMomentumSpace(cuFloatComplex *d_psi,
                                    cuFloatComplex *d_expK, int width,
                                    int height, float scale) {
  int idx = get_flat_index({.width = width, .height = height});

  cuFloatComplex psi = d_psi[idx];
  cuFloatComplex kOp = d_expK[idx];
  cuFloatComplex res = cuCmulf(psi, kOp);

  res.x *= scale;
  res.y *= scale;

  d_psi[idx] = res;
}

__global__ void initKineticOperator(cuFloatComplex *d_expK,
                                    KineticInitArgs args) {
  const auto [width, height, dk_x, dk_y, dt] = args;
  int idx = get_flat_index({.width = width, .height = height});
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  float kx_val = (i <= width / 2) ? (float)i : (float)(i - width);
  float ky_val = (j <= height / 2) ? (float)j : (float)(j - height);

  float kx = kx_val * dk_x;
  float ky = ky_val * dk_y;

  float k2 = kx * kx + ky * ky;
  float angle = -0.5f * k2 * dt;

  float c, s;
  sincosf(angle, &s, &c);
  d_expK[idx] = make_cuFloatComplex(c, s);
}
