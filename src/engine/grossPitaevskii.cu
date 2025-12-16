#include "engine/grossPitaevskii.cuh"
#include "io.h"

std::tuple<GaussianArgs, PotentialArgs, KineticInitArgs, Grid>
GrossPitaevskiiEngine::createSimulationArgs(const Params &p, float dt) const {

  float dx = ((float)p.grossPitaevskii.L) / p.grossPitaevskii.gridWidth;
  float dy = ((float)p.grossPitaevskii.L) / p.grossPitaevskii.gridHeight;

  float L_x = p.grossPitaevskii.L;
  float L_y = p.grossPitaevskii.L;

  float dk_x = (2.0f * M_PI) / L_x;
  float dk_y = (2.0f * M_PI) / L_y;

  GaussianArgs gArgs = {.x0 = p.grossPitaevskii.x0,
                        .y0 = p.grossPitaevskii.y0,
                        .sigma = p.grossPitaevskii.sigma,
                        .amplitude = p.grossPitaevskii.amp};

  PotentialArgs pArgs = {.width = params.grossPitaevskii.gridWidth,
                         .height = params.grossPitaevskii.gridHeight,
                         .dx = dx,
                         .dy = dy,
                         .trapFreqSq = p.grossPitaevskii.trapStr,
                         .V_bias = p.grossPitaevskii.V_bias,
                         .r_0 = p.grossPitaevskii.r_0,
                         .sigma = p.grossPitaevskii.sigma2,
                         .absorb_strength = p.grossPitaevskii.absorbStrength,
                         .absorb_width = p.grossPitaevskii.absorbWidth};

  KineticInitArgs kArgs = {.width = p.grossPitaevskii.gridWidth,
                           .height = p.grossPitaevskii.gridHeight,
                           .dk_x = dk_x,
                           .dk_y = dk_y,
                           .dt = dt};

  Grid grid = {.width = p.grossPitaevskii.gridWidth,
               .height = p.grossPitaevskii.gridHeight,
               .L_x = L_x,
               .L_y = L_y};

  return {gArgs, pArgs, kArgs, grid};
}

GrossPitaevskiiEngine::GrossPitaevskiiEngine(const Params &p)
    : ComputeEngine(p) {

  grid = dim3(p.grossPitaevskii.threadsPerBlockX,
              p.grossPitaevskii.threadsPerBlockY);
  block = dim3((p.grossPitaevskii.gridWidth + grid.x - 1) / grid.x,
               (p.grossPitaevskii.gridHeight + grid.y - 1) / grid.y);

  cufftPlan2d(&plan, params.grossPitaevskii.gridHeight,
              params.grossPitaevskii.gridWidth, CUFFT_C2C);

  size_t num_pixels =
      params.grossPitaevskii.gridWidth * params.grossPitaevskii.gridHeight;
  size_t size_bytes = num_pixels * sizeof(cuFloatComplex);

  cudaMalloc(&d_psi, size_bytes);
  cudaMemset(d_psi, 0, size_bytes);

  cudaMalloc(&d_V, size_bytes);
  cudaMemset(d_V, 0, size_bytes);

  cudaMalloc(&d_expK, size_bytes);
  cudaMemset(d_expK, 0, size_bytes);

  const auto [gArgs, pArgs, kArgs, gridArgs] =
      createSimulationArgs(p, params.grossPitaevskii.dt);

  initGaussian<<<grid, block>>>(d_psi, gArgs, gridArgs);
  cudaDeviceSynchronize();

  normalizePsi(d_psi, block, grid, gArgs, gridArgs);
  cudaDeviceSynchronize();

  initComplexPotential<<<grid, block>>>(d_V, pArgs, gridArgs);
  cudaDeviceSynchronize();

  initKineticOperator<<<grid, block>>>(d_expK, kArgs);
  cudaDeviceSynchronize();
}

GrossPitaevskiiEngine::~GrossPitaevskiiEngine() {
  if (d_psi) {
    cudaFree(d_psi);
    d_psi = nullptr;
  }

  if (d_V) {
    cudaFree(d_V);
    d_V = nullptr;
  }

  if (d_expK) {
    cudaFree(d_expK);
    d_expK = nullptr;
  }

  cufftDestroy(plan);
}

void GrossPitaevskiiEngine::appendFrame(std::vector<cuFloatComplex> &history) {
  size_t frame_elements =
      params.grossPitaevskii.gridWidth * params.grossPitaevskii.gridHeight;
  size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);
  size_t old_size = history.size();

  history.resize(old_size + frame_elements);
  cuFloatComplex *host_destination = history.data() + old_size;
  cudaMemcpy(host_destination, d_psi, frame_bytes, cudaMemcpyDeviceToHost);
}

void GrossPitaevskiiEngine::solveStep(int t) {
  float fft_scale = 1.0f / (float)(params.grossPitaevskii.gridWidth *
                                   params.grossPitaevskii.gridHeight);

  evolveRealSpace<<<grid, block>>>(d_psi, d_V, params.grossPitaevskii.gridWidth,
                                   params.grossPitaevskii.gridHeight,
                                   params.grossPitaevskii.g,
                                   params.grossPitaevskii.dt / 2.0f);
  cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD);
  evolveMomentumSpace<<<grid, block>>>(
      d_psi, d_expK, params.grossPitaevskii.gridWidth,
      params.grossPitaevskii.gridHeight, fft_scale);
  cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE);
  evolveRealSpace<<<grid, block>>>(d_psi, d_V, params.grossPitaevskii.gridWidth,
                                   params.grossPitaevskii.gridHeight,
                                   params.grossPitaevskii.g,
                                   params.grossPitaevskii.dt / 2.0f);
  cudaDeviceSynchronize();
}

int GrossPitaevskiiEngine::getDownloadFrequency() {
  return params.grossPitaevskii.downloadFrequency;
}

int GrossPitaevskiiEngine::getTotalSteps() {
  return params.grossPitaevskii.iterations;
}

void GrossPitaevskiiEngine::saveResults(const std::string &filename) {
  saveToBinaryJSON(
      {.filename = filename,
       .data = historyData,
       .width = params.grossPitaevskii.gridWidth,
       .height = params.grossPitaevskii.gridHeight,
       .iterations = params.grossPitaevskii.iterations,
       .downloadFrequency = params.grossPitaevskii.downloadFrequency,
       .header = params.grossPitaevskii});
}
