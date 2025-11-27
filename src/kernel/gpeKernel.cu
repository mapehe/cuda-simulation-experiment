#include "kernel/gpeKernel.h"

struct SquareMagnitude {
  __host__ __device__ float operator()(const cuFloatComplex &x) const {
    return cuCrealf(x) * cuCrealf(x) + cuCimagf(x) * cuCimagf(x);
  }
};

__global__ void scaleWavefunction(cuFloatComplex *d_psi, int totalElements,
                                  float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalElements) {
    d_psi[idx].x *= scale;
    d_psi[idx].y *= scale;
  }
}

__global__ void initKineticOperator(cuFloatComplex *d_expK, int width,
                                    int height, float dk_x, float dk_y,
                                    float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = j * width + i;

  if (i >= width || j >= height)
    return;

  float kx;
  if (i <= width / 2) {
    kx = i * dk_x;
  } else {
    kx = (i - width) * dk_x;
  }

  float ky;
  if (j <= height / 2) {
    ky = j * dk_y;
  } else {
    ky = (j - height) * dk_y;
  }

  float k2 = kx * kx + ky * ky;
  float angle = -0.5f * k2 * dt;

  float c, s;
  sincosf(angle, &s, &c);
  d_expK[idx] = make_cuFloatComplex(c, s);
}

void normalizePsi(SimulationData<CUDAKernelMode::GrossPitaevskii> &data,
                  int width, int height, float dx, float dy) {
  int numElements = width * height;

  thrust::device_ptr<cuFloatComplex> th_psi(data.d_psi);
  float sumSq =
      thrust::transform_reduce(th_psi, th_psi + numElements, SquareMagnitude(),
                               0.0f, thrust::plus<float>());

  float currentProbability = sumSq * dx * dy;

  if (currentProbability == 0.0f)
    return; // Safety check
  float scaleFactor = 1.0f / sqrtf(currentProbability);

  printf("Initial Probability: %f. Scaling by: %f\n", currentProbability,
         scaleFactor);

  scaleWavefunction<<<data.grid, data.block>>>(data.d_psi, numElements,
                                               scaleFactor);
  cudaDeviceSynchronize();
}

__global__ void initHarmonicTrap(float *d_V, int width, int height, float dx,
                                 float dy, float trapFreqSq) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = y * width + x;

  if (x >= width || y >= height) {
    return;
  }

  const float center_x = width / 2.0f;
  const float center_y = height / 2.0f;
  const float scale = fminf(width, height) / 2.0f;

  const float nx = (x - center_x) / scale;
  const float ny = (center_y - y) / scale;

  float r2 = nx * nx + ny * ny;
  d_V[idx] = 0.5f * trapFreqSq * r2;
}

__global__ void initGaussian(cuFloatComplex *d_psi, int width, int height,
                             float dx, float dy, float x0, float y0,
                             float sigma, float kx, float ky, float amplitude) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = y * width + x;

  if (x >= width || y >= height) {
    return;
  }

  const float center_x = width / 2.0f;
  const float center_y = height / 2.0f;
  const float scale = fminf(width, height) / 2.0f;

  const float nx = (x - center_x) / scale;
  const float ny = (center_y - y) / scale;

  float dist_sq = (nx - x0) * (nx - x0) + (ny - y0) * (ny - y0);
  float envelope = amplitude * expf(-dist_sq / (2.0f * sigma * sigma));

  float phase_angle = kx * nx + ky * ny;
  float cos_phase, sin_phase;
  sincosf(phase_angle, &sin_phase, &cos_phase);

  d_psi[idx] = make_cuFloatComplex(envelope * cos_phase, envelope * sin_phase);
}

__global__ void addGaussianObstacle(float *d_V, int width, int height, float dx,
                                    float dy, float x0, float y0, float sigma,
                                    float heightV) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = y * width + x;

  if (x >= width || y >= height) {
    return;
  }

  const float center_x = width / 2.0f;
  const float center_y = height / 2.0f;
  const float scale = fminf(width, height) / 2.0f;

  const float nx = (x - center_x) / scale;
  const float ny = (center_y - y) / scale;

  float dist_sq = (nx - x0) * (nx - x0) + (ny - y0) * (ny - y0);

  d_V[idx] += heightV * expf(-dist_sq / (2.0f * sigma * sigma));
}

__global__ void evolveRealSpace(cuFloatComplex *d_psi, float *d_V, int width,
                                int height, float g, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = j * width + i;

  if (i >= width || j >= height)
    return;

  cuFloatComplex psi = d_psi[idx];
  float V = d_V[idx];

  float n = cuCrealf(psi) * cuCrealf(psi) + cuCimagf(psi) * cuCimagf(psi);
  float angle = -(V + g * n) * dt;
  float c, s;
  sincosf(angle, &s, &c);
  cuFloatComplex phasor = make_cuFloatComplex(c, s);

  d_psi[idx] = cuCmulf(psi, phasor);
}

__global__ void evolveMomentumSpace(cuFloatComplex *d_psi,
                                    cuFloatComplex *d_expK, int width,
                                    int height, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = j * width + i;

  if (i >= width || j >= height)
    return;

  cuFloatComplex psi = d_psi[idx];
  cuFloatComplex kOp = d_expK[idx];
  cuFloatComplex res = cuCmulf(psi, kOp);

  res.x *= scale;
  res.y *= scale;

  d_psi[idx] = res;
}

SimulationData<CUDAKernelMode::GrossPitaevskii>
MemoryResource<CUDAKernelMode::GrossPitaevskii>::allocate(const Params &p) {
  SimulationData<CUDAKernelMode::GrossPitaevskii> data;
  data.width = p.gridWidth;
  data.height = p.gridHeight;
  data.iterations = p.iterations;
  data.downloadFrequency = p.downloadFrequency;
  data.downloadIterator = 1;
  data.dt = p.dt;
  data.g = p.g;
  cufftPlan2d(&data.plan, data.height, data.width, CUFFT_C2C);

  size_t num_pixels = p.gridWidth * p.gridHeight;

  size_t size_psi = p.gridWidth * p.gridHeight * sizeof(cuFloatComplex);
  cudaMalloc(&data.d_psi, size_psi);
  cudaMemset(data.d_psi, 0, size_psi);

  size_t size_V = num_pixels * sizeof(float);
  cudaMalloc(&data.d_V, size_V);
  cudaMemset(data.d_V, 0, size_V);

  size_t size_K = num_pixels * sizeof(cuFloatComplex);
  cudaMalloc(&data.d_expK, size_K);
  cudaMemset(data.d_expK, 0, size_K);

  std::cout << "[Memory] Allocated Resources for grid " << p.gridWidth << "x"
            << p.gridHeight << ":\n"
            << " - Wavefunction (RW)\n"
            << " - Potential Grid (R)\n"
            << " - Kinetic Operator (R)\n"
            << std::endl;

  data.grid = dim3(p.threadsPerBlockX, p.threadsPerBlockY);
  data.block = dim3((p.gridWidth + data.grid.x - 1) / data.grid.x,
                    (p.gridHeight + data.grid.y - 1) / data.grid.y);

  float dx = p.L / p.gridWidth;
  float dy = p.L / p.gridHeight;
  float L_x = p.gridWidth * dx;
  float L_y = p.gridHeight * dy;
  float dk_x = (2.0f * M_PI) / L_x;
  float dk_y = (2.0f * M_PI) / L_y;

  initGaussian<<<data.grid, data.block>>>(data.d_psi, p.gridWidth, p.gridHeight,
                                          dx, dy, p.x0, p.y0, p.sigma, p.kx,
                                          p.ky, p.amp);
  cudaDeviceSynchronize();

  normalizePsi(data, p.gridWidth, p.gridHeight, dx, dy);
  cudaDeviceSynchronize();

  initHarmonicTrap<<<data.grid, data.block>>>(data.d_V, p.gridWidth,
                                              p.gridHeight, dx, dy, p.trapStr);
  cudaDeviceSynchronize();

  addGaussianObstacle<<<data.grid, data.block>>>(
      data.d_V, p.gridWidth, p.gridHeight, dx, dy, p.obstacleX, p.obstacleY,
      p.obstacleSigma, p.obstacleHeight);

  cudaDeviceSynchronize();

  initKineticOperator<<<data.grid, data.block>>>(
      data.d_expK, p.gridWidth, p.gridHeight, dk_x, dk_y, data.dt);

  cudaDeviceSynchronize();

  return data;
}

void MemoryResource<CUDAKernelMode::GrossPitaevskii>::free(
    SimulationData<CUDAKernelMode::GrossPitaevskii> &data) {
  if (data.d_psi)
    cudaFree(data.d_psi);
  if (data.d_V)
    cudaFree(data.d_V);
  if (data.d_expK)
    cudaFree(data.d_expK);
  if (data.plan)
    cufftDestroy(data.plan);

  data.d_psi = nullptr;
  data.d_V = nullptr;
  data.d_expK = nullptr;
}

void MemoryResource<CUDAKernelMode::GrossPitaevskii>::append_frame(
    const SimulationData<CUDAKernelMode::GrossPitaevskii> &data,
    std::vector<cuFloatComplex> &history) {

  size_t frame_elements = data.width * data.height;
  size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);
  size_t old_size = history.size();
  history.resize(old_size + frame_elements);
  cuFloatComplex *host_destination = history.data() + old_size;
  cudaMemcpy(host_destination, data.d_psi, frame_bytes, cudaMemcpyDeviceToHost);
}

void KernelLauncher<CUDAKernelMode::GrossPitaevskii>::launch(
    dim3 numBlocks, dim3 threadsPerBlock,
    SimulationData<CUDAKernelMode::GrossPitaevskii> &data, int t) {
  data.downloadIterator--;
  if (data.downloadIterator == 0) {
    MemoryResource<CUDAKernelMode::GrossPitaevskii>::append_frame(data,
                                                                  data.h_data);
    data.downloadIterator = data.downloadFrequency;
  }

  float fft_scale = 1.0f / (float)(data.width * data.height);
  evolveRealSpace<<<data.grid, data.block>>>(
      data.d_psi, data.d_V, data.width, data.height, data.g, data.dt / 2.0f);
  cufftExecC2C(data.plan, data.d_psi, data.d_psi, CUFFT_FORWARD);
  evolveMomentumSpace<<<data.grid, data.block>>>(
      data.d_psi, data.d_expK, data.width, data.height, fft_scale);
  cufftExecC2C(data.plan, data.d_psi, data.d_psi, CUFFT_INVERSE);
  evolveRealSpace<<<data.grid, data.block>>>(
      data.d_psi, data.d_V, data.width, data.height, data.g, data.dt / 2.0f);
}
