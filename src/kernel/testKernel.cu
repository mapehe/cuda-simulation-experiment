#include "io.h"
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

TestSimulation::TestSimulation(const Params &p)
    : SimulationMode(p), d_grid(nullptr) {
  size_t size = width * height * sizeof(cuFloatComplex);
  cudaError_t err = cudaMalloc(&d_grid, size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate TestSimulation device memory");
  }

  cudaMemset(d_grid, 0, size);

  std::cout << "[Helper] Allocated an array (" << width << "x" << height
            << ") on device." << std::endl;
}

TestSimulation::~TestSimulation() {
  if (d_grid) {
    cudaFree(d_grid);
    d_grid = nullptr;
  }
}

void TestSimulation::appendFrame(std::vector<cuFloatComplex> &history) {
  size_t frame_elements = width * height;
  size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);
  size_t old_size = history.size();

  history.resize(old_size + frame_elements);
  cuFloatComplex *host_destination = history.data() + old_size;
  cudaMemcpy(host_destination, d_grid, frame_bytes, cudaMemcpyDeviceToHost);
}

void TestSimulation::launch(int t) {
  downloadIterator--;
  if (downloadIterator == 0) {
    appendFrame(h_data);
    downloadIterator = downloadFrequency;
  }

  testKernel<<<grid, block>>>(d_grid, width, height, t);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA Error in TestSimulation: " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}

void TestSimulation::saveResults(const std::string &filename) {
  saveToBinary(filename, this->h_data, this->width, this->height,
               this->iterations);
}
