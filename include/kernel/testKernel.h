#ifndef TEST_KERNEL_CUH
#define TEST_KERNEL_CUH

#include "config.h"

template <> struct SimulationData<CUDAKernelMode::Test> {
  cuFloatComplex *d_grid;
  std::vector<cuFloatComplex> h_data;

  int width;
  int height;
  int iterations;
  int downloadFrequency;
  int downloadIterator;
};

template <> struct MemoryResource<CUDAKernelMode::Test> {

  static SimulationData<CUDAKernelMode::Test> allocate(const Params &p) {
    SimulationData<CUDAKernelMode::Test> data;
    data.width = p.gridWidth;
    data.height = p.gridHeight;
    data.iterations = p.iterations;
    data.downloadFrequency = p.downloadFrequency;
    data.downloadIterator = 1;

    size_t size = p.gridWidth * p.gridHeight * sizeof(cuFloatComplex);

    cudaMalloc(&data.d_grid, size);
    cudaMemset(data.d_grid, 0, size);

    std::cout << "[Helper] Allocated an array (" << p.gridWidth << "x"
              << p.gridHeight << "x" << p.iterations << ") on device."
              << std::endl;

    return data;
  }

  static void free(SimulationData<CUDAKernelMode::Test> &data) {
    cudaFree(data.d_grid);
    data.d_grid = nullptr;
  }

  static void append_frame(const SimulationData<CUDAKernelMode::Test> &data,
                           std::vector<cuFloatComplex> &history) {

    size_t frame_elements = data.width * data.height;
    size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);
    size_t old_size = history.size();
    history.resize(old_size + frame_elements);
    cuFloatComplex *host_destination = history.data() + old_size;
    cudaMemcpy(host_destination, data.d_grid, frame_bytes,
               cudaMemcpyDeviceToHost);
  }
};

__global__ void testKernel(cuFloatComplex *d_array, int gridWidth,
                           int gridHeight, int time);

template <> struct KernelLauncher<CUDAKernelMode::Test> {

  static void launch(dim3 numBlocks, dim3 threadsPerBlock,
                     SimulationData<CUDAKernelMode::Test> &data, int t) {
    data.downloadIterator--;
    if (data.downloadIterator == 0) {
      MemoryResource<CUDAKernelMode::Test>::append_frame(data, data.h_data);
      data.downloadIterator = data.downloadFrequency;
    }
    testKernel<<<numBlocks, threadsPerBlock>>>(data.d_grid, data.width,
                                               data.height, t);
  }
};

#endif
