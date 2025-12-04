#include "engine/testSimulation.cuh"
#include "io.h"

TestEngine::TestEngine(const Params &p) : ComputeEngine(p), d_grid(nullptr) {
  size_t size = width * height * sizeof(cuFloatComplex);
  cudaError_t err = cudaMalloc(&d_grid, size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate TestEngine device memory");
  }
  cudaMemset(d_grid, 0, size);
}

TestEngine::~TestEngine() {
  if (d_grid) {
    cudaFree(d_grid);
    d_grid = nullptr;
  }
}

void TestEngine::appendFrame(std::vector<cuFloatComplex> &history) {
  size_t frame_elements = width * height;
  size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);
  size_t old_size = history.size();

  history.resize(old_size + frame_elements);
  cuFloatComplex *host_destination = history.data() + old_size;
  cudaMemcpy(host_destination, d_grid, frame_bytes, cudaMemcpyDeviceToHost);
}

void TestEngine::solveStep(int t) {
  testKernel<<<grid, block>>>(d_grid, width, height, t);
  cudaDeviceSynchronize();
}

void TestEngine::saveResults(const std::string &filename) {
  saveToBinaryJSON({.filename = filename,
                    .data = historyData,
                    .width = width,
                    .height = height,
                    .iterations = iterations,
                    .downloadFrequency = downloadFrequency});
}
