#include "engine/testSimulation.cuh"
#include "io.h"

TestEngine::TestEngine(const Params &p) : ComputeEngine(p), d_grid(nullptr) {
  size_t size = params.gridWidth * params.gridHeight * sizeof(cuFloatComplex);
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
  size_t frame_elements = params.gridWidth * params.gridHeight;
  size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);
  size_t old_size = history.size();

  history.resize(old_size + frame_elements);
  cuFloatComplex *host_destination = history.data() + old_size;
  cudaMemcpy(host_destination, d_grid, frame_bytes, cudaMemcpyDeviceToHost);
}

void TestEngine::solveStep(int t) {
  testKernel<<<grid, block>>>(d_grid, params.gridWidth, params.gridHeight, t);
  cudaDeviceSynchronize();
}

void TestEngine::saveResults(const std::string &filename) {
  saveToBinaryJSON({.filename = filename,
                    .data = historyData,
                    .width = params.gridWidth,
                    .height = params.gridHeight,
                    .iterations = params.iterations,
                    .downloadFrequency = params.downloadFrequency});
}
