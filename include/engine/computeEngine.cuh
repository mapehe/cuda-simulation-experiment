#ifndef SIMULATION_MODE_H
#define SIMULATION_MODE_H

#include "config.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

template <typename T> class ComputeEngine {
protected:
  const Params params;
  int downloadIterator;
  dim3 grid;
  dim3 block;

  std::vector<T> historyData;

  virtual void solveStep(int t) = 0;
  virtual void appendFrame(std::vector<cuFloatComplex> &history) = 0;

public:
  explicit ComputeEngine(const Params &p)
      : params(p), downloadIterator(1) {
    grid = dim3(p.threadsPerBlockX, p.threadsPerBlockY);
    block = dim3((p.gridWidth + grid.x - 1) / grid.x,
                 (p.gridHeight + grid.y - 1) / grid.y);
  }

  virtual ~ComputeEngine() = default;

  virtual void saveResults(const std::string &filename) = 0;

  void step(int t) {
    downloadIterator--;
    if (downloadIterator == 0) {
      appendFrame(historyData);
      downloadIterator = params.downloadFrequency;
    }

    solveStep(t);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::stringstream ss;
      ss << "CUDA Error: " << cudaGetErrorString(err);
      throw std::runtime_error(ss.str());
    }
  }
};

#endif
