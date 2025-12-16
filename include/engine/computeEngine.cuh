#ifndef SIMULATION_MODE_H
#define SIMULATION_MODE_H

#include "config.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

template <typename T> class ComputeEngine {
private:
  void step(int t) {
    if (downloadIterator == 0) {
      appendFrame(historyData);
    }
    downloadIterator = (downloadIterator + 1) % getDownloadFrequency();
    solveStep(t);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::stringstream ss;
      ss << "CUDA Error: " << cudaGetErrorString(err);
      throw std::runtime_error(ss.str());
    }
  }

protected:
  const Params params;
  int downloadIterator;

  std::vector<T> historyData;

  virtual void solveStep(int t) = 0;
  virtual int getDownloadFrequency() = 0;
  virtual int getTotalSteps() = 0;
  virtual void appendFrame(std::vector<cuFloatComplex> &history) = 0;

public:
  ComputeEngine(const Params &p) : params(p), downloadIterator(0) {};
  virtual ~ComputeEngine() = default;
  virtual void saveResults(const std::string &filename) = 0;
  void run() {
    for (int t = 0; t < getTotalSteps(); t++) {
      step(t);
    }
  }
};

#endif
