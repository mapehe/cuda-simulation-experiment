#include "config.h"
#include "io.h"
#include "kernel/gpeKernel.h"
#include "kernel/testKernel.h"
#include "simulation.h"
#include <fstream>
#include <iostream>

template <CUDAKernelMode Mode>
void runSimulationTemplate(const Params &params) {
  const dim3 threadsPerBlock(params.threadsPerBlockX, params.threadsPerBlockY);
  const dim3 numBlocks(
      (params.gridWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (params.gridHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
  const int progressStep =
      std::max(1, (int)std::round(params.iterations * 0.05));
  int nextReportIteration = progressStep;

  auto data = MemoryResource<Mode>::allocate(params);

  for (int t = 0; t < params.iterations; ++t) {
    if (t >= nextReportIteration) {
      double currentProgress = (double)t / params.iterations * 100.0;
      std::cout << "[CPU] Simulation Progress: "
                << std::round(currentProgress / 5.0) * 5.0
                << "% complete (Iteration " << t << ")\n";
      nextReportIteration += progressStep;
    }

    KernelLauncher<Mode>::launch(numBlocks, threadsPerBlock, data, t);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::stringstream ss;
      ss << "CUDA Error: " << cudaGetErrorString(err);
      MemoryResource<Mode>::free(data);
      throw std::runtime_error(ss.str());
    }

    cudaDeviceSynchronize();
  }

  std::cout << "[CPU] Simulation complete." << std::endl;

  MemoryResource<Mode>::free(data);
  saveToBinary(params.outputFile, data.h_data, params.gridWidth,
               params.gridHeight, params.iterations / params.downloadFrequency);
}

void run(json config) {
  std::cout << "[CPU] Preparing simulation..." << std::endl;
  const Params params = preprocessParams(config);
  std::cout << "[CPU] Launching CUDA Kernel..." << std::endl;

  switch (params.kernelMode) {
  case CUDAKernelMode::Test:
    runSimulationTemplate<CUDAKernelMode::Test>(params);
    break;

  case CUDAKernelMode::GrossPitaevskii:
    runSimulationTemplate<CUDAKernelMode::GrossPitaevskii>(params);
    break;

  default:
    throw std::runtime_error("Error: Invalid or unsupported CUDAKernelMode.");
  }
}
