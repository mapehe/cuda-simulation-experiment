#include "config.h"
#include "engine/grossPitaevskii.cuh"
#include "engine/testSimulation.cuh"
#include "simulation.h"
#include <fstream>
#include <iostream>

template <typename T>
std::unique_ptr<ComputeEngine<T>> getComputeEngine(const Params &params) {
  switch (params.simulationMode) {
  case SimulationMode::Test:
    return std::make_unique<TestEngine>(params);

  case SimulationMode::GrossPitaevskii:
    return std::make_unique<GrossPitaevskiiEngine>(params);

  default:
    throw std::runtime_error("Error: Invalid or unsupported SimulationMode.");
  }
}

void run(json config) {
  std::cout << "[CPU] Preparing simulation..." << std::endl;
  const Params params = preprocessParams(config);
  auto sim = getComputeEngine<cuFloatComplex>(params);

  sim->run();

  std::cout << "[CPU] Simulation complete." << std::endl;
  sim->saveResults(params.output);
}
