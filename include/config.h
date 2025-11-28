#ifndef CONFIG_H
#define CONFIG_H

#include "json.hpp"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

enum class CUDAKernelMode { Test, GrossPitaevskii };

inline void from_json(const nlohmann::json &j, CUDAKernelMode &mode) {
  static const std::unordered_map<std::string, CUDAKernelMode> str_to_enum{
      {"test", CUDAKernelMode::Test},
      {"grossPitaevskii", CUDAKernelMode::GrossPitaevskii},
  };

  const std::string s = j.get<std::string>();
  auto it = str_to_enum.find(s);

  if (it != str_to_enum.end()) {
    mode = it->second;
  } else {
    throw nlohmann::json::type_error::create(
        302, "Unknown CUDAKernelMode: " + s, &j);
  }
}

struct Params {
  int iterations;
  int gridWidth;
  int gridHeight;
  int threadsPerBlockX;
  int threadsPerBlockY;
  int downloadFrequency;
  std::string outputFile;
  CUDAKernelMode kernelMode;

  // --- Physical parameters ---
  float L;
  float sigma;
  float x0;
  float y0;
  float kx;
  float ky;
  float amp;
  float omega;
  float trapStr;

  // --- Obstacle parameters ---
  float obstacleX;
  float obstacleY;
  float obstacleSigma;
  float obstacleHeight;

  float dt;
  float g;

  float V_bias;
  float r_0;
  float sigma2;
  float absorbStrength;
  float absorbWidth;
};

Params preprocessParams(const json &j);

template <CUDAKernelMode M> struct KernelLauncher;
template <CUDAKernelMode M> struct SimulationData;
template <CUDAKernelMode M> struct MemoryResource;

#endif
