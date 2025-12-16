#ifndef CONFIG_H
#define CONFIG_H

#include "json.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

enum class SimulationMode { Test, GrossPitaevskii };

struct SimulationModeMap {
  static const std::unordered_map<std::string, SimulationMode> &get() {
    static const std::unordered_map<std::string, SimulationMode> map{
        {"test", SimulationMode::Test},
        {"grossPitaevskii", SimulationMode::GrossPitaevskii},
    };
    return map;
  }
};

inline void from_json(const nlohmann::json &j, SimulationMode &mode) {
  const std::string s = j.get<std::string>();
  const auto &map = SimulationModeMap::get();

  auto it = map.find(s);
  if (it != map.end()) {
    mode = it->second;
  } else {
    throw nlohmann::json::type_error::create(
        302, "Unknown SimulationMode: " + s, &j);
  }
}

struct CommonParams {
  int iterations;
  int gridWidth;
  int gridHeight;
  int threadsPerBlockX;
  int threadsPerBlockY;
  int downloadFrequency;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(CommonParams, iterations, gridWidth,
                                 gridHeight, threadsPerBlockY, threadsPerBlockY,
                                 downloadFrequency)
};

struct GrossPitaevskiiParams {
  int iterations;
  int gridWidth;
  int gridHeight;
  int threadsPerBlockX;
  int threadsPerBlockY;
  int downloadFrequency;
  float L;
  float sigma;
  float x0;
  float y0;
  float kx;
  float ky;
  float amp;
  float omega;
  float trapStr;

  float dt;
  float g;

  float V_bias;
  float r_0;
  float sigma2;
  float absorbStrength;
  float absorbWidth;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(GrossPitaevskiiParams, iterations, gridWidth,
                                 gridHeight, threadsPerBlockY, threadsPerBlockY,
                                 downloadFrequency, L, sigma, x0, y0, kx, ky,
                                 amp, omega, trapStr, dt, g, V_bias, r_0,
                                 sigma2, absorbStrength, absorbWidth)
};

struct Params {
  std::string output;
  SimulationMode simulationMode;

  CommonParams test;
  GrossPitaevskiiParams grossPitaevskii;
};

Params preprocessParams(const json &j);

#endif
