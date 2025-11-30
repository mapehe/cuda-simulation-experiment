#ifndef CONFIG_H
#define CONFIG_H

#include "json.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

enum class SimulationMode { Test, GrossPitaevskii };

inline void from_json(const nlohmann::json &j, SimulationMode &mode) {
  static const std::unordered_map<std::string, SimulationMode> str_to_enum{
      {"test", SimulationMode::Test},
      {"grossPitaevskii", SimulationMode::GrossPitaevskii},
  };

  const std::string s = j.get<std::string>();
  auto it = str_to_enum.find(s);

  if (it != str_to_enum.end()) {
    mode = it->second;
  } else {
    throw nlohmann::json::type_error::create(
        302, "Unknown SimulationMode: " + s, &j);
  }
}

struct GrossPitaevskiiParams {
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
};

struct Params {
  int iterations;
  int gridWidth;
  int gridHeight;
  int threadsPerBlockX;
  int threadsPerBlockY;
  int downloadFrequency;
  std::string output;
  SimulationMode simulationMode;

  GrossPitaevskiiParams grossPitaevskii;
};

Params preprocessParams(const json &j);

#endif
