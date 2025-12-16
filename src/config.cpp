#include "config.h"
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>

template <typename T, typename Validator>
T parse(const json &j, const std::string &key, Validator validator,
        const char *errMsg) {
  if (!j.contains(key)) {
    throw std::runtime_error("Missing key: " + key);
  }

  T val = j.at(key).get<T>();

  if (!validator(val)) {
    throw std::runtime_error("Key '" + key + "' " + errMsg);
  }

  return val;
}

Params preprocessParams(const json &j) {
  std::cout << "[Preprocess] Validating and calculating launch parameters...\n";

  const char *pos_err = "must be positive";
  const char *non_neg_err = "must be non-negative";

  auto is_pos = [](auto v) { return v > 0; };
  auto is_non_neg = [](auto v) { return v >= 0; };
  auto any = [](auto) { return true; };
  auto mode_ok = [](const SimulationMode &) { return true; };

  const auto &testJson = j.at("test");
  const auto &gpJson = j.at("grossPitaevskii");

  return Params{
      .output = parse<std::string>(j, "output", any, ""),
      .simulationMode = parse<SimulationMode>(j, "simulationMode", mode_ok, ""),

      .test =
          {
              .iterations = parse<int>(testJson, "iterations", is_pos, pos_err),
              .gridWidth = parse<int>(testJson, "gridWidth", is_pos, pos_err),
              .gridHeight = parse<int>(testJson, "gridHeight", is_pos, pos_err),
              .threadsPerBlockX =
                  parse<int>(testJson, "threadsPerBlockX", is_pos, pos_err),
              .threadsPerBlockY =
                  parse<int>(testJson, "threadsPerBlockY", is_pos, pos_err),
              .downloadFrequency =
                  parse<int>(testJson, "downloadFrequency", is_pos, pos_err),
          },

      .grossPitaevskii = {
          .iterations = parse<int>(gpJson, "iterations", is_pos, pos_err),
          .gridWidth = parse<int>(gpJson, "gridWidth", is_pos, pos_err),
          .gridHeight = parse<int>(gpJson, "gridHeight", is_pos, pos_err),
          .threadsPerBlockX =
              parse<int>(gpJson, "threadsPerBlockX", is_pos, pos_err),
          .threadsPerBlockY =
              parse<int>(gpJson, "threadsPerBlockY", is_pos, pos_err),
          .downloadFrequency =
              parse<int>(gpJson, "downloadFrequency", is_pos, pos_err),

          .L = parse<float>(gpJson, "L", is_pos, pos_err),
          .sigma = parse<float>(gpJson, "sigma", is_pos, pos_err),
          .x0 = parse<float>(gpJson, "x0", any, ""),
          .y0 = parse<float>(gpJson, "y0", any, ""),
          .kx = parse<float>(gpJson, "kx", any, ""),
          .ky = parse<float>(gpJson, "ky", any, ""),
          .amp = parse<float>(gpJson, "amp", any, ""),
          .omega = parse<float>(gpJson, "omega", is_non_neg, non_neg_err),
          .trapStr = parse<float>(gpJson, "trapStr", is_non_neg, non_neg_err),

          .dt = parse<float>(gpJson, "dt", is_pos, pos_err),
          .g = parse<float>(gpJson, "g", any, ""),

          .V_bias = parse<float>(gpJson, "V_bias", is_non_neg, non_neg_err),
          .r_0 = parse<float>(gpJson, "r_0", is_non_neg, non_neg_err),
          .sigma2 = parse<float>(gpJson, "sigma2", is_non_neg, non_neg_err),
          .absorbStrength =
              parse<float>(gpJson, "absorbStrength", is_non_neg, non_neg_err),
          .absorbWidth =
              parse<float>(gpJson, "absorbWidth", is_non_neg, non_neg_err),

      }};
}
