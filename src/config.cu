#include "config.h"
#include <iostream>

template <typename T>
void get_and_validate_param(T &config_field, const json &j,
                            const std::string &key,
                            std::function<bool(const T &)> validator,
                            const std::string &validation_error_message) {
  try {
    config_field = j.at(key).get<T>();
  } catch (const nlohmann::json::type_error &e) {
    throw std::runtime_error(
        "Params Error: Field '" + key +
        "' is missing or has wrong type. Details: " + e.what());
  }

  if (!validator(config_field)) {
    throw std::runtime_error("Params Error: '" + key + "' " +
                             validation_error_message);
  }
}

inline constexpr auto is_positive = [](auto const &x) { return x > 0; };
inline constexpr auto always_true = [](auto const &) { return true; };

Params preprocessParams(const json &j) {
  std::cout << "[Preprocess] Validating and calculating launch parameters..."
            << std::endl;

  Params config;

  const char *const positive_number_message = "must be a positive number.";

  const auto is_not_empty = [](const std::string &val) { return !val.empty(); };
  const char *const not_empty_message = "cannot be empty.";

  const auto modeValidator = [](const CUDAKernelMode &m) { return true; };

  get_and_validate_param<int>(config.iterations, j, "iterations", is_positive,
                              positive_number_message);

  get_and_validate_param<int>(config.gridWidth, j, "gridWidth", is_positive,
                              positive_number_message);

  get_and_validate_param<int>(config.gridHeight, j, "gridHeight", is_positive,
                              positive_number_message);

  get_and_validate_param<int>(config.downloadFrequency, j, "downloadFrequency",
                              is_positive, positive_number_message);

  get_and_validate_param<int>(config.threadsPerBlockX, j, "threadsPerBlockX",
                              is_positive, positive_number_message);

  get_and_validate_param<int>(config.threadsPerBlockY, j, "threadsPerBlockY",
                              is_positive, positive_number_message);

  get_and_validate_param<std::string>(config.outputFile, j, "outputFile",
                                      is_not_empty, not_empty_message);

  get_and_validate_param<CUDAKernelMode>(config.kernelMode, j, "kernelMode",
                                         modeValidator, not_empty_message);

  get_and_validate_param<float>(config.L, j, "L", is_positive,
                                positive_number_message);

  get_and_validate_param<float>(config.sigma, j, "sigma", is_positive,
                                positive_number_message);

  get_and_validate_param<float>(config.x0, j, "x0", always_true, "");

  get_and_validate_param<float>(config.y0, j, "y0", always_true, "");

  get_and_validate_param<float>(config.kx, j, "kx", always_true, "");

  get_and_validate_param<float>(config.ky, j, "ky", always_true, "");

  get_and_validate_param<float>(config.amp, j, "amp", always_true, "");

  get_and_validate_param<float>(config.trapStr, j, "trapStr", is_positive,
                                positive_number_message);

  // --- Obstacle parameters ---
  get_and_validate_param<float>(config.obstacleX, j, "obstacleX", always_true,
                                "");

  get_and_validate_param<float>(config.obstacleY, j, "obstacleY", always_true,
                                "");

  get_and_validate_param<float>(config.obstacleSigma, j, "obstacleSigma",
                                is_positive, positive_number_message);

  get_and_validate_param<float>(config.obstacleHeight, j, "obstacleHeight",
                                is_positive, positive_number_message);

  get_and_validate_param<float>(config.g, j, "g", is_positive,
                                positive_number_message);

  get_and_validate_param<float>(config.dt, j, "dt", is_positive,
                                positive_number_message);

  std::cout << "[Preprocess] Simulation configured to run for "
            << config.iterations << " iterations on a " << config.gridWidth
            << " x " << config.gridHeight << " grid." << std::endl;

  return config;
}
