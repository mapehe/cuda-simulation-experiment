#include "config.h"
#include <string>
#include <string_view>
#include <ranges>
#include <iostream>

inline const nlohmann::json& get_nested(const nlohmann::json& j,
                                        const std::string& key)
{
    const nlohmann::json* ptr = &j;

    for (auto part_view : std::string_view(key) | std::views::split('.')) {
        std::string part;
        for (char c : part_view) {
            part.push_back(c);
        }

        ptr = &ptr->at(part);
    }

    return *ptr;
}



template <typename T>
void get_and_validate_param(T &config_field, const json &j,
                            const std::string &key,
                            std::function<bool(const T &)> validator,
                            const std::string &validation_error_message) {
  try {
    const auto& val = get_nested(j, key);
    config_field = val.get<T>();
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
inline constexpr auto is_non_negtive = [](auto const &x) { return !(x < 0); };
inline constexpr auto always_true = [](auto const &) { return true; };

Params preprocessParams(const json &j) {
  std::cout << "[Preprocess] Validating and calculating launch parameters..."
            << std::endl;

  Params config;

  const char *const positive_number_message = "must be a positive number.";
  const char *const non_negtive_number_message = "must not be negative.";

  const auto is_not_empty = [](const std::string &val) { return !val.empty(); };
  const char *const not_empty_message = "cannot be empty.";

  const auto modeValidator = [](const SimulationMode &m) { return true; };

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

  get_and_validate_param<SimulationMode>(config.simulationMode, j,
                                         "simulationMode", modeValidator,
                                         not_empty_message);

  get_and_validate_param<float>(config.grossPitaevskii.L, j, "grossPitaevskii.L", is_positive,
                                positive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.sigma, j, "grossPitaevskii.sigma",
                                is_positive, positive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.x0, j, "grossPitaevskii.x0", always_true,
                                "");

  get_and_validate_param<float>(config.grossPitaevskii.y0, j, "grossPitaevskii.y0", always_true,
                                "");

  get_and_validate_param<float>(config.grossPitaevskii.kx, j, "grossPitaevskii.kx", always_true,
                                "");

  get_and_validate_param<float>(config.grossPitaevskii.ky, j, "grossPitaevskii.ky", always_true,
                                "");

  get_and_validate_param<float>(config.grossPitaevskii.amp, j, "grossPitaevskii.amp",
                                always_true, "");

  get_and_validate_param<float>(config.grossPitaevskii.trapStr, j, "grossPitaevskii.trapStr",
                                is_non_negtive, non_negtive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.g, j, "grossPitaevskii.g", always_true,
                                "");

  get_and_validate_param<float>(config.grossPitaevskii.dt, j, "grossPitaevskii.dt", is_positive,
                                positive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.V_bias, j, "grossPitaevskii.V_bias",
                                is_non_negtive, non_negtive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.r_0, j, "grossPitaevskii.r_0",
                                is_non_negtive, non_negtive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.sigma2, j, "grossPitaevskii.sigma2",
                                is_non_negtive, non_negtive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.absorbStrength, j,
                                "grossPitaevskii.absorbStrength", is_non_negtive,
                                non_negtive_number_message);

  get_and_validate_param<float>(config.grossPitaevskii.absorbWidth, j,
                                "grossPitaevskii.absorbWidth", is_non_negtive,
                                non_negtive_number_message);

  std::cout << "[Preprocess] Simulation configured to run for "
            << config.iterations << " iterations on a " << config.gridWidth
            << " x " << config.gridHeight << " grid." << std::endl;

  return config;
}
