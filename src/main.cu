#include "config.h"
#include "json.hpp"
#include "simulation.h"
#include "tclap/CmdLine.h"
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

const std::string CONFIG_FILENAME = "config.json";
const std::string OVERRIDES_FILENAME = "configOverrides.json";

json readFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open " + filename + "!");
  }

  json object;
  try {
    file >> object;
  } catch (const json::parse_error &e) {
    throw std::runtime_error(std::string("JSON Parse Error in ") + filename +
                             ": " + e.what());
  }

  return object;
}

json readConfig(json cmdArgs) {
  json config = readFile(CONFIG_FILENAME);
  config.merge_patch(cmdArgs);

  try {
    json overrides = readFile(OVERRIDES_FILENAME);
    config.merge_patch(overrides);
    std::cout << "[CPU] Successfully merged overrides from "
              << OVERRIDES_FILENAME << std::endl;
  } catch (const std::runtime_error &e) {
    std::cerr << "[CPU] Warning: Could not load overrides from "
              << OVERRIDES_FILENAME << ". Using default configuration. ("
              << e.what() << ")" << std::endl;
  }

  return config;
}

json parseArguments(int argc, char **argv) {
  try {
    TCLAP::CmdLine cmd("Simulator CLI", ' ', "0.0.1");

    TCLAP::ValueArg<std::string> outputArg("o", "output", "Path to output file",
                                           true, "", "string");
    cmd.add(outputArg);

    std::vector<std::string> allowedModes;
    for (const auto &pair : SimulationModeMap::get()) {
      allowedModes.push_back(pair.first);
    }
    TCLAP::ValuesConstraint<std::string> modeConstraint(allowedModes);
    TCLAP::ValueArg<std::string> modeArg("m", "mode", "Simulation Mode", true,
                                         "test", &modeConstraint);
    cmd.add(modeArg);

    cmd.parse(argc, argv);
    return json{{"output", outputArg.getValue()},
                {"simulationMode", modeArg.getValue()}};

  } catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId()
              << std::endl;
    return json();
  }
}

int main(int argc, char **argv) {
  std::cout << std::endl << "Starting FluxLab..." << std::endl;

  json cmdArgs = parseArguments(argc, argv);
  run(readConfig(cmdArgs));
  return 0;
}
