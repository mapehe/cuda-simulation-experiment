#include "json.hpp"
#include "simulation.h"
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

const std::string CONFIG_FILENAME = "config.json";
const std::string OVERRIDES_FILENAME = "configOverrides.json";

json readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + filename + "!");
    }

    json object;
    try {
        file >> object;
    } catch (const json::parse_error &e) {
        throw std::runtime_error(std::string("JSON Parse Error in ") + filename + ": " + e.what());
    }

    return object;
}

json readConfig() {
  json config = readFile(CONFIG_FILENAME);

  try {
      json overrides = readFile(OVERRIDES_FILENAME);
      config.merge_patch(overrides); 
      std::cout << "[CPU] Successfully merged overrides from " << OVERRIDES_FILENAME << std::endl;
  } catch (const std::runtime_error& e) {
      std::cerr << "[CPU] Warning: Could not load overrides from " << OVERRIDES_FILENAME 
                << ". Using default configuration. (" << e.what() << ")" << std::endl;
  }

  return config;
}

int main() {
  std::cout << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << "||                                        ||" << std::endl;
  std::cout << "||      S I M U L A T O R   v 0.0.1       ||" << std::endl;
  std::cout << "||                                        ||" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << std::endl << "STARTING..." << std::endl << std::endl;

  run(readConfig());
  return 0;
}
