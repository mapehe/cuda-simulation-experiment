#include "json.hpp"
#include "simulation.h"
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

json readConfig() {
  std::ifstream configFile("config.json");

  if (!configFile.is_open()) {
    throw std::runtime_error("Could not open config.json!");
  }

  json config;
  try {
    configFile >> config;
  } catch (const json::parse_error &e) {
    throw std::runtime_error(std::string("JSON Parse Error: ") + e.what());
  }

  std::cout << std::endl;
  return config;
}

int main() {
  std::cout << "============================================" << std::endl;
  std::cout << "||                                        ||" << std::endl;
  std::cout << "||      S I M U L A T O R   v 0.0.1       ||" << std::endl;
  std::cout << "||                                        ||" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << std::endl << "STARTING..." << std::endl << std::endl;

  run(readConfig());
  return 0;
}
