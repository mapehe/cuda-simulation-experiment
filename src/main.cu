#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "json.hpp"

using json = nlohmann::json;

int readConfig() {
    std::ifstream configFile("config.json");
    
    if (!configFile.is_open()) {
        std::cerr << "Could not open config.json!" << std::endl;
        return 1;
    }

    json config;
    try {
        configFile >> config;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON Parse Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Found a configuration file with the following key-value pairs:" << std::endl << std::endl;

    for (const auto& [key, value] : config.items()) {
        std::cout << key << " : " << value << std::endl;
    }

    std::cout << std::endl;
    return 0;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "||                                        ||" << std::endl;
    std::cout << "||      S I M U L A T O R   v 0.0.1       ||" << std::endl;
    std::cout << "||                                        ||" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::endl << "STARTING..." << std::endl << std::endl;

    readConfig();
    return 0;
}
