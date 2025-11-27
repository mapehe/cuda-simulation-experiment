#ifndef IO_H
#define IO_H

#include "json.hpp"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

inline void saveToBinary(const std::string &filename,
                         const std::vector<cuFloatComplex> &data, int width,
                         int height, int iterations) {

  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if (!out) {
    throw std::runtime_error("Could not open file for writing");
  }

  std::cout << "[Helper] Writing binary output..." << std::endl;

  out.write(reinterpret_cast<const char *>(&width), sizeof(int));
  out.write(reinterpret_cast<const char *>(&height), sizeof(int));
  out.write(reinterpret_cast<const char *>(&iterations), sizeof(int));

  size_t dataSize = data.size() * sizeof(cuFloatComplex);
  out.write(reinterpret_cast<const char *>(data.data()), dataSize);

  out.close();
  std::cout << "[Helper] Saved " << (dataSize / 1024.0 / 1024.0) << " MB to "
            << filename << std::endl;
}

#endif
