#ifndef IO_H
#define IO_H

#include "json.hpp"
#include <cuComplex.h>
#include <fstream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

struct SaveOptions {
  std::string filename;
  const std::vector<cuFloatComplex> &data;
  int width;
  int height;
  int iterations;
  int downloadFrequency;
  json header;
};

inline void saveToBinaryJSON(const SaveOptions &opts) {
  const auto &[filename, data, width, height, iterations, downloadFrequency,
               header] = opts;

  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if (!out)
    throw std::runtime_error("Could not open file");

  std::string headerStr = header.dump();
  out.write(headerStr.c_str(), headerStr.size());
  out.write("\n", 1);

  out.write(reinterpret_cast<const char *>(data.data()),
            data.size() * sizeof(cuFloatComplex));
  out.close();
}

#endif
