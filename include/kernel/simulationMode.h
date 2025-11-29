#ifndef SIMULATION_MODE_H
#define SIMULATION_MODE_H

#include "config.h"

class SimulationMode {
protected:
  int width;
  int height;
  int iterations;
  int downloadFrequency;
  int downloadIterator;
  dim3 grid;
  dim3 block;

public:
  explicit SimulationMode(const Params &p)
      : width(p.gridWidth), height(p.gridHeight), iterations(p.iterations),
        downloadFrequency(p.downloadFrequency), downloadIterator(1) {
    grid = dim3(p.threadsPerBlockX, p.threadsPerBlockY);
    block = dim3((p.gridWidth + grid.x - 1) / grid.x,
                 (p.gridHeight + grid.y - 1) / grid.y);
  }

  virtual ~SimulationMode() = default;

  virtual void launch(int t) = 0;
  virtual void appendFrame(std::vector<cuFloatComplex> &history) = 0;
  virtual void saveResults(const std::string &filename) = 0;
};

#endif
