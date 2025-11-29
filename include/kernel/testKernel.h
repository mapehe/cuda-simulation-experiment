#ifndef TEST_KERNEL_CUH
#define TEST_KERNEL_CUH

#include "config.h"
#include "kernel/simulationMode.h"

class TestSimulation : public SimulationMode {
public:
  explicit TestSimulation(const Params &p);
  ~TestSimulation() override;
  void launch(int t) override;
  void appendFrame(std::vector<cuFloatComplex> &history) override;
  std::vector<cuFloatComplex> &getHistory() { return h_data; }
  void saveResults(const std::string &filename) override;

private:
  cuFloatComplex *d_grid;
  std::vector<cuFloatComplex> h_data;
};

#endif
