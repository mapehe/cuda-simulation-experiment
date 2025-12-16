#ifndef TEST_KERNEL_CUH
#define TEST_KERNEL_CUH

#include "config.h"
#include "engine/computeEngine.cuh"
#include "kernel/testKernel.cuh"

class TestEngine : public ComputeEngine<cuFloatComplex> {
public:
  explicit TestEngine(const Params &p);
  ~TestEngine() override;
  void solveStep(int t) override;
  void appendFrame(std::vector<cuFloatComplex> &history) override;
  void saveResults(const std::string &filename) override;
  int getDownloadFrequency() override;
  int getTotalSteps() override;

private:
  cuFloatComplex *d_grid;
  dim3 grid;
  dim3 block;
};

#endif
