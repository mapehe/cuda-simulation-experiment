#ifndef GPE_KERNEL_CUH
#define GPE_KERNEL_CUH

#include "config.h"
#include "engine/computeEngine.cuh"
#include "kernel/quantum/quantumKernels.cuh"
#include "simulationMode.h"
#include <cufft.h>

class GrossPitaevskiiEngine : public ComputeEngine<cuFloatComplex> {
public:
  explicit GrossPitaevskiiEngine(const Params &p);
  ~GrossPitaevskiiEngine() override;
  void solveStep(int t) override;
  void appendFrame(std::vector<cuFloatComplex> &history) override;
  void saveResults(const std::string &filename) override;
  int getDownloadFrequency() override;
  int getTotalSteps() override;

private:
  cuFloatComplex *d_psi;
  cuFloatComplex *d_V;
  cuFloatComplex *d_expK;
  cufftHandle plan;

  std::tuple<GaussianArgs, PotentialArgs, KineticInitArgs, Grid>
  createSimulationArgs(const Params &p, float dt) const;

  dim3 grid;
  dim3 block;
};

#endif
