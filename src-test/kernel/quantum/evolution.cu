#include "kernel/quantum/quantumKernels.cuh"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <cstdio>

#define ASSERT_CUDA_SUCCESS(err)                                               \
  ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err)

class EvolutionTest : public ::testing::Test {
protected:
  cuFloatComplex *d_expK = nullptr;
  int width = 128;
  int height = 128;
  int numElements;
  KineticInitArgs args;

  void SetUp() override {
    numElements = width * height;

    float dk_x = 2.0f * M_PI / ((float) width);
    float dk_y = 2.0f * M_PI / ((float) height);

    args = {.width = width,
            .height = height,
            .dk_x = dk_x,
            .dk_y = dk_y,
            .dt = 0.1f};

    ASSERT_CUDA_SUCCESS(
        cudaMalloc(&d_expK, numElements * sizeof(cuFloatComplex)));
  }

  void TearDown() override {
    if (d_expK) {
      cudaFree(d_expK);
      d_expK = nullptr;
    }
    cudaDeviceReset();
  }

  void uploadData(const std::vector<cuFloatComplex> &h_data) {
    ASSERT_EQ(h_data.size(), numElements) << "Host data size mismatch";
    ASSERT_CUDA_SUCCESS(cudaMemcpy(d_expK, h_data.data(),
                                   numElements * sizeof(cuFloatComplex),
                                   cudaMemcpyHostToDevice));
  }

  void downloadData(std::vector<cuFloatComplex> &h_data) {
    h_data.resize(numElements);
    ASSERT_CUDA_SUCCESS(cudaMemcpy(h_data.data(), d_expK,
                                   numElements * sizeof(cuFloatComplex),
                                   cudaMemcpyDeviceToHost));
  }
};

TEST_F(EvolutionTest, InitKineticOperatorTest) {
  std::vector<cuFloatComplex> h_expK(numElements);
  float tol = 1e-5f;
  std::vector<std::complex<float>> expected(width * height);

  for (size_t i = 0; i < width; ++i) {
    for (size_t j = 0; j < height; ++j) {
      float kx_val = (i <= width / 2.0f) ? (float)i : ((float)i - (float)width);
      float ky_val = (j <= height / 2.0f) ? (float)j : ((float)j - (float)height);

      float kx = kx_val * args.dk_x;
      float ky = ky_val * args.dk_y;

      float k2 = kx * kx + ky * ky;
      float angle = -0.5f * k2 * args.dt;

      expected[j * width + i] = std::exp(std::complex<float>(0.0f, angle));
    }
  }

  dim3 grid(4, 4);
  dim3 block(32, 32);
  initKineticOperator<<<grid, block>>>(d_expK, args);
  cudaDeviceSynchronize();

  ASSERT_CUDA_SUCCESS(cudaGetLastError());

  downloadData(h_expK);

  for (size_t i = 0; i < width; ++i) {
    for (size_t j = 0; j < height; ++j) {
      const int flatIndex = width * j + i;
      const auto expKVal = h_expK[flatIndex];
      const auto expectedVal = expected[flatIndex];
      EXPECT_NEAR(expKVal.x, expectedVal.real(), tol);
      EXPECT_NEAR(expKVal.y, expectedVal.imag(), tol);
    }
  }
}
