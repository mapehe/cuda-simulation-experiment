#include "kernel/quantum/quantumKernels.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#define ASSERT_CUDA_SUCCESS(err)                                               \
  ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err)

class WavefunctionTest : public ::testing::Test {
protected:
  cuFloatComplex *d_psi = nullptr;
  int width = 128;
  int height = 128;
  int numElements;
  GaussianArgs args;
  Grid gridArgs;
  Grid smallGridArgs;
  float dx;
  float dy;

  void SetUp() override {
    numElements = width * height;

    args = {.x0 = 0.25f, .y0 = -0.69f, .sigma = 10.123f, .amplitude = 25.23f};

    gridArgs = {
        .width = width,
        .height = height,
        .L_x = 1,
        .L_y = 1,
    };

    smallGridArgs = {
        .width = 16,
        .height = 16,
        .L_x = 32,
        .L_y = 32,
    };

    dx = 1.0f / width;
    dy = 1.0f / width;

    ASSERT_CUDA_SUCCESS(
        cudaMalloc(&d_psi, numElements * sizeof(cuFloatComplex)));
  }

  void TearDown() override {
    if (d_psi) {
      cudaFree(d_psi);
      d_psi = nullptr;
    }
    cudaDeviceReset();
  }

  void uploadData(const std::vector<cuFloatComplex> &h_data) {
    ASSERT_EQ(h_data.size(), numElements) << "Host data size mismatch";
    ASSERT_CUDA_SUCCESS(cudaMemcpy(d_psi, h_data.data(),
                                   numElements * sizeof(cuFloatComplex),
                                   cudaMemcpyHostToDevice));
  }

  void downloadData(std::vector<cuFloatComplex> &h_data) {
    h_data.resize(numElements);
    ASSERT_CUDA_SUCCESS(cudaMemcpy(h_data.data(), d_psi,
                                   numElements * sizeof(cuFloatComplex),
                                   cudaMemcpyDeviceToHost));
  }
};

TEST_F(WavefunctionTest, SquareMagnitudeLogic) {
  SquareMagnitude op;
  cuFloatComplex x = make_cuFloatComplex(3.0f, 4.0f);
  EXPECT_FLOAT_EQ(op(x), 25.0f);
  cuFloatComplex y = make_cuFloatComplex(-2.0f, 4.0f);
  EXPECT_FLOAT_EQ(op(y), 20.0f);
}

TEST_F(WavefunctionTest, NormalizationEnsuresProbabilityIsOne) {
  std::vector<cuFloatComplex> h_psi(numElements);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

  for (auto &val : h_psi) {
    val = make_cuFloatComplex(dis(gen), dis(gen));
  }

  uploadData(h_psi);

  dim3 block(128);
  dim3 grid((numElements + block.x - 1) / block.x);

  normalizePsi(d_psi, block, grid, args, gridArgs);
  ASSERT_CUDA_SUCCESS(cudaGetLastError());

  downloadData(h_psi);

  float totalProbability = 0.0f;
  for (const auto &val : h_psi) {
    float magSq = (val.x * val.x) + (val.y * val.y);
    totalProbability += magSq;
  }

  totalProbability *= (dx * dy);

  EXPECT_NEAR(totalProbability, 1.0f, 1e-4);
}

TEST_F(WavefunctionTest, HandlesZeroWavefunctionGracefully) {
  std::vector<cuFloatComplex> h_psi(numElements);

  for (auto &val : h_psi) {
    val = make_cuFloatComplex(0, 0);
  }

  uploadData(h_psi);

  dim3 block(128);
  dim3 grid((numElements + block.x - 1) / block.x);

  normalizePsi(d_psi, block, grid, args, gridArgs);
  ASSERT_CUDA_SUCCESS(cudaGetLastError());

  downloadData(h_psi);

  float totalProbability = 0.0f;
  for (const auto &val : h_psi) {
    float magSq = (val.x * val.x) + (val.y * val.y);
    totalProbability += magSq;
  }

  totalProbability *= (dx * dy);

  EXPECT_NEAR(totalProbability, 0, 1e-4);
}

TEST_F(WavefunctionTest, InitGaussianTest) {
  std::vector<cuFloatComplex> h_psi(numElements);
  float tol = 1e-5f;
  float expected[16][16] = {0.0f};

  for (size_t i = 0; i < smallGridArgs.width; ++i) {
    for (size_t j = 0; j < smallGridArgs.height; ++j) {
      float x =
          smallGridArgs.L_x * ((2.0f * i) / (smallGridArgs.width - 1) - 1);
      float y =
          smallGridArgs.L_y * (1 - (2.0f * j) / (smallGridArgs.height - 1));

      float d = std::pow(x - args.x0, 2) + std::pow(y - args.y0, 2);

      expected[i][j] =
          args.amplitude * std::exp(-d / (2.0f * args.sigma * args.sigma));
    }
  }

  dim3 grid(1, 1);
  dim3 block(16, 16);
  initGaussian<<<grid, block>>>(d_psi, args, smallGridArgs);
  cudaDeviceSynchronize();

  ASSERT_CUDA_SUCCESS(cudaGetLastError());

  downloadData(h_psi);

  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      const int flatIndex = smallGridArgs.width * j + i;
      const auto psiVal = h_psi[flatIndex];
      const auto expectedVal = expected[i][j];
      EXPECT_NEAR(psiVal.x, expectedVal, tol);
      EXPECT_NEAR(psiVal.y, 0, tol);
    }
  }
}
