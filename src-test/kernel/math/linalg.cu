#include "kernel/math/linalg.cuh"
#include <gtest/gtest.h>

#define ASSERT_CUDA_SUCCESS(err)                                               \
  ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err)

__global__ void testDistanceSqKernel(float2 p1, float2 p2, float *output) {
  *output = distanceSq(p1, p2);
}

class LinalgTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(LinalgTest, distanceSqLogic) {

  float2 h_p1 = make_float2(0.5f, 7.4f);
  float2 h_p2 = make_float2(-21.2f, 3.2f);

  float *d_result;
  ASSERT_CUDA_SUCCESS(cudaMalloc(&d_result, sizeof(float)));

  testDistanceSqKernel<<<1, 1>>>(h_p1, h_p2, d_result);

  ASSERT_CUDA_SUCCESS(cudaGetLastError());
  ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

  float h_result = 0.0f;
  ASSERT_CUDA_SUCCESS(
      cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_result);
  EXPECT_FLOAT_EQ(h_result, 488.53f);
}
