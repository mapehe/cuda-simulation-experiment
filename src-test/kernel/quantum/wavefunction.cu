#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "kernel/quantum/quantumKernels.cuh"

#define ASSERT_CUDA_SUCCESS(err) \
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err)

class WavefunctionTest : public ::testing::Test {
protected:
    cuFloatComplex* d_psi = nullptr;
    int width = 128;
    int height = 128;
    int numElements;
    GaussianArgs args;

    void SetUp() override {
        numElements = width * height;
        
        args = {
          .width = width,
          .height = height,
          .dx = 0.5f,
          .dy = 0.5f,
          .x0 = 0.0f,
          .y0 = 0.0f,
          .sigma = 1.0f,
          .kx = 0.0f,
          .ky = 0.0f,
          .amplitude = 1.0f
        };

        ASSERT_CUDA_SUCCESS(cudaMalloc(&d_psi, numElements * sizeof(cuFloatComplex)));
    }
    
    void TearDown() override {
        if (d_psi) {
            cudaFree(d_psi);
            d_psi = nullptr;
        }
        cudaDeviceReset(); 
    }

    void uploadData(const std::vector<cuFloatComplex>& h_data) {
        ASSERT_EQ(h_data.size(), numElements) << "Host data size mismatch";
        ASSERT_CUDA_SUCCESS(cudaMemcpy(d_psi, h_data.data(), 
                                       numElements * sizeof(cuFloatComplex), 
                                       cudaMemcpyHostToDevice));
    }

    void downloadData(std::vector<cuFloatComplex>& h_data) {
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

    for(auto& val : h_psi) {
        val = make_cuFloatComplex(dis(gen), dis(gen));
    }

    uploadData(h_psi);

    dim3 block(128);
    dim3 grid((numElements + block.x - 1) / block.x);
    
    normalizePsi(d_psi, block, grid, args);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    downloadData(h_psi);

    float totalProbability = 0.0f;
    for (const auto& val : h_psi) {
        float magSq = (val.x * val.x) + (val.y * val.y);
        totalProbability += magSq;
    }
    
    totalProbability *= (args.dx * args.dy);

    EXPECT_NEAR(totalProbability, 1.0f, 1e-4);
}

TEST_F(WavefunctionTest, HandlesZeroWavefunctionGracefully) {
    std::vector<cuFloatComplex> h_psi(numElements);
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

    for(auto& val : h_psi) {
        val = make_cuFloatComplex(0, 0);
    }

    uploadData(h_psi);

    dim3 block(128);
    dim3 grid((numElements + block.x - 1) / block.x);
    
    normalizePsi(d_psi, block, grid, args);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    downloadData(h_psi);

    float totalProbability = 0.0f;
    for (const auto& val : h_psi) {
        float magSq = (val.x * val.x) + (val.y * val.y);
        totalProbability += magSq;
    }
    
    totalProbability *= (args.dx * args.dy);

    EXPECT_NEAR(totalProbability, 0, 1e-4);

}
