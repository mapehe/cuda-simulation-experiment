#include "engine/grossPitaevskii.cuh"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class GrossPitaevskiiEngineTest : public ::testing::Test,
                                  public GrossPitaevskiiEngine {
public:
  static Params createTestParams() {
    return Params{.output = "output.json",
                  .simulationMode = SimulationMode::GrossPitaevskii,
                  .test = {},
                  .grossPitaevskii = {.iterations = 8192,
                                      .gridWidth = 512,
                                      .gridHeight = 512,
                                      .threadsPerBlockX = 32,
                                      .threadsPerBlockY = 32,
                                      .downloadFrequency = 32,
                                      .L = 1.0f,
                                      .sigma = 0.1f,
                                      .x0 = 0.15f,
                                      .y0 = 0.15f,
                                      .kx = 0.0f,
                                      .ky = 0.0f,
                                      .amp = 1.0f,
                                      .omega = 0.0f,
                                      .trapStr = 1e3f,
                                      .dt = 6e-7f,
                                      .g = 10e3f,
                                      .V_bias = 0.0f,
                                      .r_0 = 0.0f,
                                      .sigma2 = 0.0f,
                                      .absorbStrength = 0.0f,
                                      .absorbWidth = 0.0f}};
  }

  GrossPitaevskiiEngineTest() : GrossPitaevskiiEngine(createTestParams()) {}

  void SetUp() override {}

  void TearDown() override {}

  void injectDeviceData(float real, float imag) {
    size_t size =
        params.grossPitaevskii.gridWidth * params.grossPitaevskii.gridHeight;
    std::vector<cuFloatComplex> host_data(size);

    for (auto &val : host_data) {
      val = make_cuFloatComplex(real, imag);
    }

    cudaMemcpy(this->d_psi, host_data.data(), size * sizeof(cuFloatComplex),
               cudaMemcpyHostToDevice);
  }
};

TEST_F(GrossPitaevskiiEngineTest, AppendFrame_CopiesCorrectDataFromDevice) {
  float expected_r = 1.0f;
  float expected_i = 2.0f;
  injectDeviceData(expected_r, expected_i);

  std::vector<cuFloatComplex> history;

  this->appendFrame(history);

  size_t expected_elements =
      params.grossPitaevskii.gridWidth * params.grossPitaevskii.gridHeight;
  ASSERT_EQ(history.size(), expected_elements);

  EXPECT_FLOAT_EQ(history[0].x, expected_r);
  EXPECT_FLOAT_EQ(history[0].y, expected_i);
  EXPECT_FLOAT_EQ(history[expected_elements / 2].x, expected_r);
  EXPECT_FLOAT_EQ(history[expected_elements - 1].x, expected_r);
}

TEST_F(GrossPitaevskiiEngineTest,
       ConstructorInitializesNormalizedWavefunction) {
  size_t width = params.grossPitaevskii.gridWidth;
  size_t height = params.grossPitaevskii.gridHeight;
  size_t num_elements = width * height;

  float L_x = params.grossPitaevskii.L;
  float L_y = params.grossPitaevskii.L;

  float dx = L_x / (float)width;
  float dy = L_y / (float)height;
  float cell_area = dx * dy;

  std::vector<cuFloatComplex> host_psi(num_elements);

  cudaMemcpy(host_psi.data(), this->d_psi,
             num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

  float total_probability = 0.0f;
  for (const auto &val : host_psi) {
    float abs_sq = (val.x * val.x) + (val.y * val.y);
    total_probability += abs_sq;
  }

  total_probability *= cell_area;

  EXPECT_NEAR(total_probability, 1.0f, 1e-4f)
      << "Initial wavefunction is not normalized!";
}

TEST_F(GrossPitaevskiiEngineTest, SolveStep_ConservesWavefunctionNorm) {
  size_t width = params.grossPitaevskii.gridWidth;
  size_t height = params.grossPitaevskii.gridHeight;
  size_t num_elements = width * height;

  float L_x = params.grossPitaevskii.L;
  float L_y = params.grossPitaevskii.L;

  float dx = L_x / (float)width;
  float dy = L_y / (float)height;
  float cell_area = dx * dy;

  std::vector<cuFloatComplex> host_psi_0(num_elements);
  std::vector<cuFloatComplex> host_psi(num_elements);

  cudaMemcpy(host_psi_0.data(), this->d_psi,
             num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 100; i++) {
    this->solveStep(i);
  }

  cudaMemcpy(host_psi.data(), this->d_psi,
             num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

  float total_probability = 0.0f;
  float total_diff_sq = 0.0f;

  for (size_t i = 0; i < num_elements; ++i) {
    float abs_sq =
        (host_psi[i].x * host_psi[i].x) + (host_psi[i].y * host_psi[i].y);
    total_probability += abs_sq;

    float diff_real = host_psi[i].x - host_psi_0[i].x;
    float diff_imag = host_psi[i].y - host_psi_0[i].y;
    total_diff_sq += (diff_real * diff_real) + (diff_imag * diff_imag);
  }

  total_probability *= cell_area;

  EXPECT_NEAR(total_probability, 1.0f, 1e-4f)
      << "Unitary violation: Total probability != 1 after evolution.";

  EXPECT_GT(total_diff_sq, 1e-6f) << "Static violation: The wavefunction is "
                                     "identical to the initial state.";
}

TEST_F(GrossPitaevskiiEngineTest, SolveStep_ConservesEnergy) {
  float tolerance_percent = 0.1f;
  float tolerance_fraction = tolerance_percent / 100.0f;

  size_t width = params.grossPitaevskii.gridWidth;
  size_t height = params.grossPitaevskii.gridHeight;
  size_t num_elements = width * height;

  float dx = params.grossPitaevskii.L / (float)width;
  float dy = params.grossPitaevskii.L / (float)height;
  float cell_area = dx * dy;
  float g_interaction = params.grossPitaevskii.g; // Interaction strength

  std::vector<cuFloatComplex> host_psi_0(num_elements);
  std::vector<cuFloatComplex> host_psi_final(num_elements);

  auto calculate_energy = [&](const std::vector<cuFloatComplex> &psi) -> float {
    float total_energy = 0.0f;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = y * width + x;

        cuFloatComplex val = psi[idx];
        float abs_sq = (val.x * val.x) + (val.y * val.y);

        float E_interaction = 0.5f * g_interaction * (abs_sq * abs_sq);

        int idx_l = y * width + ((x - 1 + width) % width);   // Left
        int idx_r = y * width + ((x + 1) % width);           // Right
        int idx_u = ((y - 1 + height) % height) * width + x; // Up
        int idx_d = ((y + 1) % height) * width + x;          // Down

        cuFloatComplex sum_neighbors;
        sum_neighbors.x =
            psi[idx_l].x + psi[idx_r].x + psi[idx_u].x + psi[idx_d].x;
        sum_neighbors.y =
            psi[idx_l].y + psi[idx_r].y + psi[idx_u].y + psi[idx_d].y;

        cuFloatComplex laplacian;
        laplacian.x = (sum_neighbors.x - 4.0f * val.x) / (dx * dx);
        laplacian.y = (sum_neighbors.y - 4.0f * val.y) / (dx * dx);

        float real_psi_conj_times_lap =
            (val.x * laplacian.x) + (val.y * laplacian.y);
        float E_kinetic = -0.5f * real_psi_conj_times_lap;

        total_energy += (E_kinetic + E_interaction);
      }
    }
    return total_energy * cell_area;
  };

  cudaMemcpy(host_psi_0.data(), this->d_psi,
             num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  float energy_initial = calculate_energy(host_psi_0);

  for (int i = 0; i < 20; i++) {
    this->solveStep(i);
  }

  cudaMemcpy(host_psi_final.data(), this->d_psi,
             num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  float energy_final = calculate_energy(host_psi_final);

  float energy_diff = std::abs(energy_final - energy_initial);
  float relative_error = energy_diff / std::abs(energy_initial);

  EXPECT_LT(relative_error, tolerance_fraction)
      << "Energy conservation violation > " << tolerance_percent << "%"
      << "\nInitial E: " << energy_initial << "\nFinal E:   " << energy_final
      << "\nRel Error: " << relative_error;
}
