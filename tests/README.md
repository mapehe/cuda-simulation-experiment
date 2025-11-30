# Simulation Snapshot Tests

This directory contains **snapshot tests** designed to validate the correctness
of the CUDA physical simulations.

## Purpose

The primary goal of these tests is to ensure **regression safety**. As we
refactor, optimize, or modify the CUDA kernels and host code, we need to
guarantee that the physical results of the simulation remain consistent.

These tests serve as a safeguard to verify that:
* Refactoring code structure does not alter simulation logic.
* Performance optimizations do not introduce calculation errors.
* The simulation remains deterministic for fixed inputs.

## How It Works

Snapshot testing works by comparing the current output of a simulation against
a "golden" reference file (the snapshot) stored in this repository.

1.  **Execution**: The test harness runs a specific simulation scenario.
2.  **Capture**: The simulation state.
3.  **Comparison**: This output is compared against the stored snapshot.
    * **Pass**: The output matches the snapshot (within floating-point tolerance).
    * **Fail**: The output differs, indicating a change in simulation behavior.
