# Source Directory (`src`)

This directory contains the core source code for the project, separated into host-side engine logic and device-side CUDA kernels.

## Directories

### `engine/`

This directory contains the host-side orchestration logic. It houses subclasses of the `template <typename T> class ComputeEngine`.

These classes are responsible for the high-level management of the simulation, specifically:
* **Memory Management:** Handling memory allocation and data transfer between the host (CPU) and device (GPU).
* **Simulation Steps:** Managing the main loop, time-stepping, and synchronization.
* **Output:** Collecting results and handling file I/O or data export.

### `kernel/`

This directory contains the device-side implementation.

* It holds the `.cu` files containing the **CUDA kernels** (`__global__` functions) required to execute the parallelized physics calculations on the GPU.