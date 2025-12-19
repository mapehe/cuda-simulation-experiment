# FluxLab: High-Performance Computational Physics Template

FluxLab is a modular CUDA C++ project template designed to bridge the gap
between rigorous computational physics and modern software engineering. It
provides a portable development environment that allows researchers to develop
and run CUDA simulations seamlessly, even without access to local NVIDIA
hardware.


## Purpose

Scientific code is frequently written as monolithic, single-use scripts that
are difficult to read, maintain, or scale. FluxLab serves as a prototype for
treating physical simulations as production-grade software. It provides a
scaffold for wrapping high-precision numerical solvers in a modular
architecture featuring rigorous simulation testing and
robust utility scripts.

As a proof of concept, the project currently implements a **ComputeEngine** for
the Gross-Pitaevskii Equation (GPE). This reference engine utilizes a parallel
spectral solver to demonstrate how high-performance physics logic can be
successfully decoupled from the infrastructure.

## Audience

FluxLab serves as a translation layer between two target groups:

* For **software engineers**, it transforms scientific simulation into a standard
  engineering challenge. By abstracting complex physics logic into encapsulated
  ComputeEngines, it exposes the project as a clean, modular C++ system. This
  allows engineers to focus on architecture, optimization, and infrastructure
  without needing deep domain expertise in the underlying governing equations.


* For **physicists** it tackles the architectural challenges often encountered
  in scientific computing. FluxLab provides a professional software scaffolding
  that automates parameter parsing, structured I/O, and simulation bookkeeping.
  This ensures high-quality, reproducible software design without requiring the
  researcher to function as a systems architect.

## Project Philosophy

1. **Unconstrained Extensibility:** The library never obstructs the researcher.
   The **ComputeEngine** abstractions are designed to be thin and transparent,
   accommodating arbitrary CUDA simulations. This ensures the framework
   supports unique or experimental physics logic without imposing restrictive
   architectural patterns.

2. **Testability**: Correctness is enforced through rigorous automated
   validation of results against reference snapshots and strictly monitoring
   conservation laws (e.g., energy, mass) to guarantee simulation fidelity.

3. **Cloud-First Accessibility**: The template decouples high-performance
   computing from local hardware ownership. Through simple but robust utility
   scripts, FluxLab allows users to seamlessly develop, build, and run
   simulations on Cloud VMs, removing the requirement for a physical NVIDIA GPU
   workstation.

## Implemented Physics: The Gross-Pitaevskii Equation

FluxLab currently implements a solver for the **Gross-Pitaevskii Equation
(GPE)**.

The GPE is used to describe Bose-Einstein Condensates (BECs) and allows for the
simulation of macroscopic quantum phenomena—such as vortex formation and
interference patterns. The time-evolution of the wavefunction $\psi$ is
governed by:

$$i \hbar \frac{\partial \psi}{\partial t} = \left( -\frac{\hbar^2}{2m}
\nabla^2 + V + g |\psi|^2 \right) \psi$$

### Role in the Template: Architectural Validation

Implementing the GPE serves as a comprehensive "vertical slice" to validate the
framework's internal plumbing. The physics requirements of the GPE act as a
stress test for the system's technical capabilities:

* **Complex State Management (SSFM):** The **Split-Step Fourier Method**
  necessitates rigorous bookkeeping. It requires rapid, iterative switching
  between Cartesian space and Frequency space (via cuFFT). This forces the
  framework to correctly manage device-side FFT plans, complex-valued memory
  buffers, and synchronization between spectral and potential evolution
  kernels.

* **ComputeEngine Verification:** The GPE acts as the reference implementation
  for how `ComputeEngine` modules should be built and tested. Because the GPE has strict
      invariants, it allows for validation against physical laws alongside
      **snapshot regression testing**. This ensures that the engine not only
      conserves physical quantities but also maintains exact numerical
      reproducibility across commits.

### No Local GPU? 

If you do not have access to a local NVIDIA workstation, you can run
simulations using a cloud instance. Refer to [CLOUD.md](CLOUD.md) for setup and
launch instructions.

Once your cloud environment is ready, execute the simulation remotely:

```bash
./scripts/cloud/compile_and_run.sh --mode grossPitaevskii --config config.json --upload-video --output example
```

You can then download the resulting video using the gcloud CLI:

```bash
gcloud storage cp gs://$STORAGE_BUCKET/example.mp4 .
```

## Extending

### Building Your Own Engine

To implement your own physics logic, refer to the `GrossPitaevskiiEngine`
class. This implementation serves as an example of how to inherit from the base
`ComputeEngine`class, manage custom memory, and interface with the GPU.

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Author & Contact

FluxLab is developed by Matias Heikkilä (matias@bytesauna.com).

* Bug Reports & Features: Please use the [Issue
  Tracker](https://github.com/mapehe/FluxLab/issues).

* Collaboration & Inquiries: For questions regarding custom ComputeEngines, HPC
  consulting, or research collaborations, reach out to matias@bytesauna.com.
