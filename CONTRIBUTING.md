# Contributing

Thank you for your interest in this project.

## Project Status: Early Stage

Please note that this repository is currently in an early experimental stage.
The codebase, including the CUDA kernels and cloud automation scripts, is
subject to rapid iteration and significant architectural changes.

## How to Get Involved

Because the project is moving fast, communication is vital to avoid wasted
effort.

### 1. Discuss Ideas First
If you have a feature idea, a major refactor in mind, or simply want to discuss
the simulation logic, it is best to reach out directly before writing any code.

Email: matias@bytesauna.com

### 2. Check Issues
If you found a bug or are looking for specific tasks to work on, please check
the Issues tab on GitHub.

## Submitting Changes

If we have discussed an idea or you are fixing an issue:

1. Fork the repository and create a branch.
2. Format your code: Ensure your changes adhere to the project style by running
   the utility script:
   ```bash
   ./scripts/util/format.sh
   ```
3. Run tests: Verify that your changes do not break existing functionality. You
   can run tests locally or use the cloud helper script:
   ```bash
   ./scripts/cloud/test.sh
   ```
4. Submit a Pull Request describing your changes.
