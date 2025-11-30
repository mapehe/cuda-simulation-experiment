# Scripts Directory (`scripts`)

This directory contains the automation and utility scripts for the project,
separated into infrastructure management, data processing, and code
maintenance.

## Directories

### `cloud/`

This directory contains the Google Cloud Platform (GCP) management logic. It
houses shell wrappers for the `gcloud` CLI.

These scripts are responsible for the lifecycle of the remote simulation
environment, specifically:
* **Lifecycle Management:** Provisioning, configuring, and tearing down GPU
  instances to manage costs.
* **Deployment:** Synchronizing local source code to the remote instance and
  triggering compilation/execution.
* **Interaction:** Simplifying SSH connections and file transfers (`rsync`).

### `postprocess/`

This directory contains the data analysis tools. It holds Python scripts
responsible for **parsing** raw binary outputs from the simulation and
converting them into structured formats for analysis or visualization.

### `util/`

This directory contains general maintenance utilities. It includes helper
scripts such as auto-formatting .
