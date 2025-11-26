# CUDA Analog Gravity Experiment

## The Goal
To simulate an "event horizon" in a Bose-Einstein Condensate (BEC) using
parallel spectral methods. The aim is to observe Hawking-like phonon emission
where the flow velocity exceeds the speed of sound using the dimensionless
Gross-Pitaevskii Equation (GPE):

$$i \frac{\partial \Psi}{\partial \tau} = \left( -\frac{1}{2} \nabla^2 + V +
\kappa |\Psi|^2 \right) \Psi$$

⚠️ **Experimental:** Please be aware that this repo is under active development.

## Cloud Environment

This project includes scripts to streamline running high-resolution simulations
on Google Cloud Platform (GCP) GPU instances.

### 1. Configuration
Ensure you have the `gcloud` CLI installed and authorized with permissions to
create GPU-enabled Compute Engine instances.

Create an `.env` file with the following contents.

```
export USER="your-local-username"
export PROJECT_ID="your-google-cloud-project-name"
export SERVICE_ACCOUNT="your-google-cloud-service-account"
export ZONE="us-east1-d"
export REGION="us-east1"
```

### 2. Workflow

**1. Initialize**
Load your configuration.
```bash
source .env
```

**2. Provision**
Spin up a CUDA-enabled VM.
```bash
./scripts/local/create_cuda_instance.sh
```

**3. Connect**
SSH into the machine and you will be prompted to install the NVIDIA drivers.
_Note: Wait a few minutes after creation._
```bash
./scripts/local/ssh_instance.sh
```

**4. Run**
Compile and execute the solver on the remote GPU.
```bash
./scripts/local/compile_and_run.sh
```

**5. Teardown**
*Important:* Delete the instance when finished to avoid unnecessary billing.
```bash
./scripts/local/compile_and_run.sh
```
