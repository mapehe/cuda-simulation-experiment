# Cloud Environment

This project includes scripts to streamline running high-resolution simulations
on Google Cloud Platform (GCP) GPU instances.

## 1. Configuration
Ensure you have the `gcloud` CLI installed and authorized with permissions to
create GPU-enabled Compute Engine instances.

Create an `.env` file with the following contents.

```
export USER="your-local-username"
export PROJECT_ID="your-google-cloud-project-name"
export SERVICE_ACCOUNT="your-google-cloud-service-account"
export ZONE="us-east1-d"
export REGION="us-east1
export STORAGE_BUCKET="your-storage-bucket"
```

## 2. Workflow

### 1. Initialize

Load your configuration.
```bash
source .env
```

### 2. Provision

Spin up a CUDA-enabled VM.
```bash
./scripts/cloud/create_cuda_instance.sh
```

### 3. Connect

SSH into the machine and you will be prompted to install the NVIDIA drivers.
_Note: Wait a few minutes after creation._
```bash
./scripts/cloud/ssh_instance.sh
```

### 4. Run

Compile and execute the solver on the remote GPU. This will also upload your
results to a bucket.
```bash
./scripts/cloud/compile_and_run.sh
```

### 5. Teardown

Important: Delete the instance when finished to avoid unnecessary billing.
```bash
./scripts/cloud/compile_and_run.sh
```

## 3. Export into a bucket

The following command will visualize your simulation as an `.mp4` that is uploaded
into your Google Cloud storage bucket.

```bash
./scripts/cloud/compile_and_run.sh --upload-video
```

If you want the raw binary data instead, run.

```bash
./scripts/cloud/compile_and_run.sh --upload
```
