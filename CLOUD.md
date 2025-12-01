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

### 1. Provision

Spin up a CUDA-enabled VM.
```bash
./scripts/cloud/create_cuda_instance.sh
```

### 2. Connect

SSH into the machine and you will be prompted to install the NVIDIA drivers.
_Note: Wait a few minutes after creation._
```bash
./scripts/cloud/ssh_instance.sh
```

### 3. Run

Compile and execute the solver on the remote GPU with the following script.

If you want to get the resulting binary data to cloud bucket, add the
`--upload` flag.

If you want to see the resulting `.mp4` add the `--upload-video` flag.
```bash
./scripts/cloud/compile_and_run.sh
./scripts/cloud/compile_and_run.sh --upload
./scripts/cloud/compile_and_run.sh --upload-video
```
### 4. Develop

When you run
```bash
./scripts/cloud/compile_and_run.sh
```
any changes (code or configuration) you make will be `rsync`ed to the remote VM
before the project is compiled and executed.

#### Clear cache (optional)

Sometimes the remote ends up in an invalid state. You can use
```bash
./scripts/cloud/clean_instance.sh
```
to clear the build cache and start from scratch.

### 5. Teardown

Important: Delete the instance when finished to avoid unnecessary billing.
```bash
./scripts/cloud/compile_and_run.sh
```
